#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: lag.py
Created: 2025-10-13
Author: Jinghua Jiang <jinghua.jiang21@gmail.com>
Project: Compound Flood VMD

Description:
    time lag analysis utilities for Compound Flood VMD analysis.

License: MIT
"""

__author__ = "Jinghua Jiang"
__email__ = "jinghua.jiang21@gmail.com"
__version__ = "0.1.0"

# src/lag.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional
from scipy.stats import spearmanr, pearsonr, kendalltau

LagMethod = Literal["pearson", "spearman", "kendall"]

__all__ = [
    "LagResult",
    "estimate_forward_lags",
    "apply_lags and align",
    "pick_pot_events_on_df",
    "corrbest_positive_fixedwindow"
]


@dataclass
class LagResult:
    target: str
    best_lag: Dict[str, int]  # variable -> lag in days 
    coor_at_best: Dict[str, float]  # variable -> correlation at best lag
    coor_curve: Dict[str, pd.Series]  # variable -> correlation curve (lag vs correlation)

#----------------------------------------------------------------------
# method 1: forward CCF (driver leads target)
#----------------------------------------------------------------------

def _forward_ccf_shift(
    driver: pd.Series,
    target: pd.Series,
    max_lag: int,
    method: LagMethod = "spearman",
    min_overlap: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation function (CCF) between two 1D series x and y.
    Parameters:
        driver: pd.Series of driver variable (e.g., rainfall)
        target: pd.Series of target variable (e.g., water level)
        max_lag: maximum lag (in number of samples) to consider in both directions.
        method: correlation method, one of "pearson", "spearman", "kendall".
        min_overlap: minimum number of overlapping points required to compute correlation.
    Returns:
        lags: np.ndarray of lag values (from -max_lag to +max_l
    """
    lags = np.arange(0, max_lag + 1, dtype=int)
    corrs = np.full_like(lags, np.nan, dtype=float)

    x = pd.to_numeric(driver, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")

    for i, L in enumerate(lags):
        x = driver.shift(L)  
        y = target
        df = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < min_overlap:
            continue
        corrs[i] = df.iloc[:, 0].corr(df.iloc[:, 1], method=method)
    return lags, corrs

    
def estimate_forward_lags(
    df: pd.DataFrame,
    target: str,
    drivers: List[str],
    max_lag_days: int = 30,
    method: LagMethod = "spearman",
    min_overlap: int = 10,
    positive_only: bool = True,   # True=只在 r>0 中选最大
    time_col: str = "timedate",
) -> LagResult:
    """
    Perform lag analysis between a target variable and multiple driver variables.
    Parameters:
        df: pd.DataFrame containing time series data.
        target_col: Name of the target variable column.
        driver_cols: List of names of driver variable columns.
        max_lag_days: Maximum lag (in number of samples) to consider. Defaults to 30.
        method: Correlation method, one of "pearson", "spearman", "kendall". Defaults to "spearman".
        min_overlap: Minimum number of overlapping points required to compute correlation. Defaults to 10.
        positive_only: If True, only consider positive correlations when selecting best lag. Defaults to False.
        time_col: Name of the time column. Defaults to "timedate".
    Returns:
        LagResult containing best lags, correlations at best lags, and correlation curves.
    """
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True).set_index(time_col)
    else:
        df = df.sort_index()

    y = df[target]
    best_lags: Dict[str, int] = {}
    coor_at_bests: Dict[str, float] = {}
    coor_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  

    for driver in drivers:  
        lags, corrs = _forward_ccf_shift(df[driver], y, max_lag_days, method, min_overlap)
        coor_curves[driver] = (lags, corrs)

        c_use = corrs.copy()
        if positive_only:
            c_use[c_use < 0] = np.nan
        
        if np.isfinite(c_use).any():
            idx_max = int(np.nanargmax(c_use))
            best_lags[driver] = int(lags[idx_max])
            coor_at_bests[driver] = float(corrs[idx_max])
        else:
            best_lags[driver] = 0
            coor_at_bests[driver] = np.nan  
    return LagResult(target=target, best_lag=best_lags, coor_at_best=coor_at_bests, coor_curve=coor_curves)

def apply_lags_and_align(
    df_wide: pd.DataFrame,
    lags: Dict[str, int],
    time_col: str = "timedate",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Apply specified lags to columns in a wide DataFrame and align them by time. 
    Parameters:
        df_wide: pd.DataFrame in wide format with a time column and multiple value columns.
        lags: Dict mapping column names to lag values (in number of samples). Positive lag means shifting forward.
        time_col: Name of the time column. Defaults to "timedate".
        dropna: Whether to drop rows with NaN values after alignment. Defaults to True.
    Returns:
        pd.DataFrame with lagged and aligned columns.
    """
    if time_col in df_wide.columns:
        df = df_wide.sort_values(time_col).reset_index(drop=True).set_index(time_col)
    else:
        df = df_wide.sort_index()

    out = {col: (df[col].shift(lags[col]) if col in lags else df[col]) for col in df.columns}
    aligned = pd.DataFrame(out, index=df.index).reset_index().rename(columns={df.index.name or time_col: time_col})
    return aligned.dropna() if dropna else aligned

#----------------------------------------------------------------------
# method 2: pick POT events on target, then find driver values at lagged times
#---------------------------------------------------------------------- 

def pick_pot_events_on_df(
    df: pd.DataFrame,
    time_col: str,
    val_col: str,
    q: float = 0.90,
    min_sep_days: int = 3,
) -> Tuple[List[pd.Timestamp], float]:
    """
    Pick Peaks-Over-Threshold (POT) events from a DataFrame based on a specified quantile threshold.
    Parameters:
        df: pd.DataFrame containing time series data.
        time_col: Name of the time column.
        val_col: Name of the value column to analyze.
        q: Quantile threshold for selecting peaks. Defaults to 0.90.
        min_sep_days: Minimum separation (in days) between consecutive peaks. Defaults to 3.
    Returns:
        Tuple containing a list of timestamps of selected peaks and the threshold value.
    """
    thr = float(np.nanpercentile(df[val_col].values, q * 100)  )
    cand = (
        df.loc[df[val_col] >= thr, [time_col, val_col]]
          .sort_values(val_col, ascending=False)
          .reset_index(drop=True)               
    )

    events: List[pd.Timestamp] = []
    min_sep = pd.Timedelta(days=min_sep_days)
    last = pd.Timestamp.min

    i = 0
    while i < len(cand):
        t0 = cand.loc[i, time_col]
        if t0 - last >= min_sep:
            win = cand[(cand[time_col] >= t0 - min_sep) & (cand[time_col] <= t0 + min_sep)]
            j = win[val_col].idxmax()
            events.append(cand.loc[j, time_col])
            last = cand.loc[j, time_col]
            i = j + 1
        else:
            i += 1
    return events, thr

def corrbest_positive_fixedwindow(
    df: pd.DataFrame,
    time_col: str,
    anchor_col: str, 
    target_col: str, 
    q: float = 0.90,
    min_sep_days: int = 3,
    max_lag: int = 30,
    fixed_window_days: int = 40,
    min_overlap: int = 10,
    return_stats: bool = True,
):
    """
    For each peak event in the anchor variable, compute correlation between the target variable
    at the event time and the anchor variable at lagged times within a fixed window.
    Parameters:
        df: pd.DataFrame containing time series data.
        time_col: Name of the time column.
        anchor_col: Name of the anchor variable column (e.g., water level).
        target_col: Name of the target variable column (e.g., rainfall).
        q: Quantile threshold for selecting peaks in the anchor variable. Defaults to 0.90.
        min_sep_days: Minimum separation (in days) between consecutive peaks. Defaults to 3.
        max_lag: Maximum lag (in number of samples) to consider. Defaults to 30.
        fixed_window_days: Size of the fixed window (in days) around each event to consider. Defaults to 40.
        min_overlap: Minimum number of overlapping points required to compute correlation. Defaults to 10.
        return_stats: Whether to return detailed statistics including correlation curves. Defaults to True.
    Returns:
        If return_stats is True, returns a tuple (best_lag, best_corr, corr_curve).
        If return_stats is False, returns (best_lag, best_corr).
    """ 
    events, thr = pick_pot_events_on_df(df, time_col, anchor_col, q, min_sep_days)

    ts = (
        df[[time_col, anchor_col, target_col]]
            .dropna()
            .set_index(time_col)
            .sort_index(time_col)
            .asfreq('D')
    )

    rows = []
    pot_vals = ts[anchor_col][ts[anchor_col] >= thr].dropna()

    half_win = pd.Timedelta(days=fixed_window_days // 2)


    for t_event in events:
        seg = ts.loc[t_event - half_win : t_event + half_win, [anchor_col, target_col]].copy()