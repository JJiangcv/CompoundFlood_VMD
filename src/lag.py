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
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional
from scipy.stats import spearmanr, pearsonr, kendalltau

LagMethod = Literal["pearson", "spearman", "kendall"]

__all__ = [
    "LagResult",
    "estimate_forward_lags",
    "apply_lags_and_align",
    "pick_pot_events",
    "pot_forward_lag_analysis",
    "multi_pot_forward_lag_analysis"
]


@dataclass
class LagResult:
    target: str
    best_lags: Dict[str, int]  # variable -> lag in days
    corr_at_best: Dict[str, float]  # variable -> correlation at best lag
    corr_curve: Dict[str, Tuple[np.ndarray, np.ndarray]]  # variable -> correlation curve (lag vs correlation)

#----------------------------------------------------------------------
# method 1: global cross-correlation function (CCF) approach
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
        max_lag: Maximum forward lag to test.
        method: correlation method, one of "pearson", "spearman", "kendall".
        min_overlap: minimum number of overlapping points required to compute correlation.
    Returns:
        tuple: (lags array, correlations array) where lags = 0..max_lag
    """
    lags = np.arange(0, max_lag + 1, dtype=int)
    corrs = np.full_like(lags, np.nan, dtype=float)

    driver = pd.to_numeric(driver, errors="coerce")
    target = pd.to_numeric(target, errors="coerce")

    for i, L in enumerate(lags):
        x = driver.shift(L)  
        y = target
        df = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < min_overlap:
            continue
        corrs[i] = df.iloc[:, 0].corr(df.iloc[:, 1], method=method)
    return lags, corrs

    
def estimate_forward_lags(
    df_wide: pd.DataFrame,
    target: str,
    drivers: List[str],
    max_lag_days: int = 30,
    method: LagMethod = "spearman",
    min_overlap: int = 10,
    positive_only: bool = True,
    time_col: str = "timedate",
) -> LagResult:
    """
    Perform lag analysis between a target variable and multiple driver variables.
    Parameters:
        df_wide (pd.DataFrame): Wide-format DataFrame with time and variable columns.
        target (str): Name of the target variable column.
        drivers (List[str]): List of driver variable column names.
        max_lag_days (int): Maximum forward lag to search (days). Defaults to 30.
        method (LagMethod): Correlation method. Defaults to 'spearman'.
        min_overlap (int): Minimum valid pairs required. Defaults to 10.
        positive_only (bool): Only consider positive correlations. If True and all
                              correlations are ≤0, returns lag=0 with r=NaN. Defaults to True.
        time_col (str): Name of the time column. Defaults to 'timedate'.
    Returns:
        LagResult: Object containing best lags, correlations, and full curves.
    """
    if time_col in df_wide.columns:
        df = df_wide.sort_values(time_col).reset_index(drop=True).set_index(time_col)
    else:
        df = df_wide.sort_index()

    y = df[target]
    best_lags: Dict[str, int] = {}
    corr_at_best: Dict[str, float] = {}
    corr_curve: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for drv in drivers:  
        lags, corrs = _forward_ccf_shift(df[drv], y, max_lag_days, method, min_overlap)
        corr_curve[drv] = (lags, corrs)

        c_use = corrs.copy()
        if positive_only:
            c_use[c_use < 0] = np.nan
        
        if np.isfinite(c_use).any():
            idx_max = int(np.nanargmax(c_use))
            best_lags[drv] = int(lags[idx_max])
            corr_at_best[drv] = float(corrs[idx_max])
        else:
            best_lags[drv] = 0
            corr_at_best[drv] = np.nan
    return LagResult(
        target=target,
        best_lags=best_lags,
        corr_at_best=corr_at_best,
        corr_curve=corr_curve
    )

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
    aligned = pd.DataFrame(out, index=df.index).reset_index()
    aligned = aligned.rename(columns={df.index.name or time_col: time_col})
    return aligned.dropna() if dropna else aligned

#----------------------------------------------------------------------
# method 2: pick POT events on target, then find driver values at lagged times
#---------------------------------------------------------------------- 

def pick_pot_events(
    df: pd.DataFrame,
    value_col: str,
    quantile: float = 0.90,
    min_separation_days: int = 3,
    time_col: str = "timedate",
) -> Tuple[List[pd.Timestamp], float]:
    """
    Detect Peaks Over Threshold (POT) events with declustering.
    Declustering: within each min_separation_days window, only the maximum
    value is retained as the event time.
    Parameters:
        df (pd.DataFrame): DataFrame containing the time series.
        value_col (str): Name of the value column to analyze.
        quantile (float): Quantile threshold for POT (0-1). Defaults to 0.90.
        min_separation_days (int): Minimum days between independent events. Defaults to 3.
        time_col (str): Name of the time column. Defaults to 'timedate'.
        
    Returns:
        tuple: (list of event timestamps, threshold value)
    """
    threshold = float(np.nanpercentile(df[value_col].values, quantile * 100.0))

    candidates = (
        df.loc[df[value_col] > threshold, [time_col, value_col]]
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    if candidates.empty:
        return [], threshold
    
    events: List[pd.Timestamp] = []
    min_sep = pd.Timedelta(days=min_separation_days)
    last_event_time = pd.Timestamp.min

    i = 0
    while i < len(candidates):
        t0 = candidates.loc[i, time_col]
        if t0 >= last_event_time + min_sep: # Find maximum within min_separation_days window
            window = candidates[
                (candidates[time_col] >= t0) & 
                (candidates[time_col] < t0 + min_sep)
            ]
            j = window[value_col].idxmax()
            event_time = candidates.loc[j, time_col]
            events.append(event_time)
            last_event_time = event_time
            i = j + 1
        else:
            i += 1
    return events, threshold

def pot_forward_lag_analysis(
    df: pd.DataFrame,
    anchor_col: str,
    target_col: str,
    quantile: float = 0.90,
    min_separation_days: int = 3,
    max_lag: int = 30,
    fixed_window_days: int = 40,
    min_overlap: int = 10,
    positive_only: bool = True,
    time_col: str = "timedate",
    return_stats: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict]] | pd.DataFrame:
    """
    POT event-driven forward lag analysis with fixed forward window.
    
    For each POT event in the anchor (driver) variable:
    1. Define a forward window [t_event, t_event + fixed_window_days]
    2. Test forward lags L=0..max_lag (daily frequency)
    3. Find the best lag with maximum positive correlation (if positive_only=True)

    Parameters:
        df (pd.DataFrame): DataFrame with time series (must contain time_col, anchor_col, target_col).
        anchor_col (str): Driver variable (e.g., 'tide', 'flow').
        target_col (str): Response variable (e.g., 'wl' for water level).
        quantile (float): POT quantile threshold (0-1). Defaults to 0.90.
        min_separation_days (int): Minimum days between independent events. Defaults to 3.
        max_lag (int): Maximum forward lag to search (days). Defaults to 30.
        fixed_window_days (int): Length of forward window after each event (days). Defaults to 40.
        min_overlap (int): Minimum number of valid pairs required. Defaults to 10.
        positive_only (bool): Only keep lags with positive correlation. Defaults to True.
        time_col (str): Name of the time column. Defaults to 'timedate'.
        return_stats (bool): Whether to return summary statistics. Defaults to True.
    Returns:
        If return_stats=True: (event_results_df, summary_stats_dict)
        If return_stats=False: event_results_df only
    """ 
    events, threshold = pick_pot_events(
        df, anchor_col, quantile, min_separation_days, time_col
    )
    if not events:
        empty_df = pd.DataFrame()
        if return_stats:
            return empty_df, {"N_events": 0, "threshold": threshold, "quantile": quantile}
        return empty_df
    
    # Resample to daily frequency
    ts = (
        df[[time_col, anchor_col, target_col]]
        .dropna()
        .sort_values(time_col)
        .set_index(pd.to_datetime(df[time_col]).dt.normalize())
        .asfreq("D")
    )

    # Get POT values for percentile calculation
    pot_vals = ts[anchor_col][ts[anchor_col] > threshold].dropna()

    # Analyse each event
    results=[]
    for event_time in events:
        event_date = pd.to_datetime(event_time).normalize()  
        window_end = event_date + pd.Timedelta(days=fixed_window_days)
        segment = ts.loc[event_date:window_end]
        if segment.empty:
            continue    
        try:
            event_value = ts.at[event_time, anchor_col]
        except KeyError:
            event_value = np.nan

        # Calculate percentile within POT events
        if len(pot_vals) > 0 and pd.notna(event_value):
            event_percentile = float((pot_vals < event_value).mean() * 100.0)
        else:
            event_percentile = np.nan

        # Search for best forward lag
        x_all = segment[anchor_col].to_numpy()
        y_all = segment[target_col].to_numpy()
        best_lags, best_corr, best_n_pairs = np.nan, np.nan, 0
        for L in range(0, max_lag + 1):
            if L == 0:
                xs = x_all
                ys = y_all
            else:
                xs = x_all[:-L]
                ys = y_all[L:]
            valid_mask = np.isfinite(xs) & np.isfinite(ys)
            n_valid = valid_mask.sum()

            if n_valid < min_overlap:
                continue

            rho, _ = spearmanr(xs[valid_mask], ys[valid_mask])
            r = float(rho) if rho is not None else np.nan

            if not np.isnan(r):
                if positive_only and r <= 0:
                    continue
                if np.isnan(best_corr) or r > best_corr:
                    best_lags = int(L)
                    best_corr = r
                    best_n_pairs = int(n_valid)
        results.append({
            "event_date": event_time.date(),
            "threshold_quantile": quantile,
            "threshold_value": threshold,
            "event_value": float(event_value) if pd.notna(event_value) else np.nan,
            "event_percentile_in_POT": event_percentile,
            "best_lag_days": best_lags,
            "best_correlation": best_corr,
            "n_valid_pairs": best_n_pairs,
            "fixed_window_days": fixed_window_days,
            "max_lag_searched": max_lag,
            "min_separation_days": min_separation_days,
            "min_overlap_required": min_overlap,
        })
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if not return_stats:
        return results_df
    
    # Calculate summary statistics
    lag_values = results_df["best_lag_days"].dropna().to_numpy()
    corr_values = results_df["best_correlation"].dropna().to_numpy()

    if len(lag_values) > 0:
        stats = {
            "N_events": int(len(results_df)),
            "N_valid_lags": int(len(lag_values)),
            "threshold": float(threshold),
            "quantile": float(quantile),
            "median_lag": float(np.median(lag_values)),
            "IQR_lag": float(np.percentile(lag_values, 75) - np.percentile(lag_values, 25)),
            "mean_lag": float(np.mean(lag_values)),
            "std_lag": float(np.std(lag_values)),
            "median_r": float(np.median(corr_values)),
            "IQR_r": float(np.percentile(corr_values, 75) - np.percentile(corr_values, 25)),
            "mean_r": float(np.mean(corr_values)),
            "std_r": float(np.std(corr_values)),
        }
    else:
        stats = {
            "N_events": int(len(results_df)),
            "N_valid_lags": 0,
            "threshold": float(threshold),
            "quantile": float(quantile),
        }
    
    return results_df, stats

def multi_pot_forward_lag_analysis(
    df: pd.DataFrame,
    anchor_col: str,
    target_col: str,
    quantiles: List[float] = [0.90, 0.95, 0.99],
    min_separation_days: int = 3,
    max_lag: int = 30,
    fixed_window_days: int = 40,
    min_overlap: int = 10,
    positive_only: bool = True,
    time_col: str = "timedate",
) -> Tuple[Dict[float, pd.DataFrame], pd.DataFrame]:
    """
    Run POT forward lag analysis for multiple quantile thresholds.
    
    This is useful for understanding how lag relationships vary with event magnitude.
    
    Parameters:
        df (pd.DataFrame): DataFrame with time series.
        anchor_col (str): Driver variable.
        target_col (str): Response variable.
        quantiles (List[float]): List of POT quantiles to test. Defaults to [0.90, 0.95, 0.99].
        (other parameters same as pot_forward_lag_analysis)
        
    Returns:
        tuple: (event_results_dict, summary_df)
            - event_results_dict: {quantile: event_results_df} for each quantile
            - summary_df: Summary statistics across all quantiles
    """
    event_results = {}
    summary_rows = []

    for q in quantiles:
        events_df, stats = pot_forward_lag_analysis(
            df=df,
            anchor_col=anchor_col,
            target_col=target_col,
            quantile=q,
            min_separation_days=min_separation_days,
            max_lag=max_lag,
            fixed_window_days=fixed_window_days,
            min_overlap=min_overlap,
            positive_only=positive_only,
            time_col=time_col,
            return_stats=True,
        )
        event_results[q] = events_df
        summary_rows.append(stats)
    
    summary_df = pd.DataFrame(summary_rows)
    
    return event_results, summary_df

def run_pot_lag_analysis(driver_name, target_name, driver_df, target_df, 
                        quantiles, config, output_dir, verbose=True):
    """
    Run POT lag analysis and save results.
    """
    merged = pd.merge(driver_df, target_df, on="timedate").dropna()

    events_dict, summary = multi_pot_forward_lag_analysis(
        merged, driver_name, target_name, quantiles, **config
    )
    

    for q, events_df in events_dict.items():
        events_df.to_csv(output_dir / f"{driver_name}_events_q{int(q*100)}.csv")
    summary.to_csv(output_dir / f"{driver_name}_pot_summary.csv")
    
    if verbose:
        print(summary)
    
    return summary