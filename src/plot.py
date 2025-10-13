#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: plot.py
Created: 2025-10-13
Author: Jinghua Jiang <jinghua.jiang21@gmail.com>
Project: Compound Flood VMD

Description:
    plotting utilities for Compound Flood VMD analysis.

License: MIT
"""

__author__ = "Jinghua Jiang"
__email__ = "jinghua.jiang21@gmail.com"
__version__ = "0.1.0"

# src/plots.py
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import pandas as pd
import string
from .stats import monthly_clim, norm01

def plot_climatology_plus_normalised(
    data,                   # dict or wide DataFrame
    out_path: str,
    value_col: str = "value",
    include_normalised: bool = True,
    time_col: str = "timedate",
    cols: list | None = None,
    title_map: dict | None = None,
    short_map: dict | None = None,
    auto_letter: bool = True,        
    start_letter: str = "a" 
):
    """
    Plot monthly climatology (mean ± 1 SD) for multiple variables, plus an optional
    bottom panel comparing their normalised monthly means.
    Parameters:
        data (dict or pd.DataFrame): If dict, keys are titles, values are (short_label, df_two_cols).
                                     If DataFrame, it is wide format with time_col and multiple value columns.
        out_path (str): Path to save the output PNG file.
        value_col (str): Name of the value column in the input DataFrames. Defaults to 'value'.
        include_normalised (bool): Whether to include the bottom normalised comparison panel. Defaults to True.
        time_col (str): Name of the time column in the input DataFrames. Defaults to 'timedate'.
        cols (list): If data is a DataFrame, list of columns to plot. Defaults to None (all numeric columns except time_col).
        title_map (dict): Optional mapping from column names to full titles for top panels.
        short_map (dict): Optional mapping from column names to short labels for legend in bottom panel.
        auto_letter (bool): Whether to automatically label subplots with letters. Defaults to True.
        start_letter (str): Starting letter for subplot labels if auto_letter is True. Defaults to 'a'.
    Returns:
        str: Path to the saved PNG file.
    """

    # prepare items to plot
    if isinstance(data, dict):
        items = list(data.items())
        # check each df has required columns
        for k, (short, df) in items:
            if time_col not in df.columns:  # check time_col
                assert "timedate" in df.columns and value_col in df.columns, \
                    f"DataFrame for '{k}' must contains 'timedate' and '{value_col}' columns."
    elif isinstance(data, pd.DataFrame):
        df_wide = data
        if time_col not in df_wide.columns:
            raise ValueError(f"time_col '{time_col}' is not in DataFrame columns.")
        # 选择变量列
        if cols is None:
            cols = [c for c in df_wide.columns if c != time_col and pd.api.types.is_numeric_dtype(df_wide[c])]
        if not cols:
            raise ValueError("data does not contain any numeric columns to plot.")

        items = []
        for c in cols:
            title = title_map.get(c, c) if title_map else c
            short = short_map.get(c, c) if short_map else c
            df_one = (
                df_wide[[time_col, c]]
                .rename(columns={time_col: "timedate", c: value_col})
                .dropna()
            )
            items.append((title, (short, df_one)))
    else:
        raise TypeError("data must be a dict or a pandas DataFrame.")

    n = len(items)
    if n == 0:
        raise ValueError("no data to plot.")

    # setup figure layout
    ncols = 2 if n <= 4 else (3 if n <= 9 else 4)
    nrows_top = math.ceil(n / ncols)
    add_bottom = include_normalised and n >= 2
    total_rows = nrows_top + (1 if add_bottom else 0)

    fig = plt.figure(figsize=(4.5 * ncols, 3.2 * total_rows))
    gs = GridSpec(total_rows, ncols, figure=fig, height_ratios=[1]*nrows_top + ([1] if add_bottom else []))

    letters = list(string.ascii_lowercase)
    start_idx = letters.index(start_letter.lower())

    # top panels: monthly climatology
    for i, (title, (short, df)) in enumerate(items):
        ax = fig.add_subplot(gs[divmod(i, ncols)])
        clim = monthly_clim(df, value_col=value_col)  # index=1..12, cols=['mean','std']
        ax.plot(clim.index, clim["mean"], linewidth=2, label="mean daily by month")
        ax.fill_between(clim.index, clim["mean"] - clim["std"], clim["mean"] + clim["std"],
                        color="lightgrey", alpha=0.7, label="±1 SD")
        if auto_letter:
            prefix = f"({letters[start_idx + i]}) "
            t = title.lstrip()
            if not (t.startswith("(") and ")" in t[:5]):
                title = prefix + title  # avoid double parentheses
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 12); ax.set_xlabel("Month"); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc="upper left", frameon=False)

    # bottom panel: normalised comparison
    if add_bottom:
        ax = fig.add_subplot(gs[-1, :])
        for title, (short, df) in items:
            clim_mean = monthly_clim(df, value_col=value_col)["mean"]
            ax.plot(clim_mean.index, norm01(clim_mean), label=short)
        bottom_title = "Normalised Monthly Climatology"
        if auto_letter:
            prefix = f"({letters[start_idx + n]}) "
            bottom_title = prefix + bottom_title
        ax.set_title(bottom_title)
        ax.set_xlim(1, 12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Normalised (0-1)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(n, 6), frameon=False)

    plt.tight_layout(rect=[0, 0.08 if add_bottom else 0, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600)
    plt.close()
    return out_path