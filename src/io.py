#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: io.py
Created: 2025-10-13
Author: Jinghua Jiang <jinghua.jiang21@gmail.com>
Project: Project Name

Description:
    Input/Output utilities for Compound Flood VMD analysis.

License: MIT
"""

__author__ = "Jinghua Jiang"
__email__ = "jinghua.jiang21@gmail.com"
__version__ = "0.1.0"


import os
import pandas as pd
from typing import Dict

#----------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------
def load_source_data(file_path, date_col=None, value_col=None, dayfirst=True, rename_value_to=None):
    """
    Load source data from a CSV file.

    Parameters:
        file_path (str or Path): Path to the CSV file.
        date_col (str): Name of the date column. Defaults to auto-detect.
        value_col (str): Name of the value column. Defaults to auto-detect.
        dayfirst (bool): Whether the date format is day-first. Defaults to True.
        rename_value_to (str): If provided, rename the value column to this name.
                              Useful for avoiding merge conflicts.
    Returns:
        pd.DataFrame: DataFrame with columns ['timedate', 'value'] or 
                      ['timedate', <rename_value_to>].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    # 1) select date column
    if date_col is None:
        for c in ("timedate", "datetime", "date", "timestamp"):
            if c in df.columns:
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                lc = c.lower()
                if "time" in lc or "date" in lc:
                    date_col = c; break
    if date_col is None:
        raise ValueError("Could not find date column. Please specify date_col.")

    # 2) select value column
    if value_col is None:
        # try common names
        for c in ("value","val","wl","flow","tide","rain","precipitation","waterlevel"):
            if c in df.columns:
                value_col = c; break
        if value_col is None:
            # then try numeric columns
            for c in df.columns:
                if c == date_col: continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    value_col = c; break
    if value_col is None:
        raise ValueError("Could not find value column. Please specify value_col.")

    # 3) parse date and value columns
    out = pd.DataFrame({
        "timedate": pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce"),
        "value":    pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna().sort_values("timedate").reset_index(drop=True)

    # 4) Rename if requested
    if rename_value_to:
        out = out.rename(columns={"value": rename_value_to})
    
    return out

def merge_source_data(ds, rename_map=None):
    """
    Merge multiple source data DataFrames.

    Parameters:
        ds:dict: Dictionary of DataFrames to merge.
        rename_map:dict: Optional mapping to rename columns after merging.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    series_list = []
    for name, df in ds.items():
        # Check if df has 'value' column (output from load_source_data)
        if "value" in df.columns:
            s = (
                df.set_index("timedate")["value"]
                .groupby("timedate").mean()
                .rename(name)
            )
        else:
            # Use the non-timedate column
            value_col = [c for c in df.columns if c != "timedate"][0]
            s = (
                df.set_index("timedate")[value_col]
                .groupby("timedate").mean()
                .rename(name)
            )
        series_list.append(s)
    
    merged_df = pd.concat(series_list, axis=1, join='inner').sort_index()
    
    if rename_map:
        merged_df = merged_df.rename(columns=rename_map)
    
    return merged_df.reset_index()

def prepare_for_plotting(data_dict: Dict[str, pd.DataFrame],
                        include_wl: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for the plotting functions which expect 'value' column.
    
    Parameters:
        data_dict (dict): Dictionary from load_compound_flood_data.
        include_wl (bool): Whether to include water level as "water level".
    
    Returns:
        dict: Dictionary ready for plot functions.
    """
    plot_dict = {}
    
    # Rename columns back to 'value' for plotting
    if "rain" in data_dict:
        plot_dict["rain"] = data_dict["rain"].rename(columns={"rain": "value"})
    
    if "tide" in data_dict:
        plot_dict["tide"] = data_dict["tide"].rename(columns={"tide": "value"})
    
    if "flow" in data_dict:
        plot_dict["flow"] = data_dict["flow"].rename(columns={"flow": "value"})
    
    if include_wl and "wl" in data_dict:
        plot_dict["water level"] = data_dict["wl"].rename(columns={"wl": "value"})
    
    return plot_dict