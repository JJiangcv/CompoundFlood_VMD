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

#----------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------
import os
import numpy as np
import pandas as pd

#----------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------
def load_source_data(file_path, date_col=None, value_col=None, dayfirst=True):
    """
    Load source data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        date_col (str): Name of the date column. Defaults to 'timestamp'.   
        value_col (str): Name of the value column. Defaults to 'value'.
        dayfirst (bool): Whether the date format is day-first. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the source data.
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
            # 再随便找一个名字里含 time/date 的列
            for c in df.columns:
                lc = c.lower()
                if "time" in lc or "date" in lc:
                    date_col = c; break
    if date_col is None:
        raise ValueError("未找到时间列，请传入 date_col。")

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
        raise ValueError("未找到数值列，请传入 value_col。")

    # 3) parse date and value columns
    out = pd.DataFrame({
        "timedate": pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce"),
        "value":    pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna().sort_values("timedate").reset_index(drop=True)

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
        s = (
            df.set_index("timedate")["value"].groupby("timedate").mean().rename(name)
        )
        series_list.append(s)
    merged_df = pd.concat(series_list, axis=1, join='inner').sort_index()
    if rename_map:
        merged_df = merged_df.rename(columns=rename_map)  
    return merged_df.reset_index()
