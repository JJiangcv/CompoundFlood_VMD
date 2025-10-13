#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: stats.py
Created: 2025-10-13
Author: Jinghua Jiang <jinghua.jiang21@gmail.com>
Project: Compound Flood VMD

Description:
    Statistical analysis utilities for Compound Flood VMD analysis.

License: MIT
"""

__author__ = "Jinghua Jiang"
__email__ = "jinghua.jiang21@gmail.com"
__version__ = "0.1.0"

# src/stats.py
import pandas as pd

def monthly_clim(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """
    Based on daily data in df, calculate monthly climatology (mean and std) for each month (1..12).
    Parameters:
        df (pd.DataFrame): DataFrame containing daily data with a datetime column named 'timedate'.
        value_col (str): Name of the value column to analyze. Defaults to 'value'.  
    Returns:
        pd.DataFrame: DataFrame with index as month (1-12) and columns 'mean' and 'std'.
    """
    df = df.loc[:, ["timedate", value_col]].copy()
    df["month"] = pd.to_datetime(df["timedate"]).dt.month
    df["year"]  = pd.to_datetime(df["timedate"]).dt.year

    # calculate monthly mean for each year-month
    ym = df.groupby(["year", "month"], as_index=False)[value_col].mean()

    # calculate climatology: mean and std for each month across years
    clim = (ym.groupby("month")[value_col]
              .agg(["mean", "std"])
              .reindex(range(1, 12+1)))
    return clim

def norm01(s):
    """ Normalize a pandas Series to the range [0, 1]."""
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s*0
