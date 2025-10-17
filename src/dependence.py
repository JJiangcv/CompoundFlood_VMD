#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: dependence.py
Created: 2025-10-17
Author: Jinghua Jiang <jinghua.jiang21@gmail.com>
Project: Project Name

Description:
    Brief description of this module.

License: MIT
"""

__author__ = "Jinghua Jiang"
__email__ = "jinghua.jiang21@gmail.com"
__version__ = "0.1.0"

import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal
import matplotlib.pyplot as plt
import seaborn as sns
import pyvinecopulib as pv
from scipy import stats

VineType = Literal["cvine", "dvine", "rvine"]

@dataclass
class CorrSummary:
    method: str
    matrix: pd.DataFrame

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics of the correlation matrix."""
        mask = np.triu(np.ones(self.matrix.shape), k=1).astype(bool) # Upper triangle mask
        upper_values = self.matrix.where(mask).stack().values
        return {
            "method": self.method,
            "mean_corr": np.mean(upper_values),
            "std_corr": np.std(upper_values),
            "min": np.min(upper_values),
            "max": np.max(upper_values)
        }
    
@dataclass
class VineFitResult:
    vine_type: VineType
    cols: List[str]
    structure: Optional[str]
    aic: Optional[float]
    bic: Optional[float]
    loglik: Optional[float]
    n_parameters: Optional[int]
    pv_model: Optional[object]
    pair_copulas: Optional[Dict] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """Return summary as pandas Series for easy comparison."""
        return pd.Series({
            "vine_type": self.vine_type,
            "aic": self.aic,
            "bic": self.bic,
            "loglik": self.loglik,
            "n_params": self.n_params,
            "n_vars": len(self.cols)
        })
    
class VineComparison:
    models: Dict[VineType, VineFitResult]
    best_model: VineType
    selection_criterion: str
    comparison_df: pd.DataFrame

def pairwise_corr(
    df: pd.DataFrame,
    cols: List[str],
    methods: Tuple[str, ...] = ("spearman", "kendall")
) -> Dict[str, CorrSummary]:
    """Calculate pairwise correlation matrices for specified methods.
    parmeters:
        df: Input DataFrame with data.
        cols: List of column names to compute correlations for.
        methods: Tuple of correlation methods to use.
    Returns:
        Dictionary of CorrSummary objects keyed by method name.
    """
    results = {}
    for method in methods:
        mat = df[cols].corr(method=method)
        results[method] = CorrSummary(method=method, matrix=mat)

    return results

def pseudo_observations(df_wide: pd.DataFrame, cols: List[str], 
                       ties_method: str = "average") -> pd.DataFrame:
    """
    Convert specified columns to pseudo-observations (uniform [0,1]).
    
    Parameters:
        df_wide: Input DataFrame containing the data
        cols: List of column names to convert
        ties_method: How to handle ties in ranking
        
    Returns:
        DataFrame of pseudo-observations with the same index as input
    """
    U = {}
    n = len(df_wide)
    
    for c in cols:
        # Use empirical CDF transformation
        ranks = df_wide[c].rank(method=ties_method)
        U[c] = (ranks / (n + 1)).to_numpy()
    
    return pd.DataFrame(U, index=df_wide.index)

def fit_cvine(U: pd.DataFrame, 
              family_set: Optional[list] = None,
              vine_order= None) -> VineFitResult:
    """
    Fit a C-Vine copula to the pseudo-observations.
    parameters:
        U: DataFrame of pseudo-observations.
        family_set: List of copula families to consider.
        vine_order: order of variables.
    returns:
        VineFitResult object with fitting results.  
    """
    X = U.to_numpy(dtype=float, copy=False)
    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values; drop NaNs before fitting.")

    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student,
                    pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
                    pv.BicopFamily.frank, pv.BicopFamily.joe,
                    pv.BicopFamily.bb7, pv.BicopFamily.bb1,
                    pv.BicopFamily.bb6, pv.BicopFamily.bb8],
        selection_criterion="aic",
        trunc_lvl=2,
        allow_rotations=True
    )

    if vine_order is None:
        model = pv.Vinecop.from_data(X)
    else:
        structure = pv.CVineStructure(vine_order)
        model = pv.Vinecop.from_structure(structure=structure)
    
    model.select(data=X, controls=controls)

    aic = model.aic()
    bic = model.bic()
    loglik = model.loglik()
    pair_copulas = _extract_pair_copulas(model)
    return VineFitResult(
        vine_type="cvine",
        cols=list(U.columns),
        structure=str(model.structure),
        aic=aic,
        bic=bic,
        loglik=loglik,
        n_parameters=model.__getattribute__,
        pv_model=model,
        pair_copulas=pair_copulas
    )

def fit_dvine(U: pd.DataFrame, 
              family_set: Optional[list] = None,
              order: Optional[List[str]] = None) -> VineFitResult:
    """
    Fit a D-Vine copula to the pseudo-observations.
    parameters:
        U: DataFrame of pseudo-observations.
        family_set: List of copula families to consider.
        order: Optional order of variables.
    returns:
        VineFitResult object with fitting results.  
    """
    X = U.to_numpy(dtype=float, copy=False)
    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values; drop NaNs before fitting.")
    
    controls = pv.FitControlsVinecop(family_set=family_set or [])
    if order:
        order_idx = [list(U.columns).index(col) for col in order]
        structure = pv.VinecopStructure(type=pv.VineType.dvine, order=order_idx)
    else:
        structure = pv.DVineStructure(dim=X.shape[1])   
    # Fit the model
    model = pv.Vinecop(structure=structure)
    model.select(data=X, controls=controls)
    aic = model.aic(data=X)
    bic = model.bic(data=X)
    loglik = model.loglik(data=X)
    pair_copulas = _extract_pair_copulas(model)
    return VineFitResult(
        vine_type="dvine",
        cols=list(U.columns),
        structure=str(model.structure),
        aic=aic,
        bic=bic,
        loglik=loglik,
        n_parameters=model.get_n_parameters(),
        pv_model=model,
        pair_copulas=pair_copulas
    )   

def fit_rvine(U: pd.DataFrame, 
              family_set: Optional[list] = None,
              truncation_level: Optional[int] = None) -> VineFitResult:
    """
    Fit an R-Vine copula to the pseudo-observations.
    parameters:
        U: DataFrame of pseudo-observations.
        family_set: List of copula families to consider.
        truncation_level: Optional truncation level for the vine.
    returns:
        VineFitResult object with fitting results.
    """
    X = U.to_numpy(dtype=float, copy=False)
    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values; drop NaNs before fitting.")   
    controls = pv.FitControlsVinecop(family_set=family_set or [])
    if truncation_level is not None:
        controls.truncation_level = truncation_level
    # Fit the model
    model = pv.Vinecop(dim=X.shape[1])
    model.select(data=X, controls=controls)
    aic = model.aic(data=X)
    bic = model.bic(data=X)
    loglik = model.loglik(data=X)
    pair_copulas = _extract_pair_copulas(model)
    return VineFitResult(
        vine_type="rvine",
        cols=list(U.columns),
        structure=str(model.structure),
        aic=aic,
        bic=bic,
        loglik=loglik,
        n_parameters=model.get_n_parameters(),
        pv_model=model,
        pair_copulas=pair_copulas
    )

def fit_all_vines(U: pd.DataFrame,
                  family_set: Optional[list] = None,
                  selection_criterion: str = "aic") -> VineComparison:
    """Fit C-Vine, D-Vine, and R-Vine copulas and compare them.
    parameters:
        U: DataFrame of pseudo-observations.
        family_set: List of copula families to consider.
        selection_criterion: Criterion for model selection ("aic" or "bic").
        verbose: Whether to print fitting progress.
    returns:
        VineComparison object with all fitted models and best model info.
    """
    models = {}

    try:
        models["cvine"] = fit_cvine(U, family_set)
    except Exception as e:
        print(f"  Warning: C-vine fitting failed: {e}")
    
    try:
        models["dvine"] = fit_dvine(U, family_set)
    except Exception as e:
        print(f"  Warning: D-vine fitting failed: {e}") 
    
    try:
        models["rvine"] = fit_rvine(U, family_set)
    except Exception as e:
        print(f"  Warning: R-vine fitting failed: {e}")

    comparison_data = []
    for vine_type, result in models.items():
        comparison_data.append({
            "vine_type": vine_type,
            "aic": result.aic,
            "bic": result.bic,
            "loglik": result.loglik,
            "n_params": result.n_parameters
        })

    comparison_df = pd.DataFrame(comparison_data).set_index("vine_type")

    if selection_criterion == "aic":
        best_model = comparison_df["aic"].idxmin()
    elif selection_criterion == "bic":
        best_model = comparison_df["bic"].idxmin()
    else:
        raise ValueError("selection_criterion must be 'aic' or 'bic'")
    
    return VineComparison(
        models=models,
        best_model=best_model,
        selection_criterion=selection_criterion,
        comparison_df=comparison_df
    )

def _extract_pair_copulas(model) -> Dict:
    """Extract pair copula information from a fitted Vinecop model.
    parameters:
        model: Fitted pv.Vinecop model.
    returns:
        Dictionary with pair copula details.
    """
    pair_info = {}
    if hasattr(model, 'pair_copulas'):
        for level in range(model.dim - 1):
                for edge in range(model.dim - level - 1):
                    pc = model.pair_copulas[level][edge]
                    if pc:
                        key = f"T{level+1}_E{edge+1}"
                        pair_info[key] = {
                            "family": str(pc.family),
                            "parameters": pc.parameters if hasattr(pc, 'parameters') else None,
                            "tau": pc.tau if hasattr(pc, 'tau') else None
                        }
    return pair_info

