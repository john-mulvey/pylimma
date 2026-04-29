# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   contrasts.R                Copyright (C) 2002-2024 Gordon Smyth
#   modelmatrix.R              Copyright (C) 2003-2005 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Contrast matrices and contrast fitting for pylimma.

Implements:
- model_matrix(): create design matrices from formula strings
- make_contrasts(): create contrast matrices from expressions
- contrasts_fit(): apply contrasts to a fitted model
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

from .classes import MArrayLM, _resolve_fit_input

if TYPE_CHECKING:
    from anndata import AnnData


def model_matrix(
    formula: str,
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Create a design matrix from a formula and data.

    This function creates design matrices from R-style formula strings,
    matching the behaviour of R's model.matrix() with default contrasts
    (contr.treatment / dummy coding).

    Parameters
    ----------
    formula : str
        R-style formula string. Examples:
        - "~ group" : intercept + dummy variables for group (reference coding)
        - "~ 0 + group" or "~ group - 1" : no intercept (cell-means coding)
        - "~ group + batch" : additive model with two factors
        - "~ group + age" : factor plus numeric covariate
    data : DataFrame
        Data containing the variables referenced in the formula.
        Columns should include all variables used in the formula.

    Returns
    -------
    ndarray
        Design matrix of shape (n_samples, n_coefficients).
        Use model_matrix_with_names() if column names are needed.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'group': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'age': [25, 30, 35, 40, 45, 50]
    ... })
    >>> model_matrix("~ group", data)
    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 0.],
           [1., 0., 1.],
           [1., 0., 1.]])

    >>> model_matrix("~ 0 + group", data)
    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 0., 1.]])

    Notes
    -----
    This function uses patsy for formula parsing with Treatment coding
    to match R's default contr.treatment contrast scheme.

    See Also
    --------
    make_contrasts : Create contrast matrices for hypothesis testing
    """
    try:
        import patsy
    except ImportError:
        raise ImportError(
            "patsy is required for formula-based design matrices. "
            "Install with: pip install patsy"
        )

    # Use Treatment coding to match R's contr.treatment (reference coding)
    # This is patsy's default, but we set it explicitly for clarity
    design_info = patsy.dmatrix(formula, data, return_type="dataframe")

    return np.asarray(design_info, dtype=np.float64)


def model_matrix_with_names(
    formula: str,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a design matrix from a formula, returning a DataFrame with column names.

    This is the same as model_matrix() but returns a DataFrame preserving
    the column names, which is useful for inspecting the design structure.

    Parameters
    ----------
    formula : str
        R-style formula string (see model_matrix for examples).
    data : DataFrame
        Data containing the variables referenced in the formula.

    Returns
    -------
    DataFrame
        Design matrix with named columns.

    See Also
    --------
    model_matrix : Returns a numpy array (faster, no column names)
    """
    try:
        import patsy
    except ImportError:
        raise ImportError(
            "patsy is required for formula-based design matrices. "
            "Install with: pip install patsy"
        )

    return patsy.dmatrix(formula, data, return_type="dataframe")


def make_contrasts(
    *contrasts_args: str,
    contrasts: list[str] | None = None,
    levels: list[str] | np.ndarray | pd.DataFrame,
    **named_contrasts: str,
) -> pd.DataFrame:
    """
    Construct a contrast matrix from contrast expressions.

    Parameters
    ----------
    *contrasts_args : str
        Contrast expressions like "B-A", "C-A", "(B+C)/2-A".
        Each expression becomes a column in the contrast matrix.
        The column name is the expression itself.
    contrasts : list of str, optional
        Alternative way to pass contrasts as a list (R parity).
        Cannot be used together with positional contrasts.
    levels : list of str, ndarray, or DataFrame
        Coefficient names. Can be:
        - List of coefficient names
        - Design matrix (column names extracted)
        - Factor (levels extracted)
    **named_contrasts : str
        Named contrast expressions. The keyword becomes the column name,
        the value is the expression. E.g., TreatmentVsControl="B-A".

    Returns
    -------
    DataFrame
        Contrast matrix of shape (n_levels, n_contrasts).
        Row index contains level names, columns contain contrast names.

    Examples
    --------
    >>> # Unnamed contrasts (expression becomes name)
    >>> make_contrasts("B-A", "C-A", levels=['A', 'B', 'C'])
           B-A  C-A
    A     -1.0 -1.0
    B      1.0  0.0
    C      0.0  1.0

    >>> # Named contrasts
    >>> make_contrasts(
    ...     TreatmentVsControl="B-A",
    ...     DrugVsDMSO="C-A",
    ...     levels=['A', 'B', 'C']
    ... )
           TreatmentVsControl  DrugVsDMSO
    A                    -1.0        -1.0
    B                     1.0         0.0
    C                     0.0         1.0

    >>> # Mixed: unnamed and named
    >>> make_contrasts("C-B", AvsRest="A-(B+C)/2", levels=['A', 'B', 'C'])

    Notes
    -----
    The contrast expressions are evaluated in an environment where each
    level name is bound to an indicator vector. For example, with levels
    ['A', 'B', 'C'], the expression "B-A" evaluates to [0, 1, 0] - [1, 0, 0]
    = [-1, 1, 0].
    """
    # Extract level names
    if isinstance(levels, pd.DataFrame):
        levels = list(levels.columns)
    elif isinstance(levels, np.ndarray):
        if hasattr(levels, "columns"):
            levels = list(levels.columns)
        elif levels.ndim == 2:
            levels = [f"x{i}" for i in range(levels.shape[1])]
        else:
            levels = list(levels)
    else:
        levels = list(levels)

    # Handle R's "(Intercept)" naming
    if levels and levels[0] == "(Intercept)":
        levels[0] = "Intercept"

    n = len(levels)
    if n < 1:
        raise ValueError("No levels to construct contrasts from")

    # Validate level names are valid Python identifiers (R parity)
    invalid = [lev for lev in levels if not lev.isidentifier()]
    if invalid:
        raise ValueError(
            f"Level names must be valid Python identifiers. Invalid names: {', '.join(invalid)}"
        )

    # Create indicator vectors for each level
    indicators = {lev: np.zeros(n) for lev in levels}
    for i, lev in enumerate(levels):
        indicators[lev][i] = 1.0

    # Handle contrasts= parameter (R parity)
    if contrasts is not None:
        if contrasts_args:
            raise ValueError("Cannot specify both positional contrasts and contrasts= parameter")
        contrast_exprs = list(contrasts)
    else:
        contrast_exprs = list(contrasts_args)

    # Combine unnamed and named contrasts
    # Unnamed: expression is both the name and expression
    # Named: keyword is name, value is expression
    all_contrasts = [(expr, expr) for expr in contrast_exprs]
    all_contrasts.extend((name, expr) for name, expr in named_contrasts.items())

    if not all_contrasts:
        raise ValueError("No contrasts specified")

    # Evaluate each contrast expression
    n_contrasts = len(all_contrasts)
    contrast_matrix = np.zeros((n, n_contrasts))
    contrast_names = []

    for j, (name, expr) in enumerate(all_contrasts):
        contrast_names.append(name)

        # Evaluate the expression
        try:
            result = eval(expr, {"__builtins__": {}}, indicators)
            contrast_matrix[:, j] = np.asarray(result)
        except Exception as e:
            raise ValueError(f"Could not evaluate contrast expression '{expr}': {e}")

    # Return as DataFrame with named rows and columns
    return pd.DataFrame(contrast_matrix, index=levels, columns=contrast_names)


def contrasts_fit(
    data,
    contrasts: np.ndarray | pd.DataFrame | None = None,
    coefficients: int | str | list | None = None,
    key: str = "pylimma",
) -> dict | None:
    """
    Apply contrast matrix to a fitted model.

    Transforms coefficients and standard errors to reflect contrasts of
    interest rather than the original model parameterisation.

    Parameters
    ----------
    data : AnnData or dict
        Either an AnnData object with fit results in adata.uns[key],
        or a dict returned by lm_fit().
    contrasts : ndarray or DataFrame, optional
        Contrast matrix of shape (n_original_coefs, n_contrasts).
        Each column defines a contrast. If DataFrame, column names are
        preserved as contrast names.
    coefficients : int, str, or list, optional
        Alternative to `contrasts`. Specifies which coefficients to keep
        in the revised fit object. Can be indices (int), names (str), or
        a list of either. This is a simpler way to subset coefficients
        without defining a full contrast matrix.

        .. warning::
           Integer indices are **0-based** (Python convention). R's
           ``contrasts.fit(fit, coefficients=c(2, 3))`` uses 1-based
           indices; the equivalent pylimma call is
           ``contrasts_fit(fit, coefficients=[1, 2])``. Prefer string
           names when porting R code to avoid silent off-by-one errors.
    key : str, default "pylimma"
        Key for fit results in adata.uns (AnnData input only).

    Returns
    -------
    dict or None
        If input is dict, returns updated dict with transformed coefficients.
        If input is AnnData, updates adata.uns[key] in place and returns None.

    Notes
    -----
    Exactly one of `contrasts` or `coefficients` must be provided.

    The `coefficients` parameter provides a simpler way to specify the
    contrasts matrix when the desired contrasts are just a subset of
    the original coefficients.

    The transformation preserves the relationship between coefficients and
    their standard errors. For orthogonal designs, the standard errors
    transform simply. For non-orthogonal designs, the correlation structure
    is accounted for.

    Any previous test statistics (t, p-values, etc.) are removed since they
    are no longer valid after the transformation.

    References
    ----------
    Smyth, G. K. (2004). Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments.
    Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
    """
    fit, _adata, _adata_key = _resolve_fit_input(data, key)
    is_anndata = _adata is not None

    # Validate fit object
    if "coefficients" not in fit:
        raise ValueError("fit must contain coefficients")
    if "stdev_unscaled" not in fit:
        raise ValueError("fit must contain stdev_unscaled")

    # Check that exactly one of contrasts or coefficients is provided
    if contrasts is None and coefficients is None:
        raise ValueError("Must specify either 'contrasts' or 'coefficients'")
    if contrasts is not None and coefficients is not None:
        raise ValueError("Cannot specify both 'contrasts' and 'coefficients'")

    # Remove any previous test statistics
    fit = {k: v for k, v in fit.items() if k not in ("t", "p_value", "lods", "F", "F_p_value")}

    # Get dimensions
    fit_coef = fit["coefficients"]
    stdev_unscaled = fit["stdev_unscaled"]
    n_genes, n_coef = fit_coef.shape

    # Handle coefficients parameter - convert to contrast matrix
    contrast_names = None
    if coefficients is not None:
        # Normalize to list
        if not isinstance(coefficients, (list, tuple)):
            coefficients = [coefficients]

        # Get coefficient names from fit if available
        coef_names = fit.get("coef_names")

        # Convert names to indices and build contrast matrix
        coef_indices = []
        selected_names = []
        for c in coefficients:
            if isinstance(c, str):
                if coef_names is None:
                    raise ValueError(
                        f"Cannot use coefficient name '{c}' - no coef_names in fit"
                    )
                if c not in coef_names:
                    raise ValueError(
                        f"Coefficient '{c}' not found. Available: {coef_names}"
                    )
                idx = coef_names.index(c)
                coef_indices.append(idx)
                selected_names.append(c)
            else:
                coef_indices.append(int(c))
                if coef_names is not None:
                    selected_names.append(coef_names[int(c)])
                else:
                    selected_names.append(f"coef{int(c)}")

        # Build identity-like contrast matrix for selected coefficients
        contrasts = np.zeros((n_coef, len(coef_indices)))
        for j, idx in enumerate(coef_indices):
            contrasts[idx, j] = 1.0
        contrast_names = selected_names

    # Extract contrast names if DataFrame, then convert to array
    if isinstance(contrasts, pd.DataFrame):
        contrast_names = list(contrasts.columns)
        contrasts = contrasts.values

    contrasts = np.asarray(contrasts, dtype=np.float64)
    if contrasts.ndim == 1:
        contrasts = contrasts.reshape(-1, 1)
    if contrasts.shape[0] != n_coef:
        raise ValueError(
            f"Number of rows in contrasts ({contrasts.shape[0]}) must match "
            f"number of coefficients ({n_coef})"
        )
    if np.any(np.isnan(contrasts)):
        raise ValueError("NAs not allowed in contrasts")

    fit["contrasts"] = contrasts
    n_contrasts = contrasts.shape[1]

    # Store contrast names (generate default names if not provided)
    if contrast_names is None:
        contrast_names = [f"contrast{i}" for i in range(n_contrasts)]
    fit["contrast_names"] = contrast_names

    # Handle empty contrasts
    if n_contrasts == 0:
        fit["coefficients"] = fit_coef[:, :0]
        fit["stdev_unscaled"] = stdev_unscaled[:, :0]
        # Match R's fit[,0] subsetting: every coefficient-indexed slot
        # collapses to its 0-col form so the returned MArrayLM is
        # internally shape-consistent.
        if fit.get("cov_coefficients") is not None:
            cov = np.asarray(fit["cov_coefficients"])
            fit["cov_coefficients"] = cov[:0, :0]
        if fit.get("var_prior") is not None:
            fit["var_prior"] = np.asarray(fit["var_prior"])[:0]
        if fit.get("coef_names") is not None:
            fit["coef_names"] = []
        if is_anndata:
            # Plain dict for h5ad compatibility; see lm_fit.
            data.uns[key] = dict(fit)
            return None
        if not isinstance(fit, MArrayLM):
            fit = MArrayLM(fit)
        return fit

    # R's contrasts.fit strips rows (coefficients) that are zero in every
    # contrast column before the orthogonality check (contrasts.R:77-85).
    # Comment there says "Not necessary but can make the function faster",
    # but removing those rows also changes which correlations enter the
    # lower.tri(cormatrix) orthog test and can flip the code path on
    # pathological contrasts that leave an unused-but-correlated
    # coefficient.
    contrasts_all_zero = np.where(np.all(contrasts == 0, axis=1))[0]
    if contrasts_all_zero.size and contrasts_all_zero.size < n_coef:
        keep = np.setdiff1d(np.arange(n_coef), contrasts_all_zero)
        contrasts = contrasts[keep, :]
        fit_coef = fit_coef[:, keep]
        stdev_unscaled = stdev_unscaled[:, keep]
        if fit.get("cov_coefficients") is not None:
            cov_existing = np.asarray(fit["cov_coefficients"])
            fit["cov_coefficients"] = cov_existing[np.ix_(keep, keep)]
        n_coef = len(keep)
        fit["contrasts"] = contrasts

    # Get or construct covariance matrix
    if "cov_coefficients" not in fit or fit["cov_coefficients"] is None:
        warnings.warn(
            "cov_coefficients not found in fit - assuming coefficients are orthogonal"
        )
        var_coef = np.nanmean(stdev_unscaled**2, axis=0)
        cov_coefficients = np.diag(var_coef)
        cormatrix = np.eye(n_coef)
        orthog = True
    else:
        cov_coefficients = fit["cov_coefficients"]
        # Handle NaN in cov matrix (non-estimable coefficients)
        valid_idx = ~np.isnan(np.diag(cov_coefficients))
        if not np.all(valid_idx):
            # Reduce to estimable coefficients
            est_idx = np.where(valid_idx)[0]
            cov_coefficients = cov_coefficients[np.ix_(est_idx, est_idx)]

            # Check contrasts only use estimable coefficients
            non_est_idx = np.where(~valid_idx)[0]
            if np.any(contrasts[non_est_idx, :] != 0):
                raise ValueError("Trying to take contrast of non-estimable coefficient")

            contrasts = contrasts[est_idx, :]
            fit_coef = fit_coef[:, est_idx]
            stdev_unscaled = stdev_unscaled[:, est_idx]
            n_coef = len(est_idx)

        # Compute correlation matrix
        std = np.sqrt(np.diag(cov_coefficients))
        std[std == 0] = 1  # Avoid division by zero
        cormatrix = cov_coefficients / np.outer(std, std)

        # Check if orthogonal
        if cormatrix.size < 2:
            orthog = True
        else:
            orthog = np.sum(np.abs(cormatrix[np.tril_indices(n_coef, -1)])) < 1e-12

    # Handle NA coefficients
    na_coef = np.any(np.isnan(fit_coef))
    if na_coef:
        na_mask = np.isnan(fit_coef)
        fit_coef = fit_coef.copy()
        stdev_unscaled = stdev_unscaled.copy()
        fit_coef[na_mask] = 0
        stdev_unscaled[na_mask] = 1e30

    # Transform coefficients
    new_coefficients = fit_coef @ contrasts

    # Transform covariance matrix
    R = np.linalg.cholesky(cov_coefficients).T  # Upper triangular
    new_cov = (R @ contrasts).T @ (R @ contrasts)
    fit["cov_coefficients"] = new_cov

    # Transform standard errors
    if orthog:
        # Simple case: variances add
        new_stdev = np.sqrt(stdev_unscaled**2 @ contrasts**2)
    else:
        # Non-orthogonal: need to account for correlations
        R_cor = np.linalg.cholesky(cormatrix).T
        new_stdev = np.zeros((n_genes, n_contrasts))
        for i in range(n_genes):
            # Scale contrasts by stdev
            scaled_contrasts = stdev_unscaled[i, :, np.newaxis] * contrasts
            RUC = R_cor @ scaled_contrasts
            new_stdev[i, :] = np.sqrt(np.sum(RUC**2, axis=0))

    # Restore NAs
    if na_coef:
        large_stdev = new_stdev > 1e20
        new_coefficients[large_stdev] = np.nan
        new_stdev[large_stdev] = np.nan

    fit["coefficients"] = new_coefficients
    fit["stdev_unscaled"] = new_stdev

    if not isinstance(fit, MArrayLM):
        fit = MArrayLM(fit)

    if is_anndata:
        # Plain dict for h5ad compatibility; see lm_fit.
        data.uns[key] = dict(fit)
        return None
    return fit
