# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   contrasts.R                Copyright (C) 2002-2024 Gordon Smyth
#   contrastAsCoef.R           Copyright (C) 2013-2025 Gordon Smyth
#   modelmatrix.R              Copyright (C) 2003-2005 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Contrast matrices and contrast fitting for pylimma.

Implements:
- model_matrix(): create design matrices from formula strings
- make_contrasts(): create contrast matrices from expressions
- contrasts_fit(): apply contrasts to a fitted model
- contrast_as_coef(): re-express contrasts as coefficients
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import linalg

from .classes import MArrayLM, _resolve_fit_input

if TYPE_CHECKING:
    pass


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
            "patsy is required for formula-based design matrices. Install with: pip install patsy"
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
            "patsy is required for formula-based design matrices. Install with: pip install patsy"
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

    With `coefficients`, the result is ``fit[:, coefficients]``, as in R:
    existing test statistics are kept for the selected columns, F is
    regenerated from them, and cov_coefficients, contrasts and var_prior are
    subset.

    The transformation preserves the relationship between coefficients and
    their standard errors. For orthogonal designs, the standard errors
    transform simply. For non-orthogonal designs, the correlation structure
    is accounted for.

    With `contrasts`, any previous test statistics (t, p-values, etc.) are
    removed since they are no longer valid after the transformation. The
    ``contrasts`` slot stores the matrix as supplied.

    References
    ----------
    Smyth, G. K. (2004). Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments.
    Statistical Applications in Genetics and Molecular Biology, 3(1), Article 3.
    """
    fit, _adata, _adata_key = _resolve_fit_input(data, key)
    is_anndata = _adata is not None

    # R contrasts.R:11
    if (contrasts is None) == (coefficients is None):
        raise ValueError("Must specify exactly one of contrasts or coefficients")

    # R contrasts.R:14: if coefficients are input, just subset (fit[, coefficients])
    if coefficients is not None:
        if isinstance(coefficients, (str, int, np.integer)):
            coefficients = [coefficients]
        return _return_fit(MArrayLM(fit)[:, list(coefficients)], data, key, is_anndata)

    if fit.get("coefficients") is None:
        raise ValueError("fit must contain coefficients component")
    if fit.get("stdev_unscaled") is None:
        raise ValueError("fit must contain stdev_unscaled component")

    # Remove test statistics in case e_bayes() has previously been run
    fit = MArrayLM(
        {k: v for k, v in fit.items() if k not in ("t", "p_value", "lods", "F", "F_p_value")}
    )
    fit_coef = np.asarray(fit["coefficients"], dtype=np.float64)
    stdev_unscaled = np.asarray(fit["stdev_unscaled"], dtype=np.float64)
    n_coef = fit_coef.shape[1]
    contrast_names = None

    # Extract contrast names if DataFrame, then convert to array.
    # Capture row names too so we can replicate R's row/col name check.
    contrast_rownames = None
    if isinstance(contrasts, pd.DataFrame):
        contrast_names = list(contrasts.columns)
        contrast_rownames = list(contrasts.index)
        contrasts = contrasts.values

    # R contrasts.R:31: `if(!is.numeric(contrasts)) stop("contrasts
    # must be a numeric matrix")`. Reject logical / character inputs
    # before the float coercion silently turns booleans into 0/1 and
    # parses string digits.
    contrasts_raw = np.asarray(contrasts)
    if contrasts_raw.dtype == np.bool_ or not np.issubdtype(contrasts_raw.dtype, np.number):
        raise ValueError("contrasts must be a numeric matrix")
    contrasts = contrasts_raw.astype(np.float64, copy=False)
    if contrasts.ndim == 1:
        contrasts = contrasts.reshape(-1, 1)
    if contrasts.shape[0] != n_coef:
        raise ValueError(
            f"Number of rows in contrasts ({contrasts.shape[0]}) must match "
            f"number of coefficients ({n_coef})"
        )
    if np.any(np.isnan(contrasts)):
        raise ValueError("NAs not allowed in contrasts")

    # R contrasts.R:35-40: rn = rownames(contrasts), cn = colnames(
    # fit$coefficients); rename "(Intercept)" -> "Intercept" in both
    # then warn if they don't match.
    fit_coef_names = fit.get("coef_names")
    if contrast_rownames is not None and fit_coef_names is not None:
        rn = list(contrast_rownames)
        cn = list(fit_coef_names)
        if rn and rn[0] == "(Intercept)":
            rn[0] = "Intercept"
        if cn and cn[0] == "(Intercept)":
            cn[0] = "Intercept"
        if rn != cn:
            warnings.warn("row names of contrasts don't match col names of coefficients")

    fit["contrasts"] = contrasts
    n_contrasts = contrasts.shape[1]
    if contrast_names is None:
        contrast_names = [f"contrast{i}" for i in range(n_contrasts)]
    fit["contrast_names"] = contrast_names

    # Special case of contrast matrix with 0 columns
    if not n_contrasts:
        return _return_fit(fit[:, []], data, key, is_anndata)

    # Correlation matrix of estimable coefficients. pylimma keeps a p x p
    # cov_coefficients with NaN rows for non-estimable coefficients where R
    # keeps the estimable rank x rank block in pivot order.
    cov_coefficients = fit.get("cov_coefficients")
    if cov_coefficients is None:
        warnings.warn("cov.coefficients not found in fit - assuming coefficients are orthogonal")
        cov_coefficients = np.diag(np.mean(stdev_unscaled**2, axis=0))
        cormatrix = np.eye(n_coef)
    else:
        cov_coefficients = np.asarray(cov_coefficients, dtype=np.float64)
        estimable = ~np.isnan(np.diag(cov_coefficients))
        if not estimable.all():
            if fit.get("pivot") is None:
                raise ValueError("cor.coef not full rank but pivot column not found in fit")
            est = np.asarray(fit["pivot"])[: estimable.sum()]
            cov_coefficients = cov_coefficients[np.ix_(est, est)]
        cormatrix = _cov2cor(cov_coefficients)

    # If design matrix was singular, reduce to estimable coefficients
    r = cormatrix.shape[0]
    if r < n_coef:
        est = np.asarray(fit["pivot"])[:r]
        if np.any(np.delete(contrasts, est, axis=0) != 0):
            raise ValueError("trying to take contrast of non-estimable coefficient")
        contrasts = contrasts[est, :]
        fit_coef = fit_coef[:, est]
        stdev_unscaled = stdev_unscaled[:, est]
        n_coef = r

    # Remove coefficients that don't appear in any contrast
    all_zero = np.where(np.sum(np.abs(contrasts), axis=1) == 0)[0]
    if all_zero.size:
        keep = np.setdiff1d(np.arange(n_coef), all_zero)
        contrasts = contrasts[keep, :]
        fit_coef = fit_coef[:, keep]
        stdev_unscaled = stdev_unscaled[:, keep]
        cov_coefficients = cov_coefficients[np.ix_(keep, keep)]
        cormatrix = cormatrix[np.ix_(keep, keep)]
        n_coef = keep.size

    # Replace NA coefficients with large (but finite) standard deviations
    # to allow zero contrast entries to clobber NA coefficients
    na_coef = np.isnan(fit_coef).any()
    if na_coef:
        na_mask = np.isnan(fit_coef)
        fit_coef = fit_coef.copy()
        stdev_unscaled = stdev_unscaled.copy()
        fit_coef[na_mask] = 0
        stdev_unscaled[na_mask] = 1e30

    new_coefficients = fit_coef @ contrasts

    # Test whether design was orthogonal (R contrasts.R:100-104)
    if cormatrix.size < 2:
        orthog = True
    else:
        orthog = bool(np.all(np.abs(cormatrix[np.tril_indices(n_coef, -1)]) < 1e-14))

    # New correlation matrix; R's chol() fails on a 0 x 0 matrix
    if cov_coefficients.size == 0:
        raise ValueError("'a' must have dims > 0")
    chol_cov = np.linalg.cholesky(cov_coefficients).T
    fit["cov_coefficients"] = (chol_cov @ contrasts).T @ (chol_cov @ contrasts)

    # New standard deviations
    if orthog:
        new_stdev = np.sqrt(stdev_unscaled**2 @ contrasts**2)
    else:
        chol_cor = np.linalg.cholesky(cormatrix).T
        ruc = np.einsum("ab,gb,bc->gac", chol_cor, stdev_unscaled, contrasts)
        new_stdev = np.sqrt(np.sum(ruc**2, axis=1))

    # Replace NAs if necessary
    if na_coef:
        large = new_stdev > 1e20
        new_coefficients[large] = np.nan
        new_stdev[large] = np.nan

    fit["coefficients"] = new_coefficients
    fit["stdev_unscaled"] = new_stdev
    return _return_fit(fit, data, key, is_anndata)


def _cov2cor(cov: np.ndarray) -> np.ndarray:
    """R's stats::cov2cor."""
    inv_sd = np.sqrt(1 / np.diag(cov))
    cor = inv_sd[:, None] * cov * inv_sd[None, :]
    np.fill_diagonal(cor, 1.0)
    return cor


def _return_fit(fit, data, key, is_anndata):
    """Return an MArrayLM, or store it in adata.uns[key] and return None."""
    if is_anndata:
        # Plain dict for h5ad compatibility; see lm_fit.
        data.uns[key] = dict(fit)
        return None
    return fit


def contrast_as_coef(
    design,
    contrast=None,
    first: bool = True,
) -> dict:
    """
    Reform a design matrix so that contrasts become simple coefficients.

    Port of R limma's ``contrastAsCoef``. Re-parameterises ``design`` so
    that fitting the new design directly estimates the requested
    contrasts as coefficients.

    Parameters
    ----------
    design : ndarray or DataFrame
        Design matrix of shape (n_samples, n_coef).
    contrast : ndarray or DataFrame, optional
        Contrast matrix of shape (n_coef, n_contrasts). If ``None``,
        ``design`` is returned unchanged (matching R).
    first : bool, default True
        If True, contrast columns are placed first in the new design.
        If False, they are moved to the end.

    Returns
    -------
    dict
        ``design`` : DataFrame
            Re-parameterised design with shape (n_samples, n_coef) and
            named columns.
        ``coef`` : list of int
            0-based indices of the contrast columns within the new
            design.
        ``qr`` : dict
            QR decomposition of ``contrast`` with keys ``q``, ``r``,
            ``pivot``, ``rank``.
    """
    from .lmfit import _qr_r_style

    if isinstance(design, pd.DataFrame):
        design_arr = np.asarray(design.values, dtype=np.float64)
    else:
        design_arr = np.asarray(design, dtype=np.float64)
    if design_arr.ndim == 1:
        design_arr = design_arr.reshape(-1, 1)

    if contrast is None:
        return design_arr

    contrast_names = None
    if isinstance(contrast, pd.DataFrame):
        contrast_names = list(contrast.columns)
        contrast_arr = np.asarray(contrast.values, dtype=np.float64)
    else:
        contrast_arr = np.asarray(contrast, dtype=np.float64)
    if contrast_arr.ndim == 1:
        contrast_arr = contrast_arr.reshape(-1, 1)

    if design_arr.shape[1] != contrast_arr.shape[0]:
        raise ValueError("Length of contrast doesn't match ncol(design)")

    q, r, pivot, rank = _qr_r_style(contrast_arr)
    if rank == 0:
        raise ValueError("contrast is all zero")

    designT = q.T @ design_arr.T
    R_block = r[:rank, :rank]
    designT[:rank, :] = linalg.solve_triangular(R_block, designT[:rank, :])
    new_design = designT.T

    n_cols = new_design.shape[1]
    col_names = [f"Q{i + 1}" for i in range(n_cols)]
    if contrast_names is None:
        col_names[:rank] = [f"C{int(pivot[i]) + 1}" for i in range(rank)]
    else:
        col_names[:rank] = list(contrast_names[:rank])
    coef = list(range(rank))

    if not first:
        non_coef = [i for i in range(n_cols) if i not in coef]
        new_design = np.column_stack([new_design[:, non_coef], new_design[:, coef]])
        col_names = [col_names[i] for i in non_coef] + [col_names[i] for i in coef]
        coef = list(range(n_cols - rank, n_cols))

    new_design_df = pd.DataFrame(new_design, columns=col_names)

    return {
        "design": new_design_df,
        "coef": coef,
        "qr": {"q": q, "r": r, "pivot": pivot, "rank": rank},
    }
