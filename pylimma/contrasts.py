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

import re
import warnings
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import linalg

from .classes import MArrayLM, _resolve_fit_input
from .lmfit import _design_names
from .utils import _r_as_character

if TYPE_CHECKING:
    pass

# R's reserved words (?Reserved), to which make.names() appends a dot
_R_RESERVED = frozenset(
    "if else repeat while function for in next break TRUE FALSE NULL Inf NaN NA "
    "NA_integer_ NA_real_ NA_complex_ NA_character_".split()
)
# Also reserved, for the dots arguments, though make.names() keeps them
_R_DOTS = r"\.\.(?:\.|\d+)"
# An R name: a letter, or a dot not followed by a digit, then letters, digits,
# dots and underscores
_R_NAME = r"(?:[^\W\d_]|\.(?!\d))[\w.]*"
# One token of a contrast expression: an R number (1, 1.5, .5, 1e-3, 2L, 0x1F),
# an R name, or an arithmetic operator or bracket (** is R's other spelling of ^)
_CONTRAST_TOKEN = re.compile(
    r"\s*(?:(?P<number>0[xX][0-9a-fA-F]+L?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?)"
    rf"|(?P<name>{_R_NAME})|(?P<op>\*\*|[-+*/^()]))"
)


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
    levels : list of str, categorical or design matrix
        Coefficient names. Can be:
        - List of coefficient names (a single string is one name)
        - Categorical (``pd.Categorical`` or a categorical Series): its
          categories, as R's ``levels(factor)``
        - Design matrix with column names (DataFrame or patsy
          ``DesignMatrix``)
        A matrix without column names, non-string names and duplicated
        names raise a ValueError.
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

    As in R, level names must be syntactically valid R names (equal to
    their ``make.names()``), so ``"WT.Ctrl"`` and Python keywords such as
    ``"not"`` are allowed, but ``"_x"``, ``"if"`` and the dots names
    ``"..."`` and ``"..1"`` are not. Expressions are strings and follow R
    arithmetic: ``^`` (or ``**``) is a power, numbers may be written as in
    R (``1e-3``, ``.5``, ``2L``) and ``1/0`` is ``inf``. R evaluates the
    expression in the R session, so R constants and functions (``pi``,
    ``mean()``) also work there; pylimma supports only the level names,
    numbers, ``+ - * / ^`` and brackets, and raises a ValueError for
    anything else.
    """
    levels = _contrast_levels(levels)

    # Handle R's "(Intercept)" naming
    if levels and levels[0] == "(Intercept)":
        levels[0] = "Intercept"

    n = len(levels)
    if n < 1:
        raise ValueError("No levels to construct contrasts from")
    # R gives the indicator to the last copy of a duplicated name only
    duplicated = sorted(lev for lev, count in Counter(levels).items() if count > 1)
    if duplicated:
        raise ValueError(f"levels must be unique; duplicated: {', '.join(duplicated)}")

    # R: notvalid <- (levels != make.names(levels))
    invalid = [lev for lev in levels if not _is_syntactic_name(lev)]
    if invalid:
        raise ValueError(
            "The levels must be syntactically valid names in R, see help(make.names). "
            f"Non-valid names: {', '.join(invalid)}"
        )

    # Each level is an indicator vector, bound to a placeholder name so that
    # level names need not be Python identifiers
    placeholders = {lev: f"_level{i}" for i, lev in enumerate(levels)}
    identity = np.eye(n)
    indicators = {placeholders[lev]: identity[i] for i, lev in enumerate(levels)}

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

        if not isinstance(expr, str):
            raise ValueError(f"Contrast expressions must be strings such as 'B-A', not {expr!r}")
        python_expr, numbers = _contrast_as_python(expr, placeholders)
        try:
            # R arithmetic on doubles: 1/0 is Inf and (-8)^(1/3) is NaN, silently
            with np.errstate(all="ignore"):
                result = eval(python_expr, {"__builtins__": {}}, {**indicators, **numbers})
            contrast_matrix[:, j] = np.asarray(result)
        except Exception as e:
            raise ValueError(f"Could not evaluate contrast expression '{expr}': {e}")

    # Return as DataFrame with named rows and columns
    return pd.DataFrame(contrast_matrix, index=levels, columns=contrast_names)


def _is_syntactic_name(name: str) -> bool:
    """R: name == make.names(name), except the dots names ``...`` and ``..1``.

    R cannot evaluate those as variables ("'...' used in an incorrect context").
    """
    return (
        re.fullmatch(_R_NAME, name) is not None
        and name not in _R_RESERVED
        and re.fullmatch(_R_DOTS, name) is None
    )


def _contrast_as_python(
    expr: str, placeholders: dict[str, str]
) -> tuple[str, dict[str, np.float64]]:
    """Translate an R contrast expression into Python over the level placeholders.

    Returns the Python expression and the numbers it uses, bound to their own
    placeholders as doubles so that the arithmetic is R's. R evaluates the
    expression in the R session, where any R name or function resolves; only
    arithmetic on the levels and numbers is supported here.
    """
    # R parses each line as its own expression and keeps the last one only
    if "\n" in expr or "\r" in expr:
        raise ValueError(
            f"Contrast expression {expr!r} spans several lines; R would evaluate only "
            "its last line. Write it on one line."
        )
    tokens = []
    numbers = {}
    pos = 0
    expr_end = len(expr.rstrip())
    while pos < expr_end:
        token = _CONTRAST_TOKEN.match(expr, pos)
        if token is None:
            raise ValueError(
                f"Contrast expression '{expr}': unsupported syntax at '{expr[pos:].strip()}'. "
                "Contrasts may use the level names, numbers, + - * / ^ and brackets."
            )
        if token["name"] is not None:
            if token["name"] not in placeholders:
                raise ValueError(
                    f"Contrast expression '{expr}': '{token['name']}' is not one of the "
                    "levels. Contrasts may use the level names, numbers, + - * / ^ and "
                    "brackets."
                )
            tokens.append(placeholders[token["name"]])
        elif token["number"] is not None:
            placeholder = f"_number{len(numbers)}"
            literal = token["number"].rstrip("L")  # R's integer suffix L
            is_hex = literal[:2] in ("0x", "0X")
            numbers[placeholder] = np.float64(int(literal, 16) if is_hex else literal)
            tokens.append(placeholder)
        else:
            tokens.append("**" if token["op"] == "^" else token["op"])
        pos = token.end()
    return " ".join(tokens), numbers


def _contrast_levels(levels) -> list[str]:
    """Parameter names from ``levels``, as R's makeContrasts reads them."""
    if isinstance(levels, str):
        return [levels]
    # R: if(is.factor(levels)) levels <- levels(levels)
    if isinstance(getattr(levels, "dtype", None), pd.CategoricalDtype):
        return list(_r_as_character(levels.dtype.categories))
    # R: if(!is.character(levels)) levels <- colnames(levels)
    names = _design_names(levels)
    if names is not None:
        return names
    values = np.asarray(levels, dtype=object)
    if values.ndim != 1 or not all(isinstance(v, str) for v in values):
        raise ValueError(
            "levels must be the parameter names: a list of strings, a categorical "
            "or a design matrix with column names"
        )
    return list(values)


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
    **AnnData views.** Results are written into the AnnData. If it is a
    view (e.g. ``adata[:, mask]``), anndata first turns it into a
    standalone copy, with an ``ImplicitModificationWarning``, and the
    results land on that copy rather than on the parent. Assign the view
    to a variable first (``sub = adata[:, mask]``) or pass
    ``adata[:, mask].copy()``; a bare ``fn(adata[:, mask])`` call
    discards the results.

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
