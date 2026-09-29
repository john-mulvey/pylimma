# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   decidetests.R              Copyright (C) 2004-2017 Gordon Smyth
# Python port: Copyright (C) 2026 John Mulvey
"""
Multiple testing decisions for pylimma.

Implements decision procedures for classifying genes as differentially expressed:
- decide_tests(): classify genes as up/down/not significant
- classify_tests_f(): F-test based classification for multiple contrasts
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from .classes import _is_anndata, _resolve_fit_input
from .utils import _match_arg, p_adjust

if TYPE_CHECKING:
    pass


def classify_tests_f(
    fit: dict | np.ndarray,
    cor_matrix: np.ndarray | None = None,
    df: float | np.ndarray = np.inf,
    p_value: float = 0.01,
    fstat_only: bool = False,
) -> np.ndarray | tuple[np.ndarray, int, float]:
    """
    Use F-tests to classify vectors of t-statistics into outcomes.

    This function performs an overall F-test for each gene, and optionally
    classifies which contrasts are significant using a step-down procedure.

    Parameters
    ----------
    fit : dict
        Fit object containing t-statistics and coefficient covariance.
        Must have keys: 't', and optionally 'cov_coefficients', 'df_prior',
        'df_residual'.
    p_value : float, default 0.01
        P-value threshold for significance.
    fstat_only : bool, default False
        If True, return only the F-statistics (with df1, df2 as attributes).
        If False, return a classification matrix (-1, 0, 1).

    Returns
    -------
    ndarray or tuple
        If fstat_only=True: tuple of (F-statistics, df1, df2)
        If fstat_only=False: matrix of test results (-1=down, 0=not sig, 1=up)

    Notes
    -----
    The F-statistic is computed as a quadratic form in the t-statistics,
    adjusted for correlation between coefficients. When the coefficients
    are uncorrelated, this reduces to the mean of squared t-statistics.
    """
    # R's classifyTestsF accepts either an MArrayLM-like list or a
    # bare t-statistic matrix. Support both here.
    if isinstance(fit, dict):
        tstat = np.asarray(fit["t"])
    else:
        tstat = np.asarray(fit)
        fit = None
    if tstat.ndim == 1:
        tstat = tstat.reshape(-1, 1)
    n_genes, n_tests = tstat.shape

    # df resolution: explicit kwarg takes precedence, otherwise derive
    # from df_prior + df_residual on the fit (matches R decidetests.R:190
    # and the df=Inf default). Keep df as a vector when supplied - scipy
    # broadcasts it over the gene axis.
    if fit is not None and np.isinf(np.asarray(df)).all():
        if "df_prior" in fit and "df_residual" in fit:
            df = fit["df_prior"] + fit["df_residual"]

    # Single coefficient case
    if n_tests == 1:
        fstat = tstat[:, 0] ** 2
        if fstat_only:
            return fstat, 1, df
        p = 2 * stats.t.sf(np.abs(tstat[:, 0]), df)
        result = np.sign(tstat[:, 0]) * (p < p_value)
        return result.astype(int).reshape(-1, 1)

    # Multiple coefficients: build the correlation matrix from the
    # explicit cor_matrix kwarg, otherwise derive it from the fit's
    # cov_coefficients (R falls back to diag(n_tests)/sqrt(n_tests)
    # when neither is available).
    if cor_matrix is not None:
        cor_matrix = np.asarray(cor_matrix, dtype=np.float64)
    elif fit is not None and fit.get("cov_coefficients") is not None:
        cov = np.asarray(fit["cov_coefficients"], dtype=np.float64)
        # R's cov.coefficients covers only the estimable coefficients
        # (rank x rank); pylimma pads non-estimable ones with NaN. Use the
        # estimable block as R does, but keep every t-statistic column: a
        # gene whose t-statistics include NA is classified NA below, as in R.
        estimable = ~np.isnan(np.diag(cov))
        cov = cov[np.ix_(estimable, estimable)]
        diag_vals = np.diag(cov)
        if diag_vals.size and np.min(diag_vals) == 0:
            cov = cov.copy()
            zero_mask = diag_vals == 0
            cov[np.diag_indices_from(cov)] = np.where(zero_mask, 1.0, diag_vals)
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1
        cor_matrix = cov / np.outer(std, std)

    if cor_matrix is not None:
        eigvals, eigvecs = np.linalg.eigh(cor_matrix)
        r = int(np.sum(eigvals / eigvals[-1] > 1e-8))
        Q = eigvecs[:, -r:] / np.sqrt(eigvals[-r:]) / np.sqrt(r)
    else:
        r = n_tests
        Q = np.eye(r) / np.sqrt(r)

    df2 = df
    # R's tstat %*% Q is non-conformable when the correlation matrix covers
    # fewer coefficients than there are t-statistic columns.
    conformable = Q.shape[0] == n_tests

    if fstat_only:
        if not conformable:
            raise ValueError("non-conformable arguments: cor_matrix does not match the number of tests")
        return np.sum((tstat @ Q) ** 2, axis=1), r, df2

    # Classification using step-down procedure. stats.f.ppf with
    # vector df2 returns a vector of per-gene thresholds.
    # R: qf(p.value, r, df, lower.tail=FALSE); scipy's F quantile is NaN at
    # df = Inf, where R returns the chi-squared limit qchisq(p, r) / r.
    df2_arr = np.broadcast_to(np.asarray(df2, dtype=np.float64), (n_genes,))
    qF = np.where(
        np.isinf(df2_arr),
        stats.chi2.isf(p_value, r) / r,
        stats.f.isf(p_value, r, np.where(np.isinf(df2_arr), 1.0, df2_arr)),
    )

    result = np.zeros((n_genes, n_tests), dtype=float)

    for i in range(n_genes):
        x = tstat[i, :]
        if np.any(np.isnan(x)):
            result[i, :] = np.nan  # R sets to NA, not 0
            continue
        if not conformable:
            raise ValueError("non-conformable arguments: cor_matrix does not match the number of tests")

        # Check if overall F-test is significant
        if (x @ Q @ Q.T @ x) > qF[i]:
            # Order by absolute t-statistic
            # R: order(abs(x), decreasing=TRUE) keeps ties in their original order
            order = np.argsort(-np.abs(x), kind="stable")
            result[i, order[0]] = int(np.sign(x[order[0]]))

            # Step-down: check if adding each coefficient improves the F
            for j in range(1, n_tests):
                bigger = order[:j]
                x_adj = x.copy()
                # Set larger coefficients to same magnitude as current
                x_adj[bigger] = np.sign(x[bigger]) * np.abs(x[order[j]])

                if (x_adj @ Q @ Q.T @ x_adj) > qF[i]:
                    result[i, order[j]] = int(np.sign(x[order[j]]))
                else:
                    break

    return result


_DECIDE_METHODS = ("separate", "global", "hierarchical", "nestedF")
_ADJUST_METHODS = ("none", "bonferroni", "holm", "BH", "fdr", "BY")


def _cutoff_multiplier(adjust_method: str, n: int, n_selected: int) -> float:
    """The ``a`` multiplier R applies to p.value after gene-level selection."""
    return {
        "none": 1.0,
        "bonferroni": 1.0 / n,
        "holm": 1.0 / (n - n_selected + 1),
        "BH": n_selected / n,
        "BY": n_selected / n / np.sum(1.0 / np.arange(1, n + 1)),
    }[adjust_method]


def _sign_times(is_de: np.ndarray, values: np.ndarray) -> np.ndarray:
    """R's ``sign(values) * is_de`` with NA propagating from either operand."""
    return np.sign(np.asarray(values, dtype=np.float64)) * is_de


def _as_results(results: np.ndarray) -> np.ndarray:
    """Integer matrix unless R's result contains NA."""
    return results if np.isnan(results).any() else results.astype(int)


def decide_tests(
    data,
    method: str = "separate",
    adjust_method: str = "BH",
    p_value: float = 0.05,
    lfc: float = 0.0,
    coefficients: np.ndarray | None = None,
    cor_matrix: np.ndarray | None = None,
    tstat: np.ndarray | None = None,
    df: float | np.ndarray = np.inf,
    genewise_p_value: np.ndarray | None = None,
    *,
    key: str = "pylimma",
) -> np.ndarray:
    r"""
    Classify each gene and contrast as up, down or not significant.

    Port of R limma's ``decideTests``. A fit (``MArrayLM`` dict or AnnData)
    dispatches to ``decideTests.MArrayLM``; a matrix of p-values dispatches
    to ``decideTests.default``.

    Parameters
    ----------
    data : AnnData, dict or ndarray
        Fit object (``e_bayes`` is run automatically if needed) or a
        matrix of p-values. If AnnData, reads from ``adata.uns[key]``.
    method : {"separate", "global", "hierarchical", "nestedF"}
        Partial matching as in R. ``"nestedF"`` requires a fit.
    adjust_method : {"BH", "fdr", "none", "bonferroni", "holm", "BY"}
    p_value : float, default 0.05
        Cut-off for adjusted p-values.
    lfc : float, default 0.0
        Minimum absolute log-fold-change.
    coefficients, tstat : ndarray, optional
        Signs for a p-value matrix (``tstat`` is used if ``coefficients``
        is None). Ignored for a fit, as in R.
    cor_matrix, df : optional
        Accepted for signature compatibility with R; R's methods do not
        use them.
    genewise_p_value : ndarray, optional
        Genewise p-values for ``method="hierarchical"`` on a p-value
        matrix. Ignored for a fit, as in R.
    key : str, default "pylimma"
        Key for fit results in ``adata.uns`` (AnnData input only).

    Returns
    -------
    ndarray
        Results with values -1, 0 and 1, shape (n_genes, n_coefficients).
        A float array with NaN wherever R returns NA (a fit with missing
        p-values or coefficients); otherwise integer.

    Notes
    -----
    Deliberate divergence from R (intended rather than literal
    behaviour): for a p-value matrix with ``method="hierarchical"``, R
    limma 3.66.0's ``decideTests.default`` fails with "object 'ngenes'
    not found" when ``genewise.p.value`` is supplied with any adjust
    method other than "none", because ``ngenes`` is only defined in the
    branch that computes Simes p-values itself. pylimma uses
    ``ngenes = nrow(p)`` in both cases, which is what the surrounding
    code intends; results match R's function with that one line added.
    """
    method = _match_arg(method, _DECIDE_METHODS, "method")
    adjust_method = _match_arg(adjust_method, _ADJUST_METHODS, "adjust.method")
    if adjust_method == "fdr":
        adjust_method = "BH"

    if not (_is_anndata(data) or isinstance(data, dict)):
        return _decide_tests_default(
            data,
            method=method,
            adjust_method=adjust_method,
            p_value=p_value,
            lfc=lfc,
            coefficients=coefficients if coefficients is not None else tstat,
            genewise_p_value=genewise_p_value,
        )

    fit, _adata, _adata_key = _resolve_fit_input(data, key)
    ignored = [
        name
        for name, value in (
            ("coefficients", coefficients),
            ("cor_matrix", cor_matrix),
            ("tstat", tstat),
            ("genewise_p_value", genewise_p_value),
        )
        if value is not None
    ]
    if not np.all(np.isinf(np.asarray(df))):
        ignored.append("df")
    if ignored:
        warnings.warn(
            f"decide_tests ignores {', '.join(ignored)} for a fit object (as R's decideTests.MArrayLM does)",
            stacklevel=2,
        )

    # Auto-run e_bayes if not already run. For AnnData input, persist the
    # moderated fit so later top_table / treat calls see it.
    if "p_value" not in fit:
        from .ebayes import e_bayes

        fit = e_bayes(fit)
        if _adata is not None:
            # Plain dict for h5ad compatibility; see lm_fit.
            _adata.uns[_adata_key] = dict(fit)

    coef = fit.get("coefficients")
    coef = None if coef is None else np.asarray(coef, dtype=np.float64)

    if method in ("separate", "global"):
        p = np.array(fit["p_value"], dtype=np.float64)
        observed = ~np.isnan(p)
        if method == "separate":
            for j in range(p.shape[1]):
                p[observed[:, j], j] = p_adjust(p[observed[:, j], j], method=adjust_method)
        else:
            p[observed] = p_adjust(p[observed], method=adjust_method)
        is_de = np.where(observed, (p < p_value).astype(np.float64), np.nan)
        results = _sign_times(is_de, coef)
    else:
        f_p_value = np.asarray(fit["F_p_value"], dtype=np.float64)
        if np.isnan(f_p_value).any():
            if method == "hierarchical":
                raise ValueError("Can't handle NA p-values yet")
            raise ValueError("nestedF method can't handle NA p-values")
        selected = p_adjust(f_p_value, method=adjust_method) < p_value
        a = _cutoff_multiplier(adjust_method, selected.size, int(selected.sum()))
        results = np.zeros(np.shape(fit["t"]), dtype=np.float64)
        if selected.any():
            subset = _subset_fit_genes(fit, selected)
            if method == "hierarchical":
                results[selected, :] = _classify_tests_p(subset, p_value=p_value * a, method=adjust_method)
            else:
                results[selected, :] = classify_tests_f(subset, p_value=p_value * a)

    if lfc > 0:
        if coef is None:
            warnings.warn("lfc ignored because coefficients not found", stacklevel=2)
        else:
            results = results * np.where(np.isnan(coef), np.nan, np.abs(coef) > lfc)

    return _as_results(results)


def _subset_fit_genes(fit: dict, rows: np.ndarray) -> dict:
    """``object[rows, ]`` for the slots the classifiers read."""
    n_genes = np.shape(fit["t"])[0]
    subset = {"t": np.asarray(fit["t"])[rows, :], "cov_coefficients": fit.get("cov_coefficients")}
    for slot in ("df_residual", "df_prior"):
        value = fit.get(slot)
        if value is not None:
            value = np.asarray(value, dtype=np.float64)
            subset[slot] = value[rows] if value.ndim == 1 and value.size == n_genes else value
    return subset


def _classify_tests_p(fit: dict, p_value: float, method: str) -> np.ndarray:
    """Port of R limma's ``.classifyTestsP``: row-wise adjusted t-test p-values.

    P-values are recomputed from the t-statistics with
    ``df = df.residual + df.prior`` (not capped at the pooled df as in eBayes).
    """
    tstat = np.asarray(fit["t"], dtype=np.float64)
    df = np.inf
    if fit.get("df_residual") is not None:
        df = np.asarray(fit["df_residual"], dtype=np.float64)
    if fit.get("df_prior") is not None:
        df = df + np.asarray(fit["df_prior"], dtype=np.float64)
    df = np.broadcast_to(df, tstat.shape[:1])[:, np.newaxis]
    P = 2 * stats.t.sf(np.abs(tstat), df)
    results = np.empty_like(tstat)
    for i in range(tstat.shape[0]):
        observed = ~np.isnan(P[i])
        adjusted = np.full_like(P[i], np.nan)
        adjusted[observed] = p_adjust(P[i, observed], method=method)
        results[i] = _sign_times(np.where(observed, (adjusted < p_value).astype(np.float64), np.nan), tstat[i])
    return results


def _decide_tests_default(
    p,
    method: str,
    adjust_method: str,
    p_value: float,
    lfc: float,
    coefficients: np.ndarray | None,
    genewise_p_value: np.ndarray | None,
) -> np.ndarray:
    """Port of R limma's ``decideTests.default`` (a matrix of p-values)."""
    if method == "nestedF":
        raise ValueError("nestedF adjust method requires an MArrayLM object")
    p = np.array(p, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if np.isnan(p).any():
        # R: `if(any(p>1) || any(p<0))` fails on NA
        raise ValueError("p-values contain NA (R: missing value where TRUE/FALSE needed)")
    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("object doesn't appear to be a matrix of p-values")

    if method == "separate":
        for j in range(p.shape[1]):
            p[:, j] = p_adjust(p[:, j], method=adjust_method)
    elif method == "global":
        p = p_adjust(p.ravel(), method=adjust_method).reshape(p.shape)
    else:
        ngenes, ncontrasts = p.shape
        if genewise_p_value is None:
            simes = ncontrasts / np.arange(1, ncontrasts + 1)
            genewise_p_value = np.min(np.sort(p, axis=1) * simes, axis=1)
        # Intended rather than literal limma behaviour: R defines ngenes only
        # inside its Simes branch, so a supplied genewise.p.value fails for
        # every adjust.method except "none" (object 'ngenes' not found).
        # ngenes is nrow(p) in both cases here. See known_differences.rst.
        de_gene = p_adjust(np.asarray(genewise_p_value, dtype=np.float64), method=adjust_method) <= p_value
        p[~de_gene, :] = 1.0
        for g in np.flatnonzero(de_gene):
            p[g, :] = p_adjust(p[g, :], method=adjust_method)
        p_value = _cutoff_multiplier(adjust_method, ngenes, int(de_gene.sum())) * p_value

    is_de = (p <= p_value).astype(int)
    if coefficients is not None:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != p.shape:
            raise ValueError("dim(object) disagrees with dim(coefficients)")
        is_de[coefficients < 0] = -is_de[coefficients < 0]
        if lfc > 0:
            is_de[np.abs(coefficients) < lfc] = 0
    return is_de


def summarize_test_results(
    results: np.ndarray,
    coef_names: list[str] | None = None,
) -> dict:
    """
    Summarize test results by counting up/down/not significant genes.

    Provides similar functionality to R's summary.TestResults method.

    Parameters
    ----------
    results : ndarray
        Test results matrix from decide_tests(), with values -1, 0, 1.
        Shape (n_genes, n_coefficients).
    coef_names : list of str, optional
        Names for the coefficients (columns).

    Returns
    -------
    dict
        down : ndarray - count of genes with result -1 per coefficient
        not_sig : ndarray - count of genes with result 0 per coefficient
        up : ndarray - count of genes with result 1 per coefficient
        coef_names : list - coefficient names
        total : int - total number of genes

    Examples
    --------
    >>> results = decide_tests(fit)
    >>> summary = summarize_test_results(results)
    >>> print(f"Up: {summary['up']}, Down: {summary['down']}")
    """
    results = np.asarray(results)
    if results.ndim == 1:
        results = results.reshape(-1, 1)

    n_genes, n_coefs = results.shape

    if coef_names is None:
        coef_names = [f"coef_{i}" for i in range(n_coefs)]

    # Count each category, handling NaN
    down = np.sum(results == -1, axis=0)
    not_sig = np.sum(results == 0, axis=0)
    up = np.sum(results == 1, axis=0)

    return {
        "down": down,
        "not_sig": not_sig,
        "up": up,
        "coef_names": coef_names,
        "total": n_genes,
    }
