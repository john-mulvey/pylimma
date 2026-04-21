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

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from .utils import p_adjust
from .classes import _resolve_fit_input, _is_anndata

if TYPE_CHECKING:
    from anndata import AnnData


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
        diag_vals = np.diag(cov)
        if np.min(diag_vals) == 0:
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

    # Compute F-statistic
    tQ = tstat @ Q
    fstat = np.sum(tQ**2, axis=1)

    df2 = df

    if fstat_only:
        return fstat, r, df2

    # Classification using step-down procedure. stats.f.ppf with
    # vector df2 returns a vector of per-gene thresholds.
    qF = stats.f.ppf(1 - p_value, r, df2)
    qF = np.asarray(qF)
    if qF.ndim == 0:
        qF = np.full(n_genes, float(qF))

    result = np.zeros((n_genes, n_tests), dtype=float)

    for i in range(n_genes):
        x = tstat[i, :]
        if np.any(np.isnan(x)):
            result[i, :] = np.nan  # R sets to NA, not 0
            continue

        # Check if overall F-test is significant
        if (x @ Q @ Q.T @ x) > qF[i]:
            # Order by absolute t-statistic
            order = np.argsort(np.abs(x))[::-1]
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
    """
    Classify genes as differentially expressed.

    Applies multiple testing correction and classifies each gene-coefficient
    combination as up-regulated (1), down-regulated (-1), or not significant (0).

    Parameters
    ----------
    data : AnnData, dict, or ndarray
        Fit object from e_bayes(), or a matrix of p-values.
        If AnnData, reads from adata.uns[key].
    method : str, default "separate"
        Method for multiple testing correction:

        - "separate": adjust p-values for each coefficient separately
        - "global": adjust all p-values together
        - "hierarchical": first test overall significance (F-test), then adjust within significant genes
        - "nestedF": use nested F-tests for multiple contrasts
    adjust_method : str, default "BH"
        P-value adjustment method: "BH", "bonferroni", "holm", "BY", "none".
    p_value : float, default 0.05
        Significance threshold for adjusted p-values.
    lfc : float, default 0.0
        Log fold-change threshold. Genes with \|logFC\| < lfc are set to 0.
    key : str, default "pylimma"
        Key for fit results in adata.uns (AnnData input only).

    Returns
    -------
    ndarray
        Matrix of test results with values -1 (down), 0 (not significant), 1 (up).
        Shape is (n_genes, n_coefficients).

    Notes
    -----
    The "separate" method is the default and most commonly used. It adjusts
    p-values within each coefficient independently.

    The "hierarchical" and "nestedF" methods first perform an overall F-test
    for each gene, then test individual contrasts only for genes that pass
    the F-test. This can increase power when there are many contrasts.

    Examples
    --------
    >>> fit = e_bayes(lm_fit(expr, design))
    >>> results = decide_tests(fit, p_value=0.05, lfc=1)
    >>> np.sum(results == 1, axis=0)  # count up-regulated per coefficient
    """
    # Dispatch: AnnData / dict / fallback ndarray of p-values.
    # When a bare ndarray is passed, R's decideTests.default also
    # accepts coefficients / cor.matrix / tstat / df / genewise.p.value
    # as keyword args so callers can decide without a fit dict.
    if _is_anndata(data) or isinstance(data, dict):
        fit, _adata, _adata_key = _resolve_fit_input(data, key)
    else:
        p = np.asarray(data)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return _decide_tests_p(
            p, method=method, adjust_method=adjust_method,
            p_value=p_value,
            coefficients=coefficients if coefficients is not None else tstat,
            lfc=lfc,
            genewise_p_value=genewise_p_value,
        )

    # Auto-run e_bayes if not already run (R parity). When the input
    # was AnnData, persist the moderated fit back to adata.uns[key] so
    # subsequent top_table / treat / contrasts_fit calls on the same
    # adata see the eBayes-augmented slots. Without the write-back,
    # top_table(adata) would raise "Need to run e_bayes() first" even
    # though decide_tests implicitly ran it.
    if "p_value" not in fit:
        from .ebayes import e_bayes
        fit = e_bayes(fit)
        if _adata is not None:
            # Plain dict for h5ad compatibility; see lm_fit.
            _adata.uns[_adata_key] = dict(fit)

    p = fit["p_value"]
    # Explicit kwargs override fit-derived values, matching R's
    # decideTests.default where any of coefficients/tstat/cor.matrix/
    # df/genewise.p.value takes precedence over the object slots.
    if coefficients is None:
        coefficients = fit.get("coefficients")
    if tstat is None:
        tstat = fit.get("t")

    if method == "nestedF":
        return _decide_tests_nested_f(
            fit, adjust_method=adjust_method, p_value=p_value, lfc=lfc,
            cor_matrix=cor_matrix, df=df,
        )
    elif method == "hierarchical":
        return _decide_tests_hierarchical(
            fit, adjust_method=adjust_method, p_value=p_value, lfc=lfc,
            genewise_p_value=genewise_p_value,
        )
    else:
        return _decide_tests_p(
            p, method=method, adjust_method=adjust_method,
            p_value=p_value, coefficients=coefficients, lfc=lfc,
            genewise_p_value=genewise_p_value,
        )


def _decide_tests_p(
    p: np.ndarray,
    method: str,
    adjust_method: str,
    p_value: float,
    coefficients: np.ndarray | None,
    lfc: float,
    genewise_p_value: np.ndarray | None = None,
) -> np.ndarray:
    """Decide tests from a matrix of p-values."""
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    n_genes, n_coefs = p.shape

    # Validate p-values
    if np.any((p > 1) | (p < 0)):
        raise ValueError("p-values must be between 0 and 1")

    # Adjust p-values
    if method == "separate":
        p_adj = np.zeros_like(p)
        for j in range(n_coefs):
            valid = ~np.isnan(p[:, j])
            p_adj[valid, j] = p_adjust(p[valid, j], method=adjust_method)
            p_adj[~valid, j] = np.nan
    elif method == "global":
        valid = ~np.isnan(p)
        p_adj = np.full_like(p, np.nan)
        p_adj[valid] = p_adjust(p[valid], method=adjust_method)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Classification
    is_de = (p_adj <= p_value).astype(int)

    if coefficients is not None:
        coefficients = np.asarray(coefficients)
        if coefficients.shape != p.shape:
            raise ValueError("coefficients and p-values have different shapes")
        # Apply sign
        is_de = is_de * np.sign(coefficients).astype(int)
        # Apply lfc threshold
        if lfc > 0:
            is_de[np.abs(coefficients) < lfc] = 0

    return is_de


def _decide_tests_hierarchical(
    fit: dict,
    adjust_method: str,
    p_value: float,
    lfc: float,
    genewise_p_value: np.ndarray | None = None,
) -> np.ndarray:
    """Hierarchical testing: F-test first, then individual tests."""
    p = fit["p_value"]
    coefficients = fit.get("coefficients")
    # Explicit genewise_p_value override takes precedence over the
    # fit's F.p.value (matches R decideTests.default lines 57-67).
    if genewise_p_value is None:
        f_p_value = fit.get("F_p_value")
    else:
        f_p_value = np.asarray(genewise_p_value, dtype=np.float64)

    if f_p_value is None:
        raise ValueError("F-test p-values not found. Run e_bayes() with multiple coefficients.")

    if np.any(np.isnan(f_p_value)):
        raise ValueError("Cannot handle NA F p-values in hierarchical method")

    n_genes = p.shape[0]

    # Adjust F-test p-values
    f_adj = p_adjust(f_p_value, method=adjust_method)
    de_gene = f_adj < p_value

    # Count DE genes for adjusting threshold
    n_de = np.sum(de_gene)
    n_total = np.sum(~np.isnan(f_p_value))

    # Adjust p-value cutoff based on number of DE genes
    if adjust_method.lower() in ("bh", "fdr"):
        a = n_de / n_total if n_total > 0 else 1
    elif adjust_method.lower() == "bonferroni":
        a = 1 / n_total if n_total > 0 else 1
    elif adjust_method.lower() == "holm":
        a = 1 / (n_total - n_de + 1) if n_total > n_de else 1
    elif adjust_method.lower() == "by":
        a = n_de / n_total / np.sum(1 / np.arange(1, n_total + 1)) if n_total > 0 else 1
    else:
        a = 1

    p_cutoff = a * p_value

    # Initialize result
    result = np.zeros_like(p, dtype=int)

    # For DE genes, adjust p-values row-wise
    de_idx = np.where(de_gene)[0]
    for i in de_idx:
        p_row = p_adjust(p[i, :], method=adjust_method)
        sig = p_row < p_cutoff
        if coefficients is not None:
            result[i, sig] = np.sign(coefficients[i, sig]).astype(int)
        else:
            result[i, sig] = 1

    # Apply lfc threshold
    if lfc > 0 and coefficients is not None:
        result[np.abs(coefficients) < lfc] = 0

    return result


def _decide_tests_nested_f(
    fit: dict,
    adjust_method: str,
    p_value: float,
    lfc: float,
    cor_matrix: np.ndarray | None = None,
    df: float | np.ndarray = np.inf,
) -> np.ndarray:
    """Nested F-test method for multiple contrasts."""
    f_p_value = fit.get("F_p_value")
    coefficients = fit.get("coefficients")

    if f_p_value is None:
        raise ValueError("F-test p-values not found. Run e_bayes() with multiple coefficients.")

    if np.any(np.isnan(f_p_value)):
        raise ValueError("nestedF method cannot handle NA p-values")

    n_genes = len(f_p_value)

    # Adjust F-test p-values
    f_adj = p_adjust(f_p_value, method=adjust_method)
    de_gene = f_adj < p_value

    n_de = np.sum(de_gene)
    n_total = np.sum(~np.isnan(f_p_value))

    # Adjust p-value cutoff
    if adjust_method.lower() in ("bh", "fdr"):
        a = n_de / n_total if n_total > 0 else 1
    elif adjust_method.lower() == "bonferroni":
        a = 1 / n_total if n_total > 0 else 1
    elif adjust_method.lower() == "holm":
        a = 1 / (n_total - n_de + 1) if n_total > n_de else 1
    elif adjust_method.lower() == "by":
        a = n_de / n_total / np.sum(1 / np.arange(1, n_total + 1)) if n_total > 0 else 1
    else:
        a = 1

    p_cutoff = a * p_value

    # Initialize result
    n_coefs = fit["t"].shape[1]
    result = np.zeros((n_genes, n_coefs), dtype=int)

    # For DE genes, use classify_tests_f
    if np.any(de_gene):
        # Create subset fit for DE genes
        de_idx = np.where(de_gene)[0]
        fit_subset = {
            "t": fit["t"][de_idx, :],
            "cov_coefficients": fit.get("cov_coefficients"),
            "df_prior": fit.get("df_prior"),
            "df_residual": (
                fit["df_residual"][de_idx]
                if isinstance(fit["df_residual"], np.ndarray)
                else fit["df_residual"]
            ),
        }
        # Forward explicit cor_matrix / df overrides so nestedF honours
        # R's decideTests(..., cor.matrix=..., df=...) call pattern.
        result[de_idx, :] = classify_tests_f(
            fit_subset,
            cor_matrix=cor_matrix,
            df=df,
            p_value=p_cutoff,
        )

    # Apply lfc threshold
    if lfc > 0 and coefficients is not None:
        result[np.abs(coefficients) < lfc] = 0

    return result


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
