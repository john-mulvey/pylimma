# SPDX-License-Identifier: GPL-3.0-or-later
#
# This module is a Python port of code from R limma. Original R copyrights:
#   dups.R (unwrapdups, avedups, avereps, duplicateCorrelation)
#                              Copyright (C) 2002-2021 Gordon Smyth
#
# duplicate_correlation() additionally ports two routines from the R statmod
# package (which limma's duplicateCorrelation calls per gene):
#   statmod::mixedModel2Fit    Copyright (C) Gordon Smyth, Lizhong Chen;
#                              GPL-2 | GPL-3
#   statmod::glmgam.fit        Copyright (C) Gordon Smyth, Lizhong Chen;
#                              GPL-2 | GPL-3
# Python port: Copyright (C) 2026 John Mulvey
"""
Functions to handle duplicate spots or blocking.

Provides utilities for working with within-array replicate spots.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import linalg

from .classes import EList, get_eawp, _is_anndata


def _mixed_model_2_fit(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray | None = None,
    tol: float = 1e-6,
    maxit: int = 50,
) -> tuple[float, float] | None:
    """
    Fit mixed linear model and return variance components.

    This is a Python port of statmod::mixedModel2Fit from R.
    Uses moment-based REML estimation with gamma GLM refinement.

    Parameters
    ----------
    y : ndarray
        Response vector, shape (n,).
    X : ndarray
        Fixed effects design matrix, shape (n, p).
    Z : ndarray
        Random effects design matrix, shape (n, q).
    w : ndarray, optional
        Observation weights, shape (n,).
    tol : float, default 1e-6
        Convergence tolerance for gamma GLM.
    maxit : int, default 50
        Maximum iterations for gamma GLM.

    Returns
    -------
    tuple or None
        (residual_variance, block_variance) if successful, None otherwise.
    """
    # Apply weights if provided
    # NOTE: R's statmod::mixedModel2Fit applies sqrt(w) to y and X only, NOT to Z
    if w is not None:
        sw = np.sqrt(w)
        y = sw * y
        X = sw[:, np.newaxis] * X if X.ndim == 2 else sw * X

    X = np.atleast_2d(X)
    Z = np.atleast_2d(Z)
    mx = X.shape[0]
    nx = X.shape[1]
    nz = Z.shape[1]

    # Combine Z and y for joint projection
    # fit = lm.fit(X, cbind(Z, y))
    XtX = X.T @ X
    try:
        XtX_inv = linalg.pinv(XtX)
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        return None

    # QR decomposition of X
    Q, R, pivot = linalg.qr(X, pivoting=True)
    r = np.sum(np.abs(np.diag(R)) > 1e-10)  # rank

    if r == 0:
        return None

    mq = mx - r
    if mq == 0:
        return None

    # Project Z and y onto residual space (orthogonal to X)
    # QtZ = Q2' @ Z where Q2 is the null space of X
    Q2 = Q[:, r:]  # null space basis
    QtZ = Q2.T @ Z  # shape (mq, nz)
    Qty = Q2.T @ y  # shape (mq,)

    # SVD of QtZ with full U (matches R's La.svd(QtZ, nu=mq, nv=0)).
    # full_matrices=True returns U of shape (mq, mq) with a deterministic
    # null-space extension - using full_matrices=False and filling with
    # random vectors perturbs uqy for positions where d=0 and biases the
    # variance-components regression.
    try:
        U, s, Vh = linalg.svd(QtZ, full_matrices=True)
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        return None

    # Transform y to diagonal space
    uqy = U.T @ Qty  # shape (mq,)

    # Set up regression: dy ~ dx where dy = uqy^2, dx = [1, d^2]
    d = np.zeros(mq)
    d[:len(s)] = s ** 2

    dx = np.column_stack([np.ones(mq), d])
    dy = uqy ** 2

    # Initial OLS fit
    try:
        dfit_coef, _, _, _ = linalg.lstsq(dx, dy)
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        return None

    varcomp = dfit_coef
    dfitted = dx @ varcomp

    # Refine with gamma GLM if conditions met
    # if (mq > 2 && sum(abs(d) > 1e-15) > 1 && var(d) > 1e-15)
    if mq > 2 and np.sum(np.abs(d) > 1e-15) > 1 and np.var(d) > 1e-15:
        # Starting values
        if np.all(dfitted >= 0):
            start = varcomp.copy()
        else:
            start = np.array([np.mean(dy), 0.0])

        # Gamma GLM with log link using IRLS
        varcomp = _glmgam_fit(dx, dy, start, tol=tol, maxit=maxit)
        if varcomp is None:
            varcomp = dfit_coef

    # Return variance components: [residual, block]
    # Note: negative block variance is allowed (will give negative ICC)
    # The bounding to valid correlation range happens in duplicate_correlation
    return (float(varcomp[0]), float(varcomp[1]))


def _deviance_gamma(y: np.ndarray, mu: np.ndarray) -> float:
    """Gamma deviance matching statmod::glmgam.fit's internal deviance.gamma."""
    if np.any(mu < 0):
        return np.inf
    o = (y < 1e-15) & (mu < 1e-15)
    if np.all(o):
        return 0.0
    if np.any(o):
        y1 = y[~o]
        mu1 = mu[~o]
        return 2.0 * np.sum((y1 - mu1) / mu1 - np.log(y1 / mu1))
    return 2.0 * np.sum((y - mu) / mu - np.log(y / mu))


def _glmgam_fit(
    X: np.ndarray,
    y: np.ndarray,
    start: np.ndarray,
    tol: float = 1e-6,
    maxit: int = 50,
) -> np.ndarray | None:
    """
    Fit gamma GLM with identity link via damped Fisher scoring.

    Faithful port of statmod::glmgam.fit: Levenberg-Marquardt damping on the
    Fisher information with a deviance-based line search. The simpler
    unweighted-IRLS approach drifts by ~1e-5 per fit compared to R because it
    skips the damping/line-search step.

    Parameters
    ----------
    X : ndarray
        Design matrix, shape (n, p).
    y : ndarray
        Non-negative response vector, shape (n,).
    start : ndarray
        Starting coefficients, shape (p,).
    tol : float
        Convergence tolerance on inner product of score and step.
    maxit : int
        Maximum outer iterations.
    """
    X = np.atleast_2d(X)
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64)

    if np.any(y < 0):
        raise ValueError("y must be non-negative")
    maxy = np.max(y) if y.size else 0.0
    if maxy == 0.0:
        return np.zeros(p)

    beta = np.asarray(start, dtype=np.float64).copy()
    mu = X @ beta
    if np.any(mu < 0):
        return None

    dev = _deviance_gamma(y, mu)

    I_p = np.eye(p)
    lambda_ = 0.0  # set on first iteration

    for iteration in range(1, maxit + 1):
        v = mu ** 2
        v = np.maximum(v, np.max(v) / 1e3)
        XVX = X.T @ (X / v[:, np.newaxis])
        maxinfo = np.max(np.diag(XVX))

        if iteration == 1:
            lambda_ = abs(np.mean(np.diag(XVX))) / p

        dl = X.T @ ((y - mu) / v)
        beta_old = beta
        dev_old = dev

        # Damped step with line search on deviance
        lev = 0
        dbeta = np.zeros(p)
        while True:
            lev += 1
            try:
                L = linalg.cholesky(XVX + lambda_ * I_p, lower=False)
            except linalg.LinAlgError:
                return None
            # Solve (R^T R) dbeta = dl via two triangular solves
            tmp = linalg.solve_triangular(L.T, dl, lower=True)
            dbeta = linalg.solve_triangular(L, tmp, lower=False)
            beta = beta_old + dbeta
            mu = X @ beta
            dev = _deviance_gamma(y, mu)
            if dev <= dev_old or (np.max(mu) > 0 and dev / np.max(mu) < 1e-15):
                break
            if lambda_ / maxinfo > 1e15:
                beta = beta_old
                break
            lambda_ *= 2.0

        if lambda_ / maxinfo > 1e15:
            break
        if lev == 1:
            lambda_ /= 10.0
        if dl @ dbeta < tol or (np.max(mu) > 0 and dev / np.max(mu) < 1e-15):
            break
        if iteration >= maxit:
            break

    return beta


def unwrap_dups(
    M: np.ndarray,
    ndups: int = 2,
    spacing: int = 1,
) -> np.ndarray:
    """
    Unwrap duplicate spots to long format.

    Reshapes expression matrix so that all spots for a given gene are in
    one row, with duplicate measurements in separate columns.

    Parameters
    ----------
    M : ndarray
        Expression matrix, shape (n_spots, n_arrays).
    ndups : int, default 2
        Number of duplicates.
    spacing : int, default 1
        Spacing between duplicates in the original matrix.

    Returns
    -------
    ndarray
        Unwrapped matrix, shape (n_genes, n_arrays * ndups).

    Notes
    -----
    This function matches R limma's unwrapdups behaviour. The output has
    columns ordered as: [dup0-array0, dup1-array0, ..., dup0-array1, ...].
    """
    if ndups == 1:
        return M

    M = np.asarray(M, dtype=np.float64)
    n_spots, n_arrays = M.shape
    n_groups = n_spots // ndups // spacing

    # Use Fortran order to match R's reshape behaviour
    # R: dim(M) <- c(spacing, ndups, ngroups, nslides)
    M = M.reshape((spacing, ndups, n_groups, n_arrays), order='F')

    # R: aperm(M, c(1,3,2,4))
    M = np.transpose(M, (0, 2, 1, 3))

    # R: dim(M) <- c(spacing*ngroups, ndups*nslides)
    M = M.reshape((spacing * n_groups, ndups * n_arrays), order='F')

    return M


def unique_genelist(
    genelist: np.ndarray | pd.DataFrame | list,
    ndups: int = 2,
    spacing: int = 1,
) -> np.ndarray | pd.DataFrame | list:
    """
    Extract unique gene identifiers from duplicate spots.

    Eliminates entries in genelist that correspond to duplicate spots,
    keeping only the first entry for each gene.

    Parameters
    ----------
    genelist : array_like, DataFrame, or list
        Gene identifiers or annotation.
    ndups : int, default 2
        Number of duplicates.
    spacing : int, default 1
        Spacing between duplicates.

    Returns
    -------
    array_like, DataFrame, or list
        Gene list with duplicates removed.
    """
    if ndups <= 1:
        return genelist

    # Get indices of first duplicate
    n = len(genelist) if hasattr(genelist, "__len__") else genelist.shape[0]
    indices = np.arange(n)
    unwrapped = unwrap_dups(indices.reshape(-1, 1), ndups=ndups, spacing=spacing)
    keep_idx = unwrapped[:, 0].astype(int)

    if isinstance(genelist, pd.DataFrame):
        return genelist.iloc[keep_idx].reset_index(drop=True)
    elif isinstance(genelist, (list, tuple)):
        return [genelist[i] for i in keep_idx]
    else:
        genelist = np.asarray(genelist)
        if genelist.ndim == 1:
            return genelist[keep_idx]
        else:
            return genelist[keep_idx, :]


def ave_dups(
    x: np.ndarray,
    ndups: int = 2,
    spacing: int = 1,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Average over duplicate spots.

    Computes the (weighted) mean across replicate spots for each gene.

    Parameters
    ----------
    x : ndarray
        Expression matrix, shape (n_spots, n_arrays).
    ndups : int, default 2
        Number of duplicates.
    spacing : int, default 1
        Spacing between duplicates.
    weights : ndarray, optional
        Weights matrix with same shape as x.

    Returns
    -------
    ndarray
        Averaged matrix, shape (n_genes, n_arrays).
    """
    if ndups == 1:
        return x

    x = np.asarray(x, dtype=np.float64)
    n_spots, n_arrays = x.shape
    n_groups = n_spots // ndups // spacing

    # Use Fortran order to match R's reshape behaviour
    # R: dim(x) <- c(spacing, ndups, ngroups*nslides)
    x = x.reshape((spacing, ndups, n_groups * n_arrays), order='F')

    # R: aperm(x, c(2,1,3))
    x = np.transpose(x, (1, 0, 2))

    if weights is None:
        # Simple mean along dups axis (axis 0)
        result = np.nanmean(x, axis=0)
    else:
        # Weighted mean
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights.reshape((spacing, ndups, n_groups * n_arrays), order='F')
        weights = np.transpose(weights, (1, 0, 2))

        # Handle NaN and negative weights
        weights = weights.copy()
        weights[np.isnan(weights) | np.isnan(x)] = 0
        weights[weights < 0] = 0

        with np.errstate(invalid='ignore', divide='ignore'):
            result = np.nansum(weights * x, axis=0) / np.sum(weights, axis=0)

    # Reshape back: R: dim(x) <- c(spacing*ngroups, nslides)
    result = result.reshape((spacing * n_groups, n_arrays), order='F')
    return result


def avereps(
    x,
    ID: np.ndarray | list | None = None,
) -> np.ndarray | pd.DataFrame:
    """
    Average over irregular replicate probes.

    Computes the mean across replicate probes identified by ID,
    mirroring R's ``limma::avereps``. When ``x`` carries row labels
    (pandas DataFrame index or Series ``.name``) and ``ID`` is not
    supplied, the row labels are used as probe IDs - the same default
    as R's ``avereps.default(x, ID=rownames(x))``.

    Parameters
    ----------
    x : ndarray or DataFrame
        Expression matrix, shape (n_probes, n_arrays).
    ID : array_like, optional
        Probe identifiers. Probes with the same ID are averaged. If
        None and ``x`` is a DataFrame, ``x.index`` is used. If no
        source of IDs is available, raises ``ValueError`` matching
        R's ``"No probe IDs"`` error.

    Returns
    -------
    ndarray or DataFrame
        Matrix of averaged rows, one per unique ID in order of first
        appearance. Returned as a DataFrame (indexed by the unique
        IDs, columns preserved) when ``x`` was a DataFrame, otherwise
        as an ndarray. To also recover the unique ID vector from an
        ndarray return value, read it off with
        ``np.unique(ID, return_index=True)`` or pass ``x`` as a
        DataFrame.

    Examples
    --------
    R parity (matrix-return):

    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> ID = ["A", "A", "B"]
    >>> avereps(x, ID)
    array([[2., 3.],
           [5., 6.]])

    DataFrame-in, DataFrame-out (ID defaults to index):

    >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=["A", "A", "B"])
    >>> avereps(df)

    For ``AnnData`` input, ``ID`` defaults to ``adata.var_names`` and a
    new ``AnnData`` is returned with the var axis collapsed to the
    unique ids. As with :func:`aver_arrays`, the sample-count vs
    gene-count shape change means in-place mutation via a layer is not
    possible, so AnnData-in returns a value.
    """
    if _is_anndata(x):
        return _avereps_anndata(x, ID=ID)

    # EList dispatch (mirrors R's avereps.EList at dups.R:267-285).
    # Averages E and weights, deduplicates genes, drops printer. Default
    # ID is y$genes$ID, then rownames (we use the genes DataFrame index),
    # matching R's precedence.
    if isinstance(x, EList):
        return _avereps_elist(x, ID=ID)

    is_df = isinstance(x, pd.DataFrame)
    columns = x.columns if is_df else None

    if ID is None:
        if is_df:
            ID = np.asarray(x.index)
        else:
            raise ValueError(
                "No probe IDs: pass ID=... or supply x as a DataFrame "
                "whose index holds the probe identifiers. Matches R's "
                "'No probe IDs' stop() when both ID and rownames(x) are NULL."
            )

    arr = np.asarray(x, dtype=np.float64)
    ID = np.asarray(ID)

    # Unique IDs in order of first appearance (matches R's
    # factor(ID, levels=unique(ID)) combined with rowsum(reorder=FALSE)).
    _, idx = np.unique(ID, return_index=True)
    unique_ids = ID[np.sort(idx)]

    result = np.zeros((len(unique_ids), arr.shape[1]))
    for i, uid in enumerate(unique_ids):
        mask = ID == uid
        if mask.sum() == 1:
            result[i, :] = arr[mask, :][0]
        else:
            result[i, :] = np.nanmean(arr[mask, :], axis=0)

    if is_df:
        return pd.DataFrame(result, index=unique_ids, columns=columns)
    return result


def _avereps_elist(el, ID=None):
    """EList dispatch for avereps (R: avereps.EList at dups.R:267-285).

    Averages ``E`` and ``weights`` across replicate probes, deduplicates
    ``genes``, and drops ``printer``. Default ``ID`` is ``y$genes$ID`` if
    present, otherwise the genes DataFrame's index, otherwise R's error.
    Returns a new EList with the probe axis collapsed.
    """
    E = np.asarray(el["E"], dtype=np.float64)
    weights = el.get("weights")
    genes = el.get("genes")

    if ID is None:
        if isinstance(genes, pd.DataFrame) and "ID" in genes.columns:
            ID = genes["ID"].to_numpy()
        elif isinstance(genes, pd.DataFrame):
            ID = np.asarray(genes.index)
        else:
            raise ValueError(
                "No probe IDs: pass ID=... or populate EList['genes'] "
                "with an 'ID' column or a meaningful index."
            )
    ID = np.asarray(ID)
    if ID.shape[0] != E.shape[0]:
        raise ValueError(
            f"length of ID ({ID.shape[0]}) must match number of probes "
            f"({E.shape[0]})"
        )

    # Average E (delegate to the matrix path for the numerics).
    E_new = avereps(E, ID=ID)

    # Average weights if present (matches R's avereps.EList line 274).
    weights_new = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape == E.shape:
            weights_new = avereps(w_arr, ID=ID)

    # Deduplicate genes (matches R's avereps.EList line 275-278).
    _, first_idx = np.unique(ID, return_index=True)
    keep = np.sort(first_idx)
    genes_new = None
    if isinstance(genes, pd.DataFrame):
        genes_new = genes.iloc[keep].reset_index(drop=True)

    # Build the new EList - copy original, overwrite collapsed slots,
    # drop 'printer' (R's avereps.EList line 280).
    out_dict = {k: v for k, v in el.items() if k != "printer"}
    out_dict["E"] = E_new
    if weights_new is not None:
        out_dict["weights"] = weights_new
    if genes_new is not None:
        out_dict["genes"] = genes_new
    return EList(out_dict)


def _avereps_anndata(adata, ID=None):
    """AnnData dispatch for avereps.

    Averages over duplicate probes (var rows) identified by ``ID``
    (defaulting to ``adata.var_names``) and returns a new AnnData with
    the var axis collapsed to the unique ids in order of first
    appearance. Sample (obs) axis is preserved.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required for avereps(AnnData) but is not installed"
        ) from exc

    # get_eawp densifies and transposes to limma's (n_probes, n_samples).
    eawp = get_eawp(adata)
    E = np.asarray(eawp["exprs"], dtype=np.float64)

    if ID is None:
        ID = np.asarray(adata.var_names)
    ID = np.asarray(ID)
    if ID.shape[0] != E.shape[0]:
        raise ValueError(
            f"length of ID ({ID.shape[0]}) must match number of "
            f"probes ({E.shape[0]})"
        )

    # Delegate the averaging to the ndarray path; output is
    # (n_unique_probes, n_samples).
    averaged = avereps(E, ID=ID)

    # First-occurrence ordering of the unique ids, for picking
    # representative var rows in the collapsed AnnData.
    _, first_idx = np.unique(ID, return_index=True)
    keep_order = np.sort(first_idx)
    new_var_names = np.asarray(ID)[keep_order]

    if adata.var is not None and len(adata.var.columns):
        var_new = adata.var.iloc[keep_order].copy()
    else:
        var_new = pd.DataFrame(index=pd.Index(new_var_names))
    var_new.index = pd.Index(new_var_names)

    obs_new = adata.obs.copy() if adata.obs is not None else None

    # averaged is (n_unique_probes, n_samples); AnnData wants
    # (n_samples, n_vars).
    return ad.AnnData(X=np.asarray(averaged).T, obs=obs_new, var=var_new)


def duplicate_correlation(
    M,
    design: np.ndarray | None = None,
    ndups: int = 2,
    spacing: int | str = 1,
    block: np.ndarray | None = None,
    trim: float = 0.15,
    weights: np.ndarray | None = None,
    *,
    layer: str | None = None,
    weights_layer: str | None = None,
) -> dict:
    """
    Estimate correlation between duplicate spots or blocked samples.

    Estimates the intra-block correlation using a mixed linear model,
    computed separately for each gene and then averaged using Fisher's
    z-transformation.

    Parameters
    ----------
    M : ndarray
        Expression matrix, shape (n_genes, n_samples) or (n_spots, n_arrays).
    design : ndarray, optional
        Design matrix. If None, uses intercept-only model.
    ndups : int, default 2
        Number of within-array duplicate spots.
    spacing : int or str, default 1
        Spacing between duplicates. Can be an integer or one of:
        - "columns": spacing of 1 (duplicates in adjacent rows)
        - "topbottom": spacing of n_spots/2 (duplicates in top/bottom halves)
    block : array_like, optional
        Block indicator for correlated samples. If provided, ndups and
        spacing are ignored.
    trim : float, default 0.15
        Trimmed mean proportion for consensus correlation.
    weights : ndarray, optional
        Observation weights.

    Returns
    -------
    dict
        consensus_correlation : float
            Consensus correlation (trimmed mean on Fisher z scale).
        cor : float
            Same as consensus_correlation (for compatibility).
        atanh_correlations : ndarray
            Gene-wise correlations on Fisher z scale.

    Notes
    -----
    This function uses REML estimation of variance components in a
    mixed linear model. It requires the MixedLM functionality from
    statsmodels.

    The consensus correlation should be used as the `correlation`
    argument to `gls_series()`.

    References
    ----------
    Smyth, G. K., Michaud, J. and Scott, H. S. (2005). Use of within-array
    replicate spots for assessing differential expression in microarray
    experiments. Bioinformatics, 21, 2067-2075.
    """
    from scipy.stats import trim_mean

    # Polymorphic input: ndarray / dict / EList / AnnData.
    eawp = get_eawp(M, layer=layer, weights_layer=weights_layer)
    M = np.asarray(eawp["exprs"], dtype=np.float64)
    if design is None and eawp.get("design") is not None:
        design = np.asarray(eawp["design"], dtype=np.float64)
    if weights is None and eawp.get("weights") is not None:
        weights = np.asarray(eawp["weights"], dtype=np.float64)
    n_genes, n_arrays = M.shape

    # Handle spacing shortcuts (R parity)
    if isinstance(spacing, str):
        if spacing == "columns":
            spacing = 1
        elif spacing == "topbottom":
            spacing = n_genes // 2
        else:
            raise ValueError(f"Unknown spacing shortcut: {spacing}. Use 'columns' or 'topbottom'.")

    # Check design
    if design is None:
        design = np.ones((n_arrays, 1))
    design = np.asarray(design, dtype=np.float64)
    n_coefs = design.shape[1]

    # Normalise weight shape through asMatrixWeights; helper returns
    # a fresh copy so the subsequent NaN writes are safe.
    if weights is not None:
        from .classes import _as_matrix_weights
        weights = _as_matrix_weights(weights, (n_genes, n_arrays))
        weights[weights <= 0] = np.nan
        M = M.copy()
        M[~np.isfinite(weights)] = np.nan

    # Check if block is already encoded in design matrix
    if block is not None:
        # One-hot encode block (sans intercept, like R's model.matrix)
        unique_blocks = np.unique(block)
        if len(unique_blocks) > 1:
            block_indicators = np.zeros((len(block), len(unique_blocks) - 1))
            for j, b in enumerate(unique_blocks[1:]):
                block_indicators[:, j] = (block == b).astype(float)

            # QR decomposition of design
            q, r = np.linalg.qr(design, mode='complete')
            rank = np.sum(np.abs(np.diag(r[:design.shape[1], :])) > 1e-10)

            # Project block indicators through Q'
            qt_block = q.T @ block_indicators

            # Check if projection below design rank is essentially zero
            if rank < q.shape[0] and np.max(np.abs(qt_block[rank:, :])) < 1e-8:
                warnings.warn(
                    "Block factor already encoded in the design matrix: "
                    "setting intrablock correlation to zero."
                )
                return {
                    "consensus_correlation": 0.0,
                    "cor": 0.0,
                    "atanh_correlations": np.zeros(n_genes),
                }

    # Setup blocking structure
    if block is None:
        # Within-array duplicates
        if ndups < 2:
            warnings.warn("No duplicates: setting correlation to zero.")
            return {
                "consensus_correlation": 0.0,
                "cor": 0.0,
                "atanh_correlations": np.zeros(n_genes),
            }

        # Unwrap duplicates
        M = unwrap_dups(M, ndups=ndups, spacing=spacing)
        if weights is not None:
            weights = unwrap_dups(weights, ndups=ndups, spacing=spacing)
        design = np.kron(design, np.ones((ndups, 1)))
        n_genes, n_samples = M.shape

        # Array indicator (block = array)
        Array = np.repeat(np.arange(n_arrays), ndups)
    else:
        # Sample blocking
        block = np.asarray(block)
        if len(block) != n_arrays:
            raise ValueError("Length of block does not match number of arrays")
        Array = block
        n_samples = n_arrays

        # Check for singleton blocks
        unique_blocks, counts = np.unique(block, return_counts=True)
        max_block_size = np.max(counts)
        if max_block_size == 1:
            warnings.warn("Blocks all of size 1: setting correlation to zero.")
            return {
                "consensus_correlation": 0.0,
                "cor": 0.0,
                "atanh_correlations": np.zeros(n_genes),
            }

    # Compute gene-wise correlations using mixed model
    rho = np.full(n_genes, np.nan)

    for i in range(n_genes):
        y = M[i, :]
        obs = np.isfinite(y)
        n_obs = np.sum(obs)

        if n_obs <= n_coefs + 2:
            continue

        y_obs = y[obs]
        X = design[obs, :]
        groups = Array[obs]

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        if n_groups <= 1 or n_groups >= n_obs - 1:
            continue

        # Build Z matrix (block indicator, like R's model.matrix(~0+A))
        Z = np.zeros((n_obs, n_groups))
        for j, g in enumerate(unique_groups):
            Z[:, j] = (groups == g).astype(float)

        # Get observation weights if available
        w = None
        if weights is not None:
            w = weights[i, obs] if weights.ndim == 2 else weights[obs]
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                w = None

        # Fit mixed model using statmod-compatible algorithm
        result = _mixed_model_2_fit(y_obs, X, Z, w=w, maxit=20)

        if result is not None:
            var_resid, var_block = result
            total_var = var_resid + var_block
            if total_var > 0:
                rho[i] = var_block / total_var

    # Bound correlations to ensure positive-definite correlation matrix
    rho_max = 0.99
    if block is None:
        rho_min = 1 / (1 - ndups) + 0.01
    else:
        rho_min = 1 / (1 - max_block_size) + 0.01

    rho = np.clip(rho, rho_min, rho_max)

    # Fisher z-transformation
    arho = np.arctanh(rho)

    # Trimmed mean on z scale
    valid = np.isfinite(arho)
    if np.sum(valid) > 0:
        mrho = np.tanh(trim_mean(arho[valid], trim))
    else:
        mrho = 0.0

    return {
        "consensus_correlation": float(mrho),
        "cor": float(mrho),
        "atanh_correlations": arho,
    }
