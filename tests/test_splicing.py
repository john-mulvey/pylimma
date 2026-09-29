"""
R parity tests for diff_splice, top_splice and plot_splice (differential splicing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib  # noqa: F401

    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FIXTURES = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------
# diff_splice
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _diffsplice_fit():
    from pylimma import diff_splice, lm_fit

    y = pd.read_csv(FIXTURES / "R_diffSplice_input_y.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_diffSplice_input_design.csv").values
    fit = lm_fit(y, design=design)
    n_exons = y.shape[0]
    fit["genes"] = pd.DataFrame(
        {
            "GeneID": np.repeat([f"gene{i + 1}" for i in range(20)], 5),
            "ExonID": [f"exon{i + 1}" for i in range(n_exons)],
        }
    )
    return diff_splice(fit, geneid="GeneID", exonid="ExonID", verbose=False)


def test_diffsplice_coefficients(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_coefficients.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["coefficients"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_t(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_t.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["t"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_p(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_p.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["p_value"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_gene_F(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_gene_F.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["gene_F"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_gene_F_p(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_gene_F_p.csv", index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit["gene_F_p_value"], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_gene_simes(_diffsplice_fit):
    expected = pd.read_csv(FIXTURES / "R_diffSplice_gene_simes_p.csv", index_col=0).values
    np.testing.assert_allclose(
        _diffsplice_fit["gene_simes_p_value"], expected, rtol=1e-6, atol=1e-9
    )


@pytest.fixture(scope="module")
def _diffsplice_fit_legacy():
    from pylimma import diff_splice, lm_fit

    y = pd.read_csv(FIXTURES / "R_diffSplice_input_y.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_diffSplice_input_design.csv").values
    fit = lm_fit(y, design=design)
    n_exons = y.shape[0]
    fit["genes"] = pd.DataFrame(
        {
            "GeneID": np.repeat([f"gene{i + 1}" for i in range(20)], 5),
            "ExonID": [f"exon{i + 1}" for i in range(n_exons)],
        }
    )
    return diff_splice(fit, geneid="GeneID", exonid="ExonID", legacy=True, verbose=False)


@pytest.mark.parametrize(
    "key,file",
    [
        ("coefficients", "R_diffSplice_legacy_coefficients.csv"),
        ("t", "R_diffSplice_legacy_t.csv"),
        ("p_value", "R_diffSplice_legacy_p.csv"),
        ("gene_F", "R_diffSplice_legacy_gene_F.csv"),
        ("gene_F_p_value", "R_diffSplice_legacy_gene_F_p.csv"),
        ("gene_simes_p_value", "R_diffSplice_legacy_gene_simes_p.csv"),
    ],
)
def test_diffsplice_legacy_rparity(_diffsplice_fit_legacy, key, file):
    expected = pd.read_csv(FIXTURES / file, index_col=0).values
    np.testing.assert_allclose(_diffsplice_fit_legacy[key], expected, rtol=1e-6, atol=1e-9)


def test_diffsplice_anndata_matches_ndarray():
    """diff_splice(adata) must route through _resolve_fit_input and
    produce the same output as diff_splice(fit_dict). Regression for
    the AnnData-audit bug where the isinstance check rejected any
    non-dict / non-MArrayLM input.
    """
    import anndata as ad

    from pylimma import diff_splice, lm_fit

    y = pd.read_csv(FIXTURES / "R_diffSplice_input_y.csv", index_col=0).values
    design = pd.read_csv(FIXTURES / "R_diffSplice_input_design.csv").values
    n_exons = y.shape[0]

    geneid = np.repeat([f"gene{i + 1}" for i in range(20)], 5)
    exonid = np.array([f"exon{i + 1}" for i in range(n_exons)])

    # Build a baseline fit_dict (matches the existing fixture path)
    fit_dict = lm_fit(y, design=design)
    fit_dict["genes"] = pd.DataFrame({"GeneID": geneid, "ExonID": exonid})
    out_ref = diff_splice(fit_dict, geneid="GeneID", exonid="ExonID", verbose=False)

    # AnnData path - limma orientation (n_exons x n_samples) becomes
    # (n_samples, n_exons) on the AnnData X.
    adata = ad.AnnData(X=y.T.copy())
    adata.var["GeneID"] = geneid
    adata.var["ExonID"] = exonid
    lm_fit(adata, design=design)
    out_anndata = diff_splice(adata, geneid="GeneID", exonid="ExonID", verbose=False)

    for slot in ("coefficients", "t", "p_value", "gene_F", "gene_F_p_value"):
        np.testing.assert_allclose(
            np.asarray(out_anndata[slot]),
            np.asarray(out_ref[slot]),
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{slot} differs (AnnData vs ndarray diff_splice)",
        )


# ----------------------------------------------------------------------------
# top_splice
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test,sort_by,file",
    [
        ("simes", "p", "R_topSplice_simes_p.csv"),
        ("simes", "none", "R_topSplice_simes_none.csv"),
        ("simes", "NExons", "R_topSplice_simes_NExons.csv"),
        ("F", "p", "R_topSplice_F_p.csv"),
        ("F", "none", "R_topSplice_F_none.csv"),
        ("F", "NExons", "R_topSplice_F_NExons.csv"),
        ("t", "p", "R_topSplice_t_p.csv"),
        ("t", "none", "R_topSplice_t_none.csv"),
        ("t", "logFC", "R_topSplice_t_logFC.csv"),
    ],
)
def test_top_splice_rparity(_diffsplice_fit, test, sort_by, file):
    from pylimma import top_splice

    result = top_splice(_diffsplice_fit, coef=1, test=test, number=np.inf, sort_by=sort_by)
    expected = pd.read_csv(FIXTURES / file)
    # For "none" order, we expect identical row order. For sorted orders,
    # the ranking should match.
    assert len(result) == len(expected)
    # Numeric columns
    for col in ("P.Value", "FDR"):
        if col in expected.columns:
            np.testing.assert_allclose(
                result[col].values,
                expected[col].values,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"column {col} ({test}, {sort_by})",
            )


# ----------------------------------------------------------------------------
# plot_splice
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_splice_substrate(_diffsplice_fit):
    from pylimma import plot_splice

    expected = pd.read_csv(FIXTURES / "R_plotSplice_substrate.csv")
    # Identify top gene by minimum F p-value on last coef
    gene_F_p = np.asarray(_diffsplice_fit["gene_F_p_value"])
    i = int(np.argmin(gene_F_p[:, 1]))
    first = int(_diffsplice_fit["gene_firstexon"][i])
    last = int(_diffsplice_fit["gene_lastexon"][i])
    exons = slice(first, last + 1)

    np.testing.assert_allclose(
        np.asarray(_diffsplice_fit["coefficients"])[exons, 1],
        expected["log_fc"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_diffsplice_fit["t"])[exons, 1],
        expected["t"].values,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_diffsplice_fit["p_value"])[exons, 1],
        expected["p"].values,
        rtol=1e-6,
        atol=1e-9,
    )

    ax = plot_splice(_diffsplice_fit, coef=1)
    assert ax is not None
