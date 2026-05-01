"""Tests for pylimma lmfit module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylimma.dups import unwrap_dups
from pylimma.lmfit import gls_series, is_fullrank, lm_fit, lm_series, mrlm, non_estimable

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_fullrank_true(self):
        """Test is_fullrank returns True for full rank matrix."""
        X = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
        assert is_fullrank(X)

    def test_is_fullrank_false(self):
        """Test is_fullrank returns False for rank-deficient matrix."""
        X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        assert not is_fullrank(X)

    def test_non_estimable_none(self):
        """Test non_estimable returns None for full rank."""
        X = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
        assert non_estimable(X) is None

    def test_non_estimable_identifies_redundant(self):
        """Test non_estimable identifies redundant columns."""
        # Third column is sum of first two
        X = np.array([[1, 0, 1], [1, 1, 2], [1, 0, 1], [1, 1, 2]])
        ne = non_estimable(X)
        assert ne is not None
        assert len(ne) == 1


class TestLmSeries:
    """Tests for lm_series function."""

    def test_r_parity(self):
        """Test lm_series matches R limma's lm.series."""
        # Load R fixtures
        expr = pd.read_csv(FIXTURES_DIR / "lmfit_expr.csv", index_col=0).values
        design = pd.read_csv(FIXTURES_DIR / "lmfit_design.csv").values

        ref_coef = pd.read_csv(FIXTURES_DIR / "lmfit_coefficients.csv", index_col=0).values
        ref_stdev = pd.read_csv(FIXTURES_DIR / "lmfit_stdev_unscaled.csv", index_col=0).values
        ref_stats = pd.read_csv(FIXTURES_DIR / "lmfit_stats.csv", index_col=0)

        # Fit in Python
        result = lm_series(expr, design)

        # Compare
        np.testing.assert_allclose(result["coefficients"], ref_coef, rtol=1e-10)
        np.testing.assert_allclose(result["stdev_unscaled"], ref_stdev, rtol=1e-10)
        np.testing.assert_allclose(result["sigma"], ref_stats["sigma"].values, rtol=1e-10)
        np.testing.assert_allclose(
            result["df_residual"], ref_stats["df_residual"].values, rtol=1e-10
        )

    def test_intercept_only(self):
        """Test fitting intercept-only model."""
        np.random.seed(123)
        expr = np.random.randn(10, 5)
        design = np.ones((5, 1))

        result = lm_series(expr, design)

        # Coefficients should be row means
        np.testing.assert_allclose(result["coefficients"][:, 0], np.mean(expr, axis=1), rtol=1e-10)
        assert result["rank"] == 1
        assert np.all(result["df_residual"] == 4)


class TestLmFit:
    """Tests for lm_fit function."""

    def test_dataframe_input(self):
        """Test lm_fit with DataFrame input."""
        expr_df = pd.read_csv(FIXTURES_DIR / "lmfit_expr.csv", index_col=0)
        design = pd.read_csv(FIXTURES_DIR / "lmfit_design.csv").values

        result = lm_fit(expr_df, design)

        assert isinstance(result, dict)
        assert result["genes"] == list(expr_df.index)

    def test_anndata_input(self):
        """Test lm_fit with AnnData input."""
        pytest.importorskip("anndata")
        import anndata as ad

        expr = pd.read_csv(FIXTURES_DIR / "lmfit_expr.csv", index_col=0)
        design = pd.read_csv(FIXTURES_DIR / "lmfit_design.csv").values

        # AnnData is samples x genes, so transpose
        adata = ad.AnnData(X=expr.T.values)
        adata.var_names = expr.index
        adata.obs_names = expr.columns

        result = lm_fit(adata, design)

        # Should return None and store in adata.uns
        assert result is None
        assert "pylimma" in adata.uns
        assert adata.uns["pylimma"]["coefficients"].shape == (50, 2)

    def test_default_design(self):
        """Test lm_fit with default intercept-only design."""
        np.random.seed(42)
        expr = np.random.randn(10, 5)

        result = lm_fit(expr)

        assert result["design"].shape == (5, 1)
        np.testing.assert_array_equal(result["design"], np.ones((5, 1)))

    def test_missing_values(self):
        """Test lm_fit handles missing values."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        expr[0, 0] = np.nan
        expr[5, 3] = np.nan
        design = np.ones((6, 1))

        result = lm_fit(expr, design)

        # Should complete without error
        assert not np.all(np.isnan(result["coefficients"]))
        # Gene with missing value should have df_residual = 4 (not 5)
        assert result["df_residual"][0] == 4

    def test_amean_calculation(self):
        """Test that Amean is correctly calculated."""
        expr = pd.read_csv(FIXTURES_DIR / "lmfit_expr.csv", index_col=0).values
        design = pd.read_csv(FIXTURES_DIR / "lmfit_design.csv").values
        ref_stats = pd.read_csv(FIXTURES_DIR / "lmfit_stats.csv", index_col=0)

        result = lm_fit(expr, design)

        np.testing.assert_allclose(result["Amean"], ref_stats["Amean"].values, rtol=1e-10)


class TestLmFitDispatch:
    """Tests for lm_fit dispatch logic and error handling."""

    def test_requires_correlation_with_block(self):
        """Test that correlation is required when block is provided."""
        np.random.seed(42)
        expr = np.random.randn(10, 6)
        design = np.ones((6, 1))
        block = np.array([0, 0, 1, 1, 2, 2])

        with pytest.raises(ValueError, match="correlation must be provided"):
            lm_fit(expr, design, block=block)

    def test_requires_correlation_with_ndups(self):
        """Test that correlation is required when ndups >= 2."""
        np.random.seed(42)
        expr = np.random.randn(8, 3)
        design = np.ones((3, 1))

        with pytest.raises(ValueError, match="correlation must be provided"):
            lm_fit(expr, design, ndups=2)


class TestUnwrapDups:
    """Tests for unwrap_dups function."""

    def test_unwrap_preserves_values(self):
        """Test that values are correctly mapped."""
        M = np.array(
            [
                [1, 2],  # gene 0, dup 0
                [3, 4],  # gene 0, dup 1
                [5, 6],  # gene 1, dup 0
                [7, 8],  # gene 1, dup 1
            ]
        )
        result = unwrap_dups(M, ndups=2, spacing=1)

        # Result should be (2 genes, 4 columns)
        assert result.shape == (2, 4)
        # Gene 0: [dup0-arr0, dup1-arr0, dup0-arr1, dup1-arr1] = [1, 3, 2, 4]
        np.testing.assert_array_equal(result[0, :], [1, 3, 2, 4])


class TestGlsSeries:
    """Tests for gls_series function."""

    def test_correlation_bounds(self):
        """Test that correlation must be within (-1, 1)."""
        np.random.seed(42)
        M = np.random.randn(10, 6)
        design = np.ones((6, 1))
        block = np.array([0, 0, 1, 1, 2, 2])

        with pytest.raises(ValueError, match="degenerate"):
            gls_series(M, design, block=block, correlation=1.0)

    def test_block_correlation_basic(self):
        """Test GLS with block correlation structure."""
        np.random.seed(42)
        n_genes, n_samples = 20, 12
        M = np.random.randn(n_genes, n_samples)
        design = np.column_stack([np.ones(n_samples), np.tile([0, 1], 6)])
        block = np.repeat([0, 1, 2, 3, 4, 5], 2)  # 6 blocks of 2

        result = gls_series(M, design, block=block, correlation=0.5)

        assert "coefficients" in result
        assert "sigma" in result
        assert result["coefficients"].shape == (n_genes, 2)
        assert result["correlation"] == 0.5

    def test_block_correlation_reduces_to_ols(self):
        """Test that zero correlation reduces to OLS."""
        np.random.seed(42)
        n_genes, n_samples = 20, 12
        M = np.random.randn(n_genes, n_samples)
        design = np.column_stack([np.ones(n_samples), np.random.randn(n_samples)])
        block = np.repeat([0, 1, 2, 3, 4, 5], 2)

        gls_result = gls_series(M, design, block=block, correlation=0.0)
        ols_result = lm_series(M, design)

        # Should be very close
        np.testing.assert_allclose(
            gls_result["coefficients"], ols_result["coefficients"], rtol=1e-8
        )

    def test_block_length_mismatch(self):
        """Test that block length must match samples."""
        np.random.seed(42)
        M = np.random.randn(10, 6)
        design = np.ones((6, 1))
        block = np.array([0, 0, 1, 1])  # wrong length

        with pytest.raises(ValueError, match="block does not match"):
            gls_series(M, design, block=block, correlation=0.5)

    def test_with_missing_values(self):
        """Test GLS handles missing values."""
        np.random.seed(42)
        n_genes, n_samples = 20, 12
        M = np.random.randn(n_genes, n_samples)
        M[0, 0] = np.nan
        M[5, 3] = np.nan
        design = np.ones((n_samples, 1))
        block = np.repeat([0, 1, 2, 3, 4, 5], 2)

        result = gls_series(M, design, block=block, correlation=0.5)

        # Should complete without error
        assert not np.all(np.isnan(result["coefficients"]))

    def test_duplicate_spots_warning(self):
        """Test warning when ndups < 2."""
        np.random.seed(42)
        M = np.random.randn(10, 6)
        design = np.ones((6, 1))

        with pytest.warns(UserWarning, match="No duplicates"):
            result = gls_series(M, design, ndups=1, correlation=0.5)

        # Should fall back to OLS-like behavior
        assert result["ndups"] == 1

    def test_returns_correlation_info(self):
        """Test that result contains correlation structure info."""
        np.random.seed(42)
        M = np.random.randn(10, 6)
        design = np.ones((6, 1))
        block = np.array([0, 0, 1, 1, 2, 2])

        result = gls_series(M, design, block=block, correlation=0.3)

        assert "correlation" in result
        assert "block" in result
        assert "ndups" in result
        assert result["correlation"] == 0.3
        np.testing.assert_array_equal(result["block"], block)


class TestMrlm:
    """Tests for mrlm (robust linear model) function."""

    def test_downweights_outliers(self):
        """Test that mrlm downweights outliers compared to OLS."""
        np.random.seed(42)
        n_genes, n_samples = 10, 20
        M = np.random.randn(n_genes, n_samples)
        # Add extreme outlier
        M[0, 0] = 100

        design = np.ones((n_samples, 1))

        ols_result = lm_series(M, design)
        robust_result = mrlm(M, design)

        # For gene 0, robust estimate should be closer to true mean
        # (which is approximately 0 for standard normal)
        # than OLS which is pulled by the outlier
        true_mean = 0
        ols_coef = ols_result["coefficients"][0, 0]
        robust_coef = robust_result["coefficients"][0, 0]

        # Robust should be closer to 0 than OLS
        assert abs(robust_coef - true_mean) < abs(ols_coef - true_mean)

    def test_huber_method(self):
        """Test Huber M-estimation."""
        np.random.seed(42)
        M = np.random.randn(10, 8)
        design = np.ones((8, 1))

        result = mrlm(M, design, method="huber")
        assert not np.all(np.isnan(result["coefficients"]))

    def test_bisquare_method(self):
        """Test Tukey bisquare M-estimation."""
        np.random.seed(42)
        M = np.random.randn(10, 8)
        design = np.ones((8, 1))

        result = mrlm(M, design, method="bisquare")
        assert not np.all(np.isnan(result["coefficients"]))

    def test_unknown_method_raises(self):
        """Test that unknown method raises error."""
        np.random.seed(42)
        M = np.random.randn(10, 8)
        design = np.ones((8, 1))

        with pytest.raises(ValueError, match="Unknown method"):
            mrlm(M, design, method="unknown")

    def test_weights_affect_results(self):
        """Test that weights actually change coefficient estimates."""
        np.random.seed(42)
        # Create data where one sample is an outlier
        M = np.array([[1.0, 1.0, 1.0, 1.0, 10.0]])  # Last sample is outlier
        design = np.ones((5, 1))

        # Without weights: robust fit should downweight outlier
        result_no_wt = mrlm(M, design)

        # With high weight on outlier: should pull estimate towards outlier
        weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
        result_high_wt = mrlm(M, design, weights=weights)

        # With low weight on outlier: should be close to no-weight result
        weights_low = np.array([1.0, 1.0, 1.0, 1.0, 0.1])
        result_low_wt = mrlm(M, design, weights=weights_low)

        # High weight on outlier should increase coefficient estimate
        assert result_high_wt["coefficients"][0, 0] > result_no_wt["coefficients"][0, 0]
        # Low weight on outlier should keep estimate closer to bulk
        assert result_low_wt["coefficients"][0, 0] < result_high_wt["coefficients"][0, 0]

    def test_handles_missing_values(self):
        """Test that mrlm handles missing values."""
        np.random.seed(42)
        M = np.random.randn(10, 8)
        M[0, 0] = np.nan
        M[5, 3] = np.nan
        design = np.ones((8, 1))

        result = mrlm(M, design)
        # Should complete without error
        assert not np.all(np.isnan(result["coefficients"]))


class TestAnnDataVoomLmFitWeightsBridge:
    """Regression tests for the voom -> lm_fit AnnData weights bridge.

    ``pylimma.voom(adata)`` writes ``voom_weights`` in AnnData
    orientation (samples x genes); ``pylimma.lm_fit(adata,
    layer='voom_E')`` must auto-load and transpose that layer. Prior
    to the 2026-04-20 fix, the AnnData idiom silently produced an
    unweighted fit - see
    ``memory/known_diff_voom_weights_layer_not_autoloaded.md``.
    """

    def _make_pseudobulk(self):
        import anndata as ad

        rng = np.random.default_rng(0)
        X = rng.negative_binomial(5, 0.5, size=(16, 200)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "donor": [f"d{i // 2}" for i in range(16)],
                "condition": ["ctrl", "stim"] * 8,
            }
        )
        obs.index = [f"s{i}" for i in range(16)]
        var = pd.DataFrame(index=[f"g{i}" for i in range(200)])
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_anndata_native_matches_explicit_weights(self):
        """lm_fit(adata, layer='voom_E') must auto-load voom_weights
        and produce the same fit as lm_fit with explicit weights."""
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )

        # Path A: fully AnnData-native (weights auto-loaded)
        pb_a = pb.copy()
        pylimma.voom(pb_a, design=design)
        pylimma.lm_fit(pb_a, design=design, layer="voom_E")

        # Path B: explicit transposed weights
        pb_b = pb.copy()
        pylimma.voom(pb_b, design=design)
        pylimma.lm_fit(pb_b, design=design, layer="voom_E", weights=pb_b.layers["voom_weights"].T)

        fit_a = pb_a.uns["pylimma"]
        fit_b = pb_b.uns["pylimma"]
        for slot in ("coefficients", "stdev_unscaled", "sigma", "df_residual"):
            np.testing.assert_allclose(
                np.asarray(fit_a[slot]),
                np.asarray(fit_b[slot]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{slot} differs between auto-loaded and explicit weights",
            )

    def test_explicit_weights_overrides_auto_load(self):
        """An explicit weights= kwarg wins over auto-loading."""
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.voom(pb, design=design)
        custom_w = np.ones((pb.shape[1], pb.shape[0]))  # unit weights
        pylimma.lm_fit(pb, design=design, layer="voom_E", weights=custom_w)
        fit = pb.uns["pylimma"]
        # Unit weights produce a different (unweighted-equivalent)
        # fit than voom's per-observation weights.
        pb_v = pb.copy()
        # Re-run with auto-loaded voom weights:
        pb_v.uns.pop("pylimma", None)
        pylimma.lm_fit(pb_v, design=design, layer="voom_E")
        fit_v = pb_v.uns["pylimma"]
        assert not np.allclose(np.asarray(fit["sigma"]), np.asarray(fit_v["sigma"]))

    def test_vooma_weights_auto_loaded(self):
        """vooma writes ``vooma_weights`` not ``voom_weights``; the
        generalised ``{layer[:-2]}_weights`` auto-load convention must
        pick it up. Regression for the same silent-drop class as
        voom_weights."""
        import pylimma

        # vooma expects log-normalised expression, not raw counts.
        rng = np.random.default_rng(1)
        X = rng.standard_normal((16, 200)).astype(np.float32) + 6.0
        obs = pd.DataFrame(
            {
                "donor": [f"d{i // 2}" for i in range(16)],
                "condition": ["ctrl", "stim"] * 8,
            }
        )
        obs.index = [f"s{i}" for i in range(16)]
        var = pd.DataFrame(index=[f"g{i}" for i in range(200)])
        import anndata as ad

        pb = ad.AnnData(X=X, obs=obs, var=var)
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.vooma(pb, design=design)
        assert "vooma_E" in pb.layers
        assert "vooma_weights" in pb.layers

        # Path A: AnnData-native (weights auto-loaded)
        pb_a = pb.copy()
        pylimma.lm_fit(pb_a, design=design, layer="vooma_E")
        # Path B: explicit transposed weights
        pb_b = pb.copy()
        pylimma.lm_fit(pb_b, design=design, layer="vooma_E", weights=pb_b.layers["vooma_weights"].T)

        for slot in ("coefficients", "stdev_unscaled", "sigma"):
            np.testing.assert_allclose(
                np.asarray(pb_a.uns["pylimma"][slot]),
                np.asarray(pb_b.uns["pylimma"][slot]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{slot} differs (vooma auto-load)",
            )

    def test_non_voom_layer_does_not_auto_load(self):
        """A caller passing ``layer='normalized'`` or similar should
        NOT silently pick up a random ``voom_weights`` layer that may
        have been left over from an earlier step. The gate is strict:
        ``{layer[:-2]}_weights`` only."""

        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.voom(pb, design=design)
        # Manually store a renamed copy of the E layer with an
        # unrelated suffix. There is no 'normalized_weights' layer.
        pb.layers["normalized"] = pb.layers["voom_E"].copy()

        pb_a = pb.copy()
        pylimma.lm_fit(pb_a, design=design, layer="normalized")
        # Should be unweighted (no 'normalized_weights' to pick up).
        # Compare against an explicitly-unweighted call on the same
        # layer to confirm they match.
        pb_b = pb.copy()
        pylimma.lm_fit(pb_b, design=design, layer="normalized", weights=None)
        np.testing.assert_allclose(
            np.asarray(pb_a.uns["pylimma"]["sigma"]),
            np.asarray(pb_b.uns["pylimma"]["sigma"]),
            rtol=1e-14,
        )

    def test_array_weights_auto_loads_voom_weights(self):
        """array_weights(adata, layer='voom_E') must auto-load the
        companion voom_weights layer. Previously silently produced the
        unweighted-equivalent result because get_eawp returned
        weights=None for AnnData regardless of companion layers.
        """
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.voom(pb, design=design)

        # Path A: AnnData-native
        aw_a = pylimma.array_weights(pb, design=design, layer="voom_E")
        # Path B: explicit transposed weights
        aw_b = pylimma.array_weights(
            pb,
            design=design,
            layer="voom_E",
            weights=pb.layers["voom_weights"].T,
        )
        np.testing.assert_allclose(aw_a, aw_b, rtol=1e-12, atol=1e-14)

        # And must differ from the unweighted-equivalent (no weights):
        pb_no_w = pb.copy()
        del pb_no_w.layers["voom_weights"]
        aw_unweighted = pylimma.array_weights(
            pb_no_w,
            design=design,
            layer="voom_E",
        )
        assert not np.allclose(aw_a, aw_unweighted)

    def test_sparse_anndata_matches_dense(self):
        """get_eawp must densify scipy.sparse adata.X / layers. Scanpy
        stores counts as sparse by default; without densification,
        np.asarray(sparse) produces a 0-d object array and the whole
        pipeline either crashes or returns nonsense.
        """
        import anndata as ad
        import scipy.sparse as sp

        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )

        X_dense = np.asarray(pb.X)
        pb_csr = ad.AnnData(
            X=sp.csr_matrix(X_dense),
            obs=pb.obs.copy(),
            var=pb.var.copy(),
        )
        pb_dense = ad.AnnData(
            X=X_dense,
            obs=pb.obs.copy(),
            var=pb.var.copy(),
        )

        for a in (pb_csr, pb_dense):
            pylimma.voom(a, design=design)
            pylimma.lm_fit(a, design=design, layer="voom_E")

        for slot in ("coefficients", "stdev_unscaled", "sigma", "df_residual"):
            np.testing.assert_allclose(
                np.asarray(pb_csr.uns["pylimma"][slot]),
                np.asarray(pb_dense.uns["pylimma"][slot]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{slot} differs sparse vs dense",
            )

    def test_sparse_layer_densified(self):
        """A sparse adata.layers[...] entry (e.g. from scanpy.pp.normalize_total)
        must also be densified by get_eawp when passed as ``layer=``.
        """
        import anndata as ad
        import scipy.sparse as sp

        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        X_dense = np.asarray(pb.X)

        a = ad.AnnData(
            X=np.zeros_like(X_dense),
            obs=pb.obs.copy(),
            var=pb.var.copy(),
        )
        a.layers["counts"] = sp.csr_matrix(X_dense)
        pylimma.voom(a, design=design, layer="counts")
        pylimma.lm_fit(a, design=design, layer="voom_E")
        # Just confirm it ran to completion and wrote a sensible fit.
        assert np.all(np.isfinite(a.uns["pylimma"]["sigma"]))

    def test_duplicate_correlation_auto_loads_voom_weights(self):
        """duplicate_correlation(adata, layer='voom_E', block=...) must
        auto-load voom_weights. A wrong consensus correlation would
        silently propagate into any downstream lm_fit(block=..., correlation=...).
        """
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        block = pb.obs["donor"].values
        pylimma.voom(pb, design=design)

        # Path A: AnnData-native
        dc_a = pylimma.duplicate_correlation(
            pb,
            design=design,
            block=block,
            layer="voom_E",
        )
        # Path B: explicit transposed weights
        dc_b = pylimma.duplicate_correlation(
            pb,
            design=design,
            block=block,
            layer="voom_E",
            weights=pb.layers["voom_weights"].T,
        )
        assert np.isclose(
            dc_a["consensus_correlation"], dc_b["consensus_correlation"], rtol=1e-10, atol=1e-12
        )

    def test_anndata_design_auto_loaded_from_voom_uns(self):
        """lm_fit(adata, layer='voom_E') without a design= kwarg must
        pick up the design that voom stashed at
        ``adata.uns['voom']['design']``, mirroring R's lmFit one-liner
        ``if(is.null(design)) design <- y$design``. Pre-fix, lm_fit
        silently defaulted to an intercept-only model, producing
        silently-wrong coefficients.
        """
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )

        pb_a = pb.copy()
        pylimma.voom(pb_a, design=design)
        pylimma.lm_fit(pb_a, layer="voom_E")  # no design=

        pb_b = pb.copy()
        pylimma.voom(pb_b, design=design)
        pylimma.lm_fit(pb_b, design=design, layer="voom_E")

        fit_a = pb_a.uns["pylimma"]
        fit_b = pb_b.uns["pylimma"]
        assert fit_a["design"].shape == design.shape
        for slot in ("coefficients", "stdev_unscaled", "sigma", "df_residual"):
            np.testing.assert_allclose(
                np.asarray(fit_a[slot]),
                np.asarray(fit_b[slot]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{slot} differs (design auto-load from uns)",
            )

    def test_anndata_design_auto_loaded_from_vooma_uns(self):
        """Same contract for vooma: adata.uns['vooma']['design'] must be
        picked up when lm_fit is called with layer='vooma_E'.
        """
        import anndata as ad

        import pylimma

        rng = np.random.default_rng(2)
        X = rng.standard_normal((16, 200)).astype(np.float32) + 6.0
        obs = pd.DataFrame(
            {
                "donor": [f"d{i // 2}" for i in range(16)],
                "condition": ["ctrl", "stim"] * 8,
            }
        )
        obs.index = [f"s{i}" for i in range(16)]
        var = pd.DataFrame(index=[f"g{i}" for i in range(200)])
        pb = ad.AnnData(X=X, obs=obs, var=var)
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )

        pb_a = pb.copy()
        pylimma.vooma(pb_a, design=design)
        pylimma.lm_fit(pb_a, layer="vooma_E")  # no design=
        assert pb_a.uns["pylimma"]["design"].shape == design.shape

    def test_explicit_design_overrides_uns_design(self):
        """An explicit design= kwarg wins over the uns fallback."""
        import pylimma

        pb = self._make_pseudobulk()
        design_voom = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        # Different design used at lm_fit time (intercept only).
        design_fit = np.ones((pb.shape[0], 1))

        pylimma.voom(pb, design=design_voom)
        pylimma.lm_fit(pb, design=design_fit, layer="voom_E")
        assert pb.uns["pylimma"]["design"].shape == design_fit.shape

    def test_elist_design_auto_loaded(self):
        """lm_fit on an EList with a populated design slot must pick it
        up without re-passing, matching R's lmFit(y) semantics.
        """
        import pylimma

        rng = np.random.default_rng(3)
        n_samples, n_genes = 8, 60
        E = rng.standard_normal((n_genes, n_samples)) + 6.0
        design = np.column_stack([np.ones(n_samples), [0] * 4 + [1] * 4])
        el = pylimma.EList({"E": E, "design": design})

        fit_a = pylimma.lm_fit(el)
        fit_b = pylimma.lm_fit(el, design=design)
        for slot in ("coefficients", "stdev_unscaled", "sigma"):
            np.testing.assert_allclose(
                np.asarray(fit_a[slot]),
                np.asarray(fit_b[slot]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{slot} differs (EList design auto-load)",
            )

    def test_custom_weights_layer_read_side(self):
        """voom(adata, weights_layer='custom_w') paired with
        lm_fit(adata, weights_layer='custom_w') round-trips bit-exactly
        to the default-convention pair. Previously the read side was
        hard-coded to ``{stem}_weights`` so non-default weights_layer
        names silently ran unweighted.
        """
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )

        pb_custom = pb.copy()
        pylimma.voom(pb_custom, design=design, out_layer="custom_E", weights_layer="custom_w")
        pylimma.lm_fit(pb_custom, design=design, layer="custom_E", weights_layer="custom_w")

        pb_default = pb.copy()
        pylimma.voom(pb_default, design=design)
        pylimma.lm_fit(pb_default, design=design, layer="voom_E")

        for slot in ("coefficients", "stdev_unscaled", "sigma", "df_residual"):
            np.testing.assert_allclose(
                np.asarray(pb_custom.uns["pylimma"][slot]),
                np.asarray(pb_default.uns["pylimma"][slot]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{slot} differs (custom weights_layer)",
            )

    def test_weights_layer_raises_on_missing(self):
        """An explicit weights_layer= that doesn't exist in
        adata.layers must raise KeyError, not silently fall through to
        the stem-based convention or run unweighted.
        """
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.voom(pb, design=design)
        with pytest.raises(KeyError, match="weights_layer"):
            pylimma.lm_fit(pb, design=design, layer="voom_E", weights_layer="does_not_exist")

    def test_weights_layer_rejected_for_non_anndata(self):
        """weights_layer= is AnnData-only; EList/ndarray must reject it
        the same way layer= is rejected.
        """
        import pylimma

        el = pylimma.EList({"E": np.zeros((10, 6))})
        with pytest.raises(TypeError, match="weights_layer"):
            pylimma.lm_fit(el, weights_layer="foo")

    def test_voom_accepts_formula_string(self):
        """voom(adata, design='~ group') must parse the formula through
        patsy against adata.obs, matching lm_fit's dispatch. Pre-fix
        the string was sent straight into np.asarray and raised
        ``could not convert string to float``.
        """
        import pylimma

        pb = self._make_pseudobulk()
        # Formula path
        pb_a = pb.copy()
        pylimma.voom(pb_a, design="~ condition")
        design_ref = pylimma.model_matrix("~ condition", pb.obs)
        # Explicit-matrix path
        pb_b = pb.copy()
        pylimma.voom(pb_b, design=design_ref)

        np.testing.assert_allclose(
            np.asarray(pb_a.layers["voom_E"]),
            np.asarray(pb_b.layers["voom_E"]),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(pb_a.layers["voom_weights"]),
            np.asarray(pb_b.layers["voom_weights"]),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_vooma_accepts_formula_string(self):
        """vooma(adata, design='~ condition') must parse the formula."""
        import anndata as ad

        import pylimma

        rng = np.random.default_rng(5)
        X = rng.standard_normal((16, 200)).astype(np.float32) + 6.0
        obs = pd.DataFrame(
            {
                "condition": ["ctrl", "stim"] * 8,
            }
        )
        obs.index = [f"s{i}" for i in range(16)]
        var = pd.DataFrame(index=[f"g{i}" for i in range(200)])
        pb = ad.AnnData(X=X, obs=obs, var=var)

        pb_a = pb.copy()
        pylimma.vooma(pb_a, design="~ condition")
        design_ref = pylimma.model_matrix("~ condition", pb.obs)
        pb_b = pb.copy()
        pylimma.vooma(pb_b, design=design_ref)

        np.testing.assert_allclose(
            np.asarray(pb_a.layers["vooma_weights"]),
            np.asarray(pb_b.layers["vooma_weights"]),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_voom_with_quality_weights_accepts_formula_string(self):
        """voom_with_quality_weights(adata, design='~ condition') parses
        the formula at entry so both internal voom() calls see ndarray.
        """
        import pylimma

        pb = self._make_pseudobulk()
        pb_a = pb.copy()
        pylimma.voom_with_quality_weights(pb_a, design="~ condition")
        design_ref = pylimma.model_matrix("~ condition", pb.obs)
        pb_b = pb.copy()
        pylimma.voom_with_quality_weights(pb_b, design=design_ref)

        np.testing.assert_allclose(
            np.asarray(pb_a.layers["voom_weights"]),
            np.asarray(pb_b.layers["voom_weights"]),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_h5ad_roundtrip_preserves_fit(self):
        """Writing adata to h5ad and reading it back must preserve the
        full fit. Pre-fix, lm_fit stored an MArrayLM in adata.uns, and
        anndata's IO registry (which dispatches on exact type) raised
        ``IORegistryError`` on write. We now store a plain dict.
        """
        import tempfile

        import anndata as ad

        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.voom(pb, design=design)
        pylimma.lm_fit(pb, design=design, layer="voom_E")
        pylimma.e_bayes(pb)
        slots_before = sorted(pb.uns["pylimma"].keys())

        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
            path = tmp.name
        pb.write_h5ad(path)
        pb_reloaded = ad.read_h5ad(path)

        slots_after = sorted(pb_reloaded.uns["pylimma"].keys())
        assert slots_before == slots_after
        # Reloaded fit must still drive top_table.
        tt = pylimma.top_table(pb_reloaded, coef=1, number=5)
        assert len(tt) == 5

    def test_uns_fit_is_plain_dict(self):
        """adata.uns[key] after lm_fit / e_bayes / contrasts_fit / treat
        must be a plain ``dict`` (not MArrayLM) for h5ad compatibility.
        Users who want the MArrayLM API can re-wrap.
        """
        import pylimma

        pb = self._make_pseudobulk()
        design = np.column_stack(
            [
                np.ones(pb.shape[0]),
                (pb.obs["condition"] == "stim").astype(float).values,
            ]
        )
        pylimma.voom(pb, design=design)
        pylimma.lm_fit(pb, design=design, layer="voom_E")
        assert type(pb.uns["pylimma"]) is dict
        pylimma.e_bayes(pb)
        assert type(pb.uns["pylimma"]) is dict
        # Re-wrap still works.
        wrapped = pylimma.MArrayLM(pb.uns["pylimma"])
        assert hasattr(wrapped, "coefficients")

    def test_genas_accepts_anndata(self):
        """genas(adata) must route through _resolve_fit_input and
        operate on adata.uns[key]. Pre-fix it crashed with
        ``AttributeError: 'AnnData' object has no attribute 'get'``.
        """
        import anndata as ad

        import pylimma

        rng = np.random.default_rng(0)
        # Three-level factor so genas has two non-intercept coefs.
        X = rng.standard_normal((12, 40)).astype(np.float32) + 6.0
        obs = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4})
        obs.index = [f"s{i}" for i in range(12)]
        adata = ad.AnnData(X=X, obs=obs)
        design = np.column_stack(
            [
                np.ones(12),
                (obs["group"] == "B").astype(float).values,
                (obs["group"] == "C").astype(float).values,
            ]
        )
        pylimma.lm_fit(adata, design=design)
        pylimma.e_bayes(adata)

        out = pylimma.genas(adata, coef=(1, 2))
        assert isinstance(out, dict)
        assert "technical_correlation" in out

    def test_pred_fcm_accepts_anndata(self):
        """pred_fcm(adata) must route through _resolve_fit_input."""
        import anndata as ad

        import pylimma

        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 40)).astype(np.float32) + 6.0
        obs = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4})
        obs.index = [f"s{i}" for i in range(12)]
        adata = ad.AnnData(X=X, obs=obs)
        design = np.column_stack(
            [
                np.ones(12),
                (obs["group"] == "B").astype(float).values,
                (obs["group"] == "C").astype(float).values,
            ]
        )
        pylimma.lm_fit(adata, design=design)
        pylimma.e_bayes(adata)

        pfc = pylimma.pred_fcm(adata, coef=1)
        assert isinstance(pfc, np.ndarray)
        assert pfc.shape == (40,)

    def test_fit_targets_populated_from_adata_obs(self):
        """lm_fit must propagate adata.obs into fit['targets'],
        mirroring R's ``fit$targets <- y$targets``.
        """
        import anndata as ad

        import pylimma

        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 40)).astype(np.float32) + 10
        obs = pd.DataFrame(
            {
                "group": pd.Categorical(["A"] * 4 + ["B"] * 4),
                "donor": [1, 2, 3, 4, 1, 2, 3, 4],
            }
        )
        obs.index = [f"s{i}" for i in range(8)]
        adata = ad.AnnData(X=X, obs=obs)

        design = np.column_stack([np.ones(8), [0] * 4 + [1] * 4])
        pylimma.lm_fit(adata, design=design)

        fit = adata.uns["pylimma"]
        assert "targets" in fit
        targets = fit["targets"]
        # pandas DataFrame with the same rows as the obs we passed
        assert list(targets.columns) == ["group", "donor"]
        assert len(targets) == 8

    def test_fit_targets_populated_from_elist(self):
        """lm_fit on an EList with a targets slot must carry it through
        to fit['targets']."""
        import pylimma

        rng = np.random.default_rng(1)
        E = rng.standard_normal((30, 6)) + 6.0
        design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
        targets = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)]})
        el = pylimma.EList({"E": E, "design": design, "targets": targets})

        fit = pylimma.lm_fit(el)
        assert "targets" in fit
        assert list(fit["targets"]["sample_id"]) == [f"s{i}" for i in range(6)]

    def test_get_eawp_captures_var_index_without_columns(self):
        """get_eawp must populate y['probes'] as a DataFrame whose
        index carries var_names, even when adata.var has no annotation
        columns. Pre-fix the zero-columns gate dropped var_names on
        the common scanpy state where adata.var_names is set but
        adata.var is empty.
        """
        import anndata as ad

        import pylimma

        X = np.ones((6, 20))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(20)]
        # adata.var has zero annotation columns
        assert len(adata.var.columns) == 0

        eawp = pylimma.get_eawp(adata)
        assert eawp["probes"] is not None
        assert list(eawp["probes"].index) == list(adata.var_names)

    def test_put_eawp_broadcasts_1d_weights(self):
        """put_eawp must write 2-D weights layers even when the caller
        supplied 1-D array weights. A 1-D write would raise
        AnnData's ``Shape mismatch``.
        """
        import anndata as ad

        from pylimma.classes import put_eawp

        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 20))
        adata = ad.AnnData(X=X)
        E = rng.standard_normal((20, 6))  # limma orientation
        # length-6 array weights (one per sample)
        array_w = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5])

        put_eawp(
            {"E": E, "weights": array_w},
            adata,
            out_layer="foo_E",
            weights_layer="foo_weights",
            uns_key="foo",
        )
        assert adata.layers["foo_weights"].shape == (6, 20)
        # Array-weight broadcast: each sample's row has identical values.
        np.testing.assert_allclose(
            adata.layers["foo_weights"][0, :],
            array_w[0],
        )
