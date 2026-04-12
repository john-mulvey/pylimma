"""Tests for pylimma dups module."""

import numpy as np
import pandas as pd
import pytest

from pylimma.dups import unwrap_dups, unique_genelist, ave_dups, avereps


class TestUnwrapDups:
    """Tests for unwrap_dups function."""

    def test_no_dups_returns_original(self):
        """Test that ndups=1 returns original matrix."""
        M = np.array([[1, 2], [3, 4], [5, 6]])
        result = unwrap_dups(M, ndups=1)
        np.testing.assert_array_equal(result, M)

    def test_basic_unwrap_spacing1(self):
        """Test basic unwrapping with spacing=1."""
        # 4 spots (2 genes x 2 dups), 2 arrays
        M = np.array([
            [1, 2],   # gene 0, dup 0
            [3, 4],   # gene 0, dup 1
            [5, 6],   # gene 1, dup 0
            [7, 8],   # gene 1, dup 1
        ])
        result = unwrap_dups(M, ndups=2, spacing=1)

        # Should be 2 genes, 4 columns (ndups * arrays)
        assert result.shape == (2, 4)
        # Gene 0: [dup0-arr0, dup1-arr0, dup0-arr1, dup1-arr1] = [1, 3, 2, 4]
        np.testing.assert_array_equal(result[0, :], [1, 3, 2, 4])
        # Gene 1: [5, 7, 6, 8]
        np.testing.assert_array_equal(result[1, :], [5, 7, 6, 8])

    def test_output_shape(self):
        """Test that output shape is correct."""
        # 12 spots, 3 arrays
        M = np.random.randn(12, 3)
        result = unwrap_dups(M, ndups=2, spacing=1)

        # Should be 6 genes, 6 columns
        assert result.shape == (6, 6)

    def test_with_spacing(self):
        """Test unwrapping with spacing > 1."""
        # 8 spots with spacing=2 and ndups=2
        # n_groups = 8 / 2 / 2 = 2
        # output shape = (spacing * n_groups, ndups * n_arrays) = (4, 4)
        M = np.arange(16).reshape(8, 2)
        result = unwrap_dups(M, ndups=2, spacing=2)

        # Shape is (spacing * n_groups, ndups * n_arrays) = (4, 4)
        assert result.shape == (4, 4)


class TestUniqueGenelist:
    """Tests for unique_genelist function."""

    def test_no_dups_returns_original(self):
        """Test that ndups<=1 returns original."""
        genelist = ['A', 'B', 'C']
        result = unique_genelist(genelist, ndups=1)
        assert result == genelist

    def test_extracts_unique_list(self):
        """Test extraction of unique genes from list."""
        genelist = ['A', 'A', 'B', 'B', 'C', 'C']  # 3 genes, 2 dups each
        result = unique_genelist(genelist, ndups=2)
        assert len(result) == 3

    def test_extracts_unique_array(self):
        """Test extraction of unique genes from array."""
        genelist = np.array(['A', 'A', 'B', 'B'])
        result = unique_genelist(genelist, ndups=2)
        assert len(result) == 2

    def test_extracts_unique_dataframe(self):
        """Test extraction of unique genes from DataFrame."""
        genelist = pd.DataFrame({'gene': ['A', 'A', 'B', 'B'], 'desc': [1, 2, 3, 4]})
        result = unique_genelist(genelist, ndups=2)
        assert len(result) == 2
        assert isinstance(result, pd.DataFrame)


class TestAveDups:
    """Tests for ave_dups function."""

    def test_no_dups_returns_original(self):
        """Test that ndups=1 returns original."""
        M = np.array([[1, 2], [3, 4]])
        result = ave_dups(M, ndups=1)
        np.testing.assert_array_equal(result, M)

    def test_simple_average(self):
        """Test simple averaging of duplicates."""
        M = np.array([
            [1.0, 2.0],  # gene 0, dup 0
            [3.0, 4.0],  # gene 0, dup 1
            [5.0, 6.0],  # gene 1, dup 0
            [7.0, 8.0],  # gene 1, dup 1
        ])
        result = ave_dups(M, ndups=2, spacing=1)

        expected = np.array([
            [2.0, 3.0],  # mean of [1,3], [2,4]
            [6.0, 7.0],  # mean of [5,7], [6,8]
        ])
        np.testing.assert_allclose(result, expected)

    def test_weighted_average(self):
        """Test weighted averaging."""
        M = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        weights = np.array([
            [1.0, 1.0],  # equal weight
            [3.0, 3.0],  # 3x weight
        ])
        result = ave_dups(M, ndups=2, weights=weights)

        # Weighted mean: (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        expected = np.array([[2.5, 3.5]])
        np.testing.assert_allclose(result, expected)

    def test_handles_nan(self):
        """Test handling of NaN values."""
        M = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
        ])
        result = ave_dups(M, ndups=2)

        # Mean ignoring NaN: [2, 4]
        expected = np.array([[2.0, 4.0]])
        np.testing.assert_allclose(result, expected)


class TestAvereps:
    """Tests for avereps function."""

    def test_no_id_raises_error(self):
        """Test that no ID raises ValueError when input is a raw ndarray
        (R parity: avereps.default stops with 'No probe IDs' when both
        ID and rownames(x) are NULL)."""
        M = np.array([[1, 2], [3, 4], [5, 6]])
        with pytest.raises(ValueError, match="No probe IDs"):
            avereps(M)

    def test_dataframe_index_used_when_id_none(self):
        """DataFrame index acts as default ID, mirroring R's rownames(x)."""
        import pandas as pd
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                          index=["A", "A", "B"])
        result = avereps(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["A", "B"]
        np.testing.assert_allclose(result.values, [[2.0, 3.0], [5.0, 6.0]])

    def test_averages_replicates(self):
        """Test averaging of replicate probes."""
        M = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        ID = ['A', 'A', 'B']
        result = avereps(M, ID)

        expected = np.array([
            [2.0, 3.0],  # mean of A
            [5.0, 6.0],  # B
        ])
        np.testing.assert_allclose(result, expected)

    def test_preserves_order(self):
        """Test that unique IDs are in order of first appearance (via
        DataFrame round-trip so we can inspect the row labels)."""
        import pandas as pd
        df = pd.DataFrame(np.random.randn(4, 2), index=['B', 'A', 'B', 'A'])
        result = avereps(df)

        # B appears first, then A
        assert list(result.index) == ['B', 'A']

    def test_handles_nan(self):
        """Test handling of NaN values."""
        M = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
        ])
        ID = ['A', 'A']
        result = avereps(M, ID)

        # Mean ignoring NaN
        expected = np.array([[2.0, 4.0]])
        np.testing.assert_allclose(result, expected)

    def test_anndata_default_id_uses_var_names(self):
        """AnnData with no ID= must fall back to adata.var_names and
        return a new AnnData with the var axis collapsed to unique
        probes in first-appearance order."""
        import anndata as ad

        X = np.arange(24, dtype=float).reshape(4, 6)     # 4 samples, 6 probes
        adata = ad.AnnData(X=X.copy())
        adata.var_names = ["A", "A", "B", "B", "C", "C"]
        adata.obs_names = [f"s{i}" for i in range(4)]

        out = avereps(adata)
        assert isinstance(out, ad.AnnData)
        assert out.shape == (4, 3)
        assert list(out.var_names) == ["A", "B", "C"]
        # obs axis unchanged
        assert list(out.obs_names) == list(adata.obs_names)
        # Values match the ndarray path
        ref = avereps(X.T, ID=adata.var_names.values)    # (3, 4)
        np.testing.assert_allclose(out.X, ref.T, rtol=1e-14)

    def test_anndata_explicit_id(self):
        """Explicit ID= overrides var_names."""
        import anndata as ad

        X = np.arange(16, dtype=float).reshape(2, 8)
        adata = ad.AnnData(X=X.copy())
        adata.var_names = [f"ENSG{i}" for i in range(8)]
        adata.var["symbol"] = ["TP53"] * 8
        custom_id = ["a", "a", "b", "b", "c", "c", "d", "d"]

        out = avereps(adata, ID=custom_id)
        assert list(out.var_names) == ["a", "b", "c", "d"]
        # var columns picked from first-occurrence row per group
        assert "symbol" in out.var.columns
