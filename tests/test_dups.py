"""Tests for pylimma dups module."""

import numpy as np
import pandas as pd
import pytest

from pylimma.dups import ave_dups, avereps, unique_genelist, unwrap_dups


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
        M = np.array(
            [
                [1, 2],  # gene 0, dup 0
                [3, 4],  # gene 0, dup 1
                [5, 6],  # gene 1, dup 0
                [7, 8],  # gene 1, dup 1
            ]
        )
        result = unwrap_dups(M, ndups=2, spacing=1)

        # Should be 2 genes, 4 columns (ndups * arrays)
        assert result.shape == (2, 4)
        # Gene 0: [dup0-arr0, dup1-arr0, dup0-arr1, dup1-arr1] = [1, 3, 2, 4]
        np.testing.assert_array_equal(result[0, :], [1, 3, 2, 4])
        # Gene 1: [5, 7, 6, 8]
        np.testing.assert_array_equal(result[1, :], [5, 7, 6, 8])

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
        genelist = ["A", "B", "C"]
        result = unique_genelist(genelist, ndups=1)
        assert result == genelist

    def test_extracts_unique_list(self):
        """Test extraction of unique genes from list."""
        genelist = ["A", "A", "B", "B", "C", "C"]  # 3 genes, 2 dups each
        result = unique_genelist(genelist, ndups=2)
        assert len(result) == 3

    def test_extracts_unique_array(self):
        """Test extraction of unique genes from array."""
        genelist = np.array(["A", "A", "B", "B"])
        result = unique_genelist(genelist, ndups=2)
        assert len(result) == 2

    def test_extracts_unique_dataframe(self):
        """Test extraction of unique genes from DataFrame."""
        genelist = pd.DataFrame({"gene": ["A", "A", "B", "B"], "desc": [1, 2, 3, 4]})
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
        M = np.array(
            [
                [1.0, 2.0],  # gene 0, dup 0
                [3.0, 4.0],  # gene 0, dup 1
                [5.0, 6.0],  # gene 1, dup 0
                [7.0, 8.0],  # gene 1, dup 1
            ]
        )
        result = ave_dups(M, ndups=2, spacing=1)

        expected = np.array(
            [
                [2.0, 3.0],  # mean of [1,3], [2,4]
                [6.0, 7.0],  # mean of [5,7], [6,8]
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_weighted_average(self):
        """Test weighted averaging."""
        M = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        weights = np.array(
            [
                [1.0, 1.0],  # equal weight
                [3.0, 3.0],  # 3x weight
            ]
        )
        result = ave_dups(M, ndups=2, weights=weights)

        # Weighted mean: (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        expected = np.array([[2.5, 3.5]])
        np.testing.assert_allclose(result, expected)

    def test_handles_nan(self):
        """Test handling of NaN values."""
        M = np.array(
            [
                [1.0, np.nan],
                [3.0, 4.0],
            ]
        )
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

        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=["A", "A", "B"])
        result = avereps(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["A", "B"]
        np.testing.assert_allclose(result.values, [[2.0, 3.0], [5.0, 6.0]])

    def test_averages_replicates(self):
        """Test averaging of replicate probes."""
        M = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        ID = ["A", "A", "B"]
        result = avereps(M, ID)

        expected = np.array(
            [
                [2.0, 3.0],  # mean of A
                [5.0, 6.0],  # B
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_ids_as_r_forms_groups(self):
        """R (fixture script): IDs compared as as.character, NA IDs one
        group with rowsum's warning, an ID of the wrong length stops."""
        x = np.arange(1.0, 7.0).reshape(2, 3).T
        with pytest.warns(UserWarning, match="missing values for 'group'"):
            np.testing.assert_array_equal(avereps(x, ID=["a", None, None]), [[1, 4], [2.5, 5.5]])
        # NaN and pandas NA are missing too, in the same group
        with pytest.warns(UserWarning, match="missing values for 'group'"):
            np.testing.assert_array_equal(avereps(x, ID=["a", np.nan, pd.NA]), [[1, 4], [2.5, 5.5]])
        np.testing.assert_array_equal(avereps(x, ID=[1, 1.0, 2]), [[1.5, 4.5], [3, 6]])
        # a numeric NaN is pandas' missing value: one group, with the warning
        # (R's NaN ID "NaN" gives the same averages without it)
        with pytest.warns(UserWarning, match="missing values for 'group'"):
            by_nan = avereps(pd.DataFrame(x), ID=[1.0, np.nan, np.nan])
        np.testing.assert_array_equal(by_nan.to_numpy(), [[1, 4], [2.5, 5.5]])
        # a one-column matrix is a vector (R's as.character)
        np.testing.assert_array_equal(
            avereps(x, ID=np.array([["a"], ["b"], ["a"]])), avereps(x, ID=["a", "b", "a"])
        )
        with pytest.raises(ValueError, match="incorrect length for 'group'"):
            avereps(x, ID=["a", "b"])
        # a single ID is a vector of one
        np.testing.assert_array_equal(avereps(x[:1], ID="a"), x[:1])
        with pytest.raises(ValueError, match="ID must be a vector"):
            avereps(x, ID=np.array([["a", "b"], ["a", "c"], ["b", "d"]]))

    def test_one_column_id_matrix_for_every_input_class(self):
        anndata = pytest.importorskip("anndata")
        from pylimma import EList

        x = np.arange(1.0, 7.0).reshape(2, 3).T
        vector, column = ["a", "b", "a"], np.array([["a"], ["b"], ["a"]])
        np.testing.assert_array_equal(
            avereps(EList({"E": x}), ID=column)["E"], avereps(EList({"E": x}), ID=vector)["E"]
        )
        adata = anndata.AnnData(X=x.T)
        np.testing.assert_array_equal(
            avereps(adata, ID=column.T).X, avereps(adata, ID=vector).X
        )

    def test_preserves_order(self):
        """Test that unique IDs are in order of first appearance (via
        DataFrame round-trip so we can inspect the row labels)."""
        import pandas as pd

        df = pd.DataFrame(np.random.randn(4, 2), index=["B", "A", "B", "A"])
        result = avereps(df)

        # B appears first, then A
        assert list(result.index) == ["B", "A"]

    def test_handles_nan(self):
        """Test handling of NaN values."""
        M = np.array(
            [
                [1.0, np.nan],
                [3.0, 4.0],
            ]
        )
        ID = ["A", "A"]
        result = avereps(M, ID)

        # Mean ignoring NaN
        expected = np.array([[2.0, 4.0]])
        np.testing.assert_allclose(result, expected)

    def test_anndata_default_id_uses_var_names(self):
        """AnnData with no ID= must fall back to adata.var_names and
        return a new AnnData with the var axis collapsed to unique
        probes in first-appearance order."""
        import anndata as ad

        X = np.arange(24, dtype=float).reshape(4, 6)  # 4 samples, 6 probes
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
        ref = avereps(X.T, ID=adata.var_names.values)  # (3, 4)
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


class TestDuplicateCorrelationInputs:
    """Input forms R accepts (fixture script)."""

    FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"

    def _data(self):
        y = pd.read_csv(self.FIXTURES / "R_dci_y.csv").to_numpy()
        expected = pd.read_csv(self.FIXTURES / "R_dci.csv").set_index("case")["cor"]
        return y, expected

    def test_character_block_as_r_factor(self):
        from pylimma import duplicate_correlation

        y, expected = self._data()
        block = list(np.repeat(["b", "a", "c"], 4))
        design = np.column_stack([np.ones(12), np.tile([0.0, 1.0], 6)])
        out = duplicate_correlation(y, design, block=block)
        np.testing.assert_allclose(
            out["consensus_correlation"], expected["character_block"], rtol=1e-6
        )

    def test_vector_design_is_one_column(self):
        from pylimma import duplicate_correlation

        y, expected = self._data()
        out = duplicate_correlation(y, np.ones(12), block=np.repeat(["b", "a", "c"], 4))
        np.testing.assert_allclose(
            out["consensus_correlation"], expected["vector_design"], rtol=1e-6
        )
        with pytest.raises(ValueError, match="Number of rows of design matrix"):
            duplicate_correlation(y, np.ones((11, 1)), block=np.repeat(["b", "a", "c"], 4))

    def test_block_checks_follow_r(self):
        """R: a saturated design passes the block-in-design check (the
        residual part is empty), giving 0; a single block stops in
        model.matrix (pylimma: a clear error)."""
        from pylimma import duplicate_correlation

        y, _ = self._data()
        block = np.repeat(["b", "a", "c"], 4)
        with pytest.warns(UserWarning, match="already encoded"):
            assert duplicate_correlation(y, np.eye(12), block=block)["consensus_correlation"] == 0
        with pytest.raises(ValueError, match="block must have at least two levels"):
            duplicate_correlation(y, block=np.ones(12))

    def test_correlation_from_any_variance_components(self, monkeypatch):
        """R: rho <- s[2]/sum(s) whenever the fit succeeds, then clipped to
        [rhomin, 0.99]; components (1, -2) give rho = 2, clipped to 0.99."""
        import pylimma.dups as dups

        y, _ = self._data()
        monkeypatch.setattr(dups, "_mixed_model_2_fit", lambda *a, **k: (1.0, -2.0))
        out = dups.duplicate_correlation(y, block=np.repeat(["b", "a", "c"], 4))
        np.testing.assert_allclose(out["consensus_correlation"], 0.99)

    def test_no_fitted_gene_gives_nan_as_r(self):
        """R: the consensus is mean(numeric(0)), NaN (fixture script)."""
        from pylimma import duplicate_correlation

        y = np.array([[1.0, 3.0, 2.0, 5.0]])
        design = np.column_stack([np.ones(4), [0.0, 1.0, 0.0, 1.0]])
        out = duplicate_correlation(y, design, block=[1, 1, 2, 2])
        assert np.isnan(out["consensus_correlation"]) and np.isnan(out["cor"])
        assert np.isnan(out["atanh_correlations"]).all()

    @pytest.mark.parametrize("case", ["trend", "block_column"])
    def test_tiny_scale_design_column_keeps_its_rank_as_r(self, case):
        """qr() and mixedModel2Fit (lm.fit) decide rank relative to each
        column, so a column scaled by 1e-12 is kept (an absolute 1e-10
        cutoff dropped it)."""
        from pylimma import duplicate_correlation

        y = pd.read_csv(self.FIXTURES / "R_dct_y.csv").to_numpy()
        block = np.repeat([1, 2, 3], 2)
        column = np.arange(1.0, 7.0) if case == "trend" else (block == 2).astype(float)
        design = np.column_stack([np.ones(6), 1e-12 * column])
        expected = pd.read_csv(self.FIXTURES / "R_dct.csv").set_index("case")["cor"][case]
        out = duplicate_correlation(y, design, block=block)
        np.testing.assert_allclose(out["consensus_correlation"], expected, rtol=1e-6)


class TestDuplicateCounts:
    """R's ndups and spacing are ordinary numbers (2 is a double), so 2.0
    works; a fraction raises, where R fails in dim<-."""

    def test_whole_number_floats_match_integers(self):
        from pylimma import duplicate_correlation

        rng = np.random.default_rng(3)
        x = rng.normal(size=(12, 2))
        w = rng.uniform(0.5, 2, size=(12, 2))
        np.testing.assert_array_equal(unwrap_dups(x, ndups=2.0, spacing=1.0), unwrap_dups(x, 2, 1))
        np.testing.assert_array_equal(unwrap_dups(x, ndups=np.array(2)), unwrap_dups(x, 2))
        np.testing.assert_array_equal(unwrap_dups(x, ndups=np.array([2])), unwrap_dups(x, 2))
        np.testing.assert_array_equal(
            ave_dups(x, ndups=2.0, spacing=2.0, weights=w), ave_dups(x, 2, 2, weights=w)
        )
        assert unique_genelist(list("aabbccddeeff"), ndups=2.0) == unique_genelist(
            list("aabbccddeeff"), 2
        )
        y = rng.normal(size=(40, 4))
        np.testing.assert_allclose(
            duplicate_correlation(y, ndups=2.0)["consensus_correlation"],
            duplicate_correlation(y, ndups=2)["consensus_correlation"],
        )

    @pytest.mark.parametrize(
        "kwargs", [{"ndups": 2.5}, {"ndups": 2, "spacing": 1.5}, {"ndups": "2"}]
    )
    def test_fractions_raise(self, kwargs):
        with pytest.raises(ValueError, match="must be a whole number"):
            unwrap_dups(np.ones((12, 2)), **kwargs)

    def test_fewer_than_two_duplicates_as_r(self):
        """R: uniquegenelist returns the list for ndups <= 1;
        duplicateCorrelation warns 'No duplicates' for ndups < 2;
        unwrapdups / avedups return their input for ndups == 1, whatever
        spacing is (a copy, as R's copy-on-modify)."""
        from pylimma import duplicate_correlation

        x = np.ones((4, 2))
        for result in (unwrap_dups(x, ndups=1, spacing=None), ave_dups(x, ndups=1, spacing=0)):
            np.testing.assert_array_equal(result, x)
            assert not np.shares_memory(result, x)

        assert unique_genelist(list("abcd"), ndups=0) == list("abcd")
        with pytest.warns(UserWarning, match="No duplicates"):
            assert duplicate_correlation(np.ones((4, 3)), ndups=0)["consensus_correlation"] == 0


def test_avereps_elist_averages_other_and_vector_weights_as_r():
    """R's avereps.EList averages E, each matrix of other and the weights;
    avereps.default takes as.matrix, so vector weights are one column."""
    from pylimma import EList

    X = np.arange(36.0).reshape(12, 3)
    ids = np.repeat(list("abcdef"), 2)
    el = EList(
        {
            "E": X,
            "other": {"x": X * 2},
            "weights": np.arange(12.0),
            "genes": pd.DataFrame({"ID": ids}),
        }
    )
    out = avereps(el)
    r = pd.read_csv(
        __import__("pathlib").Path(__file__).parent / "fixtures" / "R_avereps_elist.csv"
    )
    np.testing.assert_allclose(out["E"], r.filter(like="E.").to_numpy())
    np.testing.assert_allclose(out["other"]["x"], r.filter(like="other.").to_numpy())
    np.testing.assert_allclose(out["weights"], r[["weights"]].to_numpy())
    # R: rowsum() stops for weights of another length (fixture script)
    el["weights"] = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="incorrect length for 'group'"):
        avereps(el)
