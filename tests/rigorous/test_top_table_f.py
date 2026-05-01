"""
Rigorous per-branch parity tests for pylimma.toptable.top_table_f.

Each test exercises a specific R branch of the deprecated public
``topTableF`` (R limma's topTableF.R) which is the function
pylimma's ``top_table_f`` is a port of (per the SPDX header in
toptable.py:1-6 and the docstring at toptable.py:480). The internal
``.topTableF`` in toptable.R has the same body except for the
deprecation message and is also reached via ``top_table(coef=NULL)``
multi-coef dispatch.

Tests added by a rigorous single-function audit on 2026-04-29.

Tests run a live R subprocess via helpers.run_r_comparison so any
regression surfaces immediately. Tolerances are tight (rtol=1e-8 for
stats, log10_diff<=1.0 for p-values).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.contrasts import contrasts_fit, make_contrasts
from pylimma.ebayes import e_bayes
from pylimma.lmfit import lm_fit
from pylimma.toptable import top_table_f

from ..helpers import (
    compare_arrays,
    limma_available,
    run_r_comparison,
)

pytestmark = pytest.mark.skipif(not limma_available(), reason="R/limma not available")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _three_group_fit(seed=0, n_genes=20, n_samples=12):
    """Return ``(expr_df, design, fit)`` for a 3-level cell-means model.

    ``fit`` is post-eBayes, post-contrasts (giving F-statistics across
    two pairwise contrasts so ``top_table_f`` has multi-coefficient
    F input).
    """
    rng = np.random.default_rng(seed)
    expr = rng.standard_normal((n_genes, n_samples))
    design = np.zeros((n_samples, 3))
    design[:4, 0] = 1
    design[4:8, 1] = 1
    design[8:, 2] = 1
    expr_df = pd.DataFrame(
        expr,
        index=[f"g{i + 1}" for i in range(n_genes)],
        columns=[f"s{j + 1}" for j in range(n_samples)],
    )
    fit = lm_fit(expr_df, design)
    contrasts = make_contrasts("x1-x0", "x2-x0", levels=["x0", "x1", "x2"])
    fit = contrasts_fit(fit, contrasts)
    fit = e_bayes(fit)
    return expr_df, design, fit


def _save_inputs(expr, design):
    """Write expr/design as DataFrames with row names for R."""
    if isinstance(expr, pd.DataFrame):
        df_expr = expr
    else:
        df_expr = pd.DataFrame(
            expr,
            index=[f"g{i + 1}" for i in range(expr.shape[0])],
            columns=[f"s{j + 1}" for j in range(expr.shape[1])],
        )
    df_design = pd.DataFrame(
        design,
        index=[f"s{j + 1}" for j in range(design.shape[0])],
        columns=[f"x{j}" for j in range(design.shape[1])],
    )
    return {"expr": df_expr, "design": df_design}


# Standard R script template that builds an eBayes-fitted model with
# two contrasts, mirroring _three_group_fit. We then drive topTableF
# downstream with whatever options the test wants.
_R_PRELUDE = """
suppressMessages(library(limma))
expr <- as.matrix(read.csv('{tmpdir}/expr.csv', row.names=1))
design <- as.matrix(read.csv('{tmpdir}/design.csv', row.names=1))
colnames(design) <- c('x0','x1','x2')
contrasts <- makeContrasts('x1-x0', 'x2-x0', levels=c('x0','x1','x2'))
fit <- lmFit(expr, design)
fit2 <- contrasts.fit(fit, contrasts)
fit2 <- eBayes(fit2)
"""


# ----------------------------------------------------------------------


class TestRigorousTopTableF:
    """One test per uncovered/partial R branch of topTableF()."""

    # ------------------------------------------------------------------
    # R-B2 (topTableF.R:7): always emits a deprecation message().
    # ------------------------------------------------------------------
    def test_deprecation_message_emitted(self):
        """Exercises R-B2: topTableF.R:7 - deprecation `message()`.

        R always emits "topTableF is obsolete..." via message().
        pylimma should mirror this with a DeprecationWarning (or
        equivalent FutureWarning / warnings.warn) so users porting code
        get the same heads-up. This test confirms whether pylimma emits
        any such warning; if not, it documents the divergence.
        """
        _, _, fit = _three_group_fit(seed=0, n_genes=15, n_samples=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            top_table_f(fit, number=5)
        msgs = [str(w.message).lower() for w in caught]
        assert any("obsolete" in m or "deprecat" in m or "toptablef" in m for m in msgs), (
            "R's topTableF emits a deprecation message at every call "
            "(topTableF.R:7); pylimma's top_table_f should mirror this "
            f"with a DeprecationWarning or similar. Captured: {msgs}"
        )

    # ------------------------------------------------------------------
    # R-B3 (topTableF.R:10): missing coefficients -> stop with informative msg.
    # ------------------------------------------------------------------
    def test_missing_coefficients_raises(self):
        """Exercises R-B3: topTableF.R:10.

        R: ``stop("Coefficients not found in fit")``. pylimma should
        raise an analogous error - currently raises KeyError when the
        ``fit["coefficients"]`` lookup fails (toptable.py:533).
        """
        _, _, fit = _three_group_fit(seed=1, n_genes=10)
        fit.pop("coefficients", None)
        with pytest.raises((ValueError, KeyError)):
            top_table_f(fit, number=5)

    # ------------------------------------------------------------------
    # R-B6 (topTableF.R:13): default coef column names "Coef1..N".
    # ------------------------------------------------------------------
    def test_default_coef_column_names_match_r(self):
        """Exercises R-B6: topTableF.R:13.

        R sets ``colnames(M) <- paste("Coef",1:ncol(M),sep="")`` when
        the coefficient matrix has no colnames - so the output has
        columns ``Coef1, Coef2, ...``. pylimma uses ``coef_0, coef_1,
        ...`` which differs in case, separator, and base.

        We compare against R when ``contrast_names``/``coef_names`` is
        absent from the fit (forcing the default-name path).
        """
        _, _, fit = _three_group_fit(seed=2, n_genes=10)
        # Drop name slots so the default-name branch fires
        fit.pop("contrast_names", None)
        fit.pop("coef_names", None)
        py_tt = top_table_f(fit, number=5)

        r_template = (
            _R_PRELUDE
            + "colnames(fit2$coefficients) <- NULL\n"
            + "tt <- topTableF(fit2, number=5)\n"
            + "cols <- colnames(tt)\n"
        )
        inputs = _save_inputs(*_three_group_fit(seed=2, n_genes=10)[:2])
        r_out = run_r_comparison(inputs, r_template, ["cols"])
        r_cols = [str(c).strip('"') for c in np.asarray(r_out["cols"]).ravel()]
        # R's default-name columns
        expected = [c for c in r_cols if c.startswith("Coef")]
        assert expected == ["Coef1", "Coef2"], f"Sanity: R should emit Coef1,Coef2; got {r_cols}"
        # pylimma must expose the same coefficient column names
        py_coef_cols = [
            c for c in py_tt.columns if c not in ("ave_expr", "F", "p_value", "adj_p_value")
        ]
        assert py_coef_cols == expected, (
            f"R uses 'Coef1','Coef2' as default names (topTableF.R:13); "
            f"pylimma emits {py_coef_cols}"
        )

    # ------------------------------------------------------------------
    # R-B9 (topTableF.R:20): vector genelist -> data.frame(ProbeID=...)
    # ------------------------------------------------------------------
    def test_vector_genelist_probeid_column(self):
        """Exercises R-B9: topTableF.R:20.

        R wraps a vector ``genelist`` as ``data.frame(ProbeID=genelist,
        stringsAsFactors=FALSE)`` so a ``ProbeID`` column appears in
        the output table. pylimma's top_table_f keeps it as a list and
        does not emit the column.
        """
        _, _, fit = _three_group_fit(seed=3, n_genes=12)
        ids = [f"PROBE_{i + 1}" for i in range(12)]
        py_tt = top_table_f(fit, number=10, genelist=ids)

        r_template = (
            _R_PRELUDE
            + "ids <- paste0('PROBE_', 1:nrow(expr))\n"
            + "tt <- topTableF(fit2, number=10, genelist=ids)\n"
            + "cols <- colnames(tt)\n"
        )
        inputs = _save_inputs(*_three_group_fit(seed=3, n_genes=12)[:2])
        r_out = run_r_comparison(inputs, r_template, ["cols"])
        r_cols = [str(c).strip('"') for c in np.asarray(r_out["cols"]).ravel()]
        assert "ProbeID" in r_cols, f"Sanity: expected ProbeID in R cols {r_cols}"
        assert "ProbeID" in py_tt.columns, (
            f"R wraps vector genelist as data.frame(ProbeID=...) "
            f"(topTableF.R:20); pylimma must emit a ProbeID column. "
            f"Got: {list(py_tt.columns)}"
        )

    # ------------------------------------------------------------------
    # R-B12 (topTableF.R:38): match.arg(sort.by, c("F","none")) -> error
    # ------------------------------------------------------------------
    def test_invalid_sort_by_should_error(self):
        """Exercises R-B12: topTableF.R:38.

        R's ``match.arg(sort.by, c("F","none"))`` errors on anything
        other than ``"F"`` or ``"none"`` (with partial-prefix matching
        allowed). pylimma's _get_sort_order silently accepts e.g.
        ``"AveExpr"``, ``"p"``, ``"B"`` and dispatches to a different
        sort.
        """
        _, _, fit = _three_group_fit(seed=4, n_genes=10)
        # Confirm R errors on "AveExpr"
        r_template = (
            _R_PRELUDE
            + "out <- tryCatch(\n"
            + "    {{ topTableF(fit2, number=5, sort.by='AveExpr'); 'OK' }},\n"
            + "    error=function(e) paste('ERR', conditionMessage(e))\n"
            + ")\n"
            + "out_marker <- substr(out, 1, 3)\n"
        )
        inputs = _save_inputs(*_three_group_fit(seed=4, n_genes=10)[:2])
        r_out = run_r_comparison(inputs, r_template, ["out_marker"])
        r_marker = str(np.asarray(r_out["out_marker"]).ravel()[0]).strip('"')
        assert r_marker == "ERR", f"Sanity: expected R to error; got {r_marker}"

        # pylimma should also raise
        with pytest.raises((ValueError, KeyError)):
            top_table_f(fit, number=5, sort_by="AveExpr")

    # ------------------------------------------------------------------
    # R-B14d (topTableF.R:48): fc overrides lfc silently
    # ------------------------------------------------------------------
    def test_fc_overrides_lfc_silently(self):
        """Exercises R-B14d: topTableF.R:44-49.

        R's logic is:
            if(is.null(fc)) {
                if(is.null(lfc)) lfc <- 0
            } else {
                if(fc < 1) stop(...)
                lfc <- log2(fc)   # overrides any user lfc!
            }
        Passing both fc and lfc is silently allowed in R; lfc gets
        clobbered by log2(fc). pylimma raises ValueError when both
        are passed (toptable.py:527).
        """
        _, _, fit = _three_group_fit(seed=5, n_genes=15)

        # R: passes both, lfc=99 is ignored, lfc=log2(1.2)=0.263 is used
        r_template = (
            _R_PRELUDE
            + "tt <- topTableF(fit2, number=Inf, fc=1.2, lfc=99)\n"
            + "n_rows <- nrow(tt)\n"
        )
        inputs = _save_inputs(*_three_group_fit(seed=5, n_genes=15)[:2])
        r_out = run_r_comparison(inputs, r_template, ["n_rows"])
        r_n = int(np.asarray(r_out["n_rows"]).ravel()[0])
        # Sanity: with lfc=log2(1.2)=0.263 the filter keeps most rows;
        # with lfc=99 it would keep zero.
        assert r_n > 0, f"Sanity: R kept {r_n} rows with fc=1.2; expected >0"

        # pylimma should match (silently use fc and ignore lfc)
        try:
            py_tt = top_table_f(fit, number=np.inf, fc=1.2, lfc=99)
        except Exception as exc:
            pytest.fail(
                f"R silently overrides lfc with log2(fc) when both are "
                f"given (topTableF.R:48); pylimma raised "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            assert len(py_tt) == r_n, f"R kept {r_n} rows; pylimma kept {len(py_tt)}"

    # ------------------------------------------------------------------
    # R-B19 (topTableF.R:88): genelist columns appear BEFORE coef columns
    # ------------------------------------------------------------------
    def test_column_order_genelist_before_coefs(self):
        """Exercises R-B19: topTableF.R:88.

        R: ``tab <- data.frame(genelist[o,,drop=FALSE], M[o,,drop=FALSE])``
        - genelist columns come first, then coefficient columns. pylimma
        appends genelist columns at the END of the DataFrame
        (toptable.py:653-657), changing column order.
        """
        _, _, fit = _three_group_fit(seed=6, n_genes=12)
        gl = pd.DataFrame(
            {"symbol": [f"S{i + 1}" for i in range(12)], "chrom": ["chr1"] * 12},
            index=[f"g{i + 1}" for i in range(12)],
        )
        py_tt = top_table_f(fit, number=5, genelist=gl)

        # Build expected R column order
        r_template = (
            _R_PRELUDE
            + "gl <- data.frame(symbol=paste0('S',1:nrow(expr)),\n"
            + "                 chrom=rep('chr1', nrow(expr)),\n"
            + "                 stringsAsFactors=FALSE)\n"
            + "rownames(gl) <- rownames(expr)\n"
            + "tt <- topTableF(fit2, number=5, genelist=gl)\n"
            + "cols <- colnames(tt)\n"
        )
        inputs = _save_inputs(*_three_group_fit(seed=6, n_genes=12)[:2])
        r_out = run_r_comparison(inputs, r_template, ["cols"])
        r_cols = [str(c).strip('"') for c in np.asarray(r_out["cols"]).ravel()]
        # In R, expected order: symbol, chrom, x1-x0, x2-x0, AveExpr, F, P.Value, adj.P.Val
        sym_pos = r_cols.index("symbol")
        chr_pos = r_cols.index("chrom")
        # Coef columns should come AFTER genelist columns in R
        coef_positions = [
            i
            for i, c in enumerate(r_cols)
            if c not in ("symbol", "chrom", "AveExpr", "F", "P.Value", "adj.P.Val")
        ]
        assert all(p > sym_pos and p > chr_pos for p in coef_positions), (
            f"Sanity: in R, coef cols come after genelist; got {r_cols}"
        )

        # pylimma column order
        py_cols = list(py_tt.columns)
        py_sym_pos = py_cols.index("symbol") if "symbol" in py_cols else -1
        # If symbol is at end, py_sym_pos > coef positions -> bug
        py_coef_positions = [
            i
            for i, c in enumerate(py_cols)
            if c not in ("symbol", "chrom", "ave_expr", "F", "p_value", "adj_p_value")
        ]
        assert py_coef_positions and all(p > py_sym_pos for p in py_coef_positions), (
            f"R places genelist columns BEFORE coefficient columns "
            f"(topTableF.R:88). pylimma column order: {py_cols}"
        )

    # ------------------------------------------------------------------
    # R-B20 (topTableF.R:89): NULL Amean -> AveExpr column not added
    # ------------------------------------------------------------------
    def test_null_amean_omits_avexpr_column(self):
        """Exercises R-B20: topTableF.R:89.

        R: ``tab$AveExpr <- Amean[o]``. When ``Amean`` is NULL,
        ``Amean[o]`` is NULL and ``data.frame$col <- NULL`` is a no-op
        in R (the column is not added). pylimma uses
        ``fit.get("Amean", np.full(n, nan))`` and unconditionally
        writes ``df["ave_expr"]``, so it always has the column - filled
        with NaN.
        """
        _, _, fit = _three_group_fit(seed=7, n_genes=12)
        fit.pop("Amean", None)

        py_tt = top_table_f(fit, number=5)

        r_template = (
            _R_PRELUDE
            + "fit2$Amean <- NULL\n"
            + "tt <- topTableF(fit2, number=5)\n"
            + "cols <- colnames(tt)\n"
        )
        inputs = _save_inputs(*_three_group_fit(seed=7, n_genes=12)[:2])
        r_out = run_r_comparison(inputs, r_template, ["cols"])
        r_cols = [str(c).strip('"') for c in np.asarray(r_out["cols"]).ravel()]
        assert "AveExpr" not in r_cols, (
            f"Sanity: R should drop AveExpr when Amean=NULL; got {r_cols}"
        )
        assert "ave_expr" not in py_tt.columns, (
            f"R omits AveExpr column when Amean is NULL "
            f"(topTableF.R:89); pylimma columns: {list(py_tt.columns)}"
        )

    # ------------------------------------------------------------------
    # R-B11a (topTableF.R:27-28): duplicated rownames + null genelist -> ID col
    # ------------------------------------------------------------------
    def test_duplicate_rownames_emit_id_column(self):
        """Exercises R-B11a: topTableF.R:27-28.

        Duplicated rownames + NULL genelist -> R promotes them to an
        ``ID`` column and replaces rownames with integers ``1:nrow``.
        """
        expr_df, design, fit = _three_group_fit(seed=8, n_genes=10)
        # Force duplicate rownames AT THE COEFFICIENT LEVEL (which is
        # what topTableF uses via rownames(M)). The Python implementation
        # currently never reads rownames from coefficients.
        coef_df = pd.DataFrame(
            fit["coefficients"],
            index=[f"g{i + 1}" if i % 2 == 0 else f"g{i}" for i in range(10)],
        )
        # 0->g1, 1->g1 (dup), 2->g3, 3->g3 (dup), ...
        # So at least one duplicate exists.
        fit["coefficients"] = coef_df  # DataFrame so rownames are visible

        py_tt = top_table_f(fit, number=10)

        # Compare against R which reads rownames(M) from the matrix
        r_template = (
            _R_PRELUDE
            + "rn <- ifelse(seq_len(nrow(expr)) %% 2 == 1,\n"
            + "             paste0('g', seq_len(nrow(expr))),\n"
            + "             paste0('g', seq_len(nrow(expr)) - 1))\n"
            + "rownames(fit2$coefficients) <- rn\n"
            + "tt <- topTableF(fit2, number=10)\n"
            + "cols <- colnames(tt)\n"
            + "rn_out <- rownames(tt)\n"
        )
        inputs = _save_inputs(expr_df, design)
        r_out = run_r_comparison(inputs, r_template, ["cols", "rn_out"])
        r_cols = [str(c).strip('"') for c in np.asarray(r_out["cols"]).ravel()]
        assert "ID" in r_cols, f"Sanity: R emits ID column for duplicated rownames; got {r_cols}"
        assert "ID" in py_tt.columns, (
            f"R promotes duplicated coef rownames to an ID column "
            f"(topTableF.R:28); pylimma columns: {list(py_tt.columns)}"
        )

    # ------------------------------------------------------------------
    # R-B17 (topTableF.R:76): number<1 -> empty data.frame
    # ------------------------------------------------------------------
    def test_number_zero_returns_empty(self):
        """Exercises R-B17: topTableF.R:76.

        ``if(number < 1) return(data.frame())`` -> empty frame.
        """
        _, _, fit = _three_group_fit(seed=9, n_genes=10)
        py_tt = top_table_f(fit, number=0)
        assert len(py_tt) == 0, (
            f"R returns empty data.frame when number<1 (topTableF.R:76); "
            f"pylimma returned {len(py_tt)} rows"
        )

    # ------------------------------------------------------------------
    # R-B18a + R-B22 (topTableF.R:80-91): default sort_by="F" full slot match
    # ------------------------------------------------------------------
    def test_default_sort_by_f_full_slot_match(self):
        """Exercises R-B18a, R-B19, R-B21, R-B22 simultaneously.

        Default sort_by="F" with all R defaults; compare every output
        slot to live R: rownames, F, p_value, adj_p_value, both
        coefficient columns, AveExpr.

        Note R's ``data.frame()`` (topTableF.R:88) calls ``make.names()``
        on column names, so R's output has ``x1.x0``/``x2.x0`` while
        pylimma keeps ``x1-x0``/``x2-x0``. We compare values via
        positional column index 1 and 2 in the contrast block.
        """
        expr_df, design, fit = _three_group_fit(seed=10, n_genes=15)
        py_tt = top_table_f(fit, number=15)  # all rows

        # Save coefficient columns by index (1 and 2 in R, since the
        # data.frame's first two columns are the two contrasts).
        r_template = (
            _R_PRELUDE
            + "tt <- topTableF(fit2, number=15)\n"
            + "rn <- rownames(tt)\n"
            + "AveExpr <- tt$AveExpr\n"
            + "F_stat <- tt$F\n"
            + "P_Value <- tt$P.Value\n"
            + "adj_P_Val <- tt$adj.P.Val\n"
            + "coef1 <- tt[, 1]\n"
            + "coef2 <- tt[, 2]\n"
        )
        inputs = _save_inputs(expr_df, design)
        r_out = run_r_comparison(
            inputs,
            r_template,
            ["rn", "AveExpr", "F_stat", "P_Value", "adj_P_Val", "coef1", "coef2"],
        )

        # Rank match
        r_rn = [str(s).strip('"') for s in np.asarray(r_out["rn"]).ravel()]
        py_rn = list(py_tt.index)
        assert py_rn == r_rn, f"Default sort_by='F': ranking differs.\nR={r_rn}\nPy={py_rn}"

        # Each slot
        for r_key, py_key in [
            ("AveExpr", "ave_expr"),
            ("F_stat", "F"),
            ("P_Value", "p_value"),
            ("adj_P_Val", "adj_p_value"),
            ("coef1", "x1-x0"),
            ("coef2", "x2-x0"),
        ]:
            r_arr = np.asarray(r_out[r_key]).astype(float).ravel()
            py_arr = py_tt[py_key].values.astype(float)
            res = compare_arrays(r_arr, py_arr, rtol=1e-8)
            assert res["match"], (
                f"slot {r_key}/{py_key} differs: "
                f"max_rel={res['max_rel_diff']:.2e}, "
                f"max_abs={res['max_abs_diff']:.2e}"
            )

    # ------------------------------------------------------------------
    # R-B18b (topTableF.R:82): sort_by='none' uses 1:number rows in fit order
    # ------------------------------------------------------------------
    def test_sort_by_none_preserves_order(self):
        """Exercises R-B18b: topTableF.R:82.

        ``sort.by='none'`` -> ``o <- 1:number`` (first N rows in
        original fit order, no ranking).
        """
        expr_df, design, fit = _three_group_fit(seed=11, n_genes=15)
        py_tt = top_table_f(fit, number=10, sort_by="none")

        r_template = (
            _R_PRELUDE
            + "tt <- topTableF(fit2, number=10, sort.by='none')\n"
            + "rn <- rownames(tt)\n"
            + "F_stat <- tt$F\n"
        )
        inputs = _save_inputs(expr_df, design)
        r_out = run_r_comparison(inputs, r_template, ["rn", "F_stat"])
        r_rn = [str(s).strip('"') for s in np.asarray(r_out["rn"]).ravel()]
        py_rn = list(py_tt.index)
        assert py_rn == r_rn, f"sort_by='none' rank diff:\nR={r_rn}\nPy={py_rn}"

        res_f = compare_arrays(
            np.asarray(r_out["F_stat"]).astype(float).ravel(),
            py_tt["F"].values,
            rtol=1e-8,
        )
        assert res_f["match"], (
            f"F differs after sort_by='none': max_rel={res_f['max_rel_diff']:.2e}"
        )

    # ------------------------------------------------------------------
    # R-B15a (topTableF.R:53-54): F-path lfc filter is STRICT `>`
    # ------------------------------------------------------------------
    def test_lfc_filter_strict_gt(self):
        """Exercises R-B15a: topTableF.R:54.

        ``big <- rowSums(abs(M)>lfc, na.rm=TRUE) > 0`` - STRICT `>`
        means a gene whose every coefficient equals lfc exactly is
        DROPPED from the F-path output.
        """
        expr_df, design, fit = _three_group_fit(seed=12, n_genes=15)
        # Force first gene's both contrasts to exactly equal cutoff
        cutoff = 1.0
        fit["coefficients"] = np.asarray(fit["coefficients"]).copy()
        fit["coefficients"][0, :] = cutoff

        py_tt = top_table_f(fit, number=np.inf, lfc=cutoff)
        assert "g1" not in list(py_tt.index), (
            f"F-path uses STRICT `>` (topTableF.R:54); gene exactly at "
            f"the boundary should be dropped. Got rows: {list(py_tt.index)}"
        )

    # ------------------------------------------------------------------
    # R-B15c (topTableF.R:57-59): NaN adj_p_value -> set FALSE in sig
    # ------------------------------------------------------------------
    def test_nan_adjpval_treated_as_false_in_sig(self):
        """Exercises R-B15c: topTableF.R:59 (`sig[is.na(sig)] <- FALSE`).

        When p.value < 1 and a gene's adj_p_value is NaN, R explicitly
        sets sig=FALSE for that gene (drops it). pylimma uses numpy
        comparison semantics where ``np.nan <= x`` is False, achieving
        the same effect implicitly. Confirm parity by comparing row
        counts and identities to live R for an injected NaN.
        """
        expr_df, design, fit = _three_group_fit(seed=13, n_genes=15)
        # We can't trivially inject NaN into adj_p_val mid-pipeline, but
        # we can inject NaN in F_p_value which propagates through
        # p.adjust(method="BH").
        fit["F_p_value"] = np.asarray(fit["F_p_value"]).copy()
        fit["F_p_value"][2] = np.nan

        py_tt = top_table_f(fit, number=np.inf, p_value=0.5)

        r_template = (
            _R_PRELUDE
            + "fit2$F.p.value[3] <- NA\n"
            + "tt <- topTableF(fit2, number=Inf, p.value=0.5)\n"
            + "n_out <- nrow(tt)\n"
            + "rn <- rownames(tt)\n"
        )
        inputs = _save_inputs(expr_df, design)
        r_out = run_r_comparison(inputs, r_template, ["n_out", "rn"])
        r_n = int(np.asarray(r_out["n_out"]).ravel()[0])
        r_rn = [str(s).strip('"') for s in np.asarray(r_out["rn"]).ravel()]

        # Same row count
        assert len(py_tt) == r_n, f"NaN F p-value row count: R={r_n}, Py={len(py_tt)}"
        # Same identities
        py_rn = list(py_tt.index)
        assert set(py_rn) == set(r_rn), (
            f"NaN handling differs.\nR set: {sorted(r_rn)}\nPy set: {sorted(py_rn)}"
        )
        # The NaN row (g3) must NOT appear in the output (filter applied)
        assert "g3" not in py_rn, "Gene with NaN F p-value should be filtered out when p_value<1"

    # ------------------------------------------------------------------
    # R-B11d (topTableF.R:34): duplicated rn replaced by 1:nrow integers
    # ------------------------------------------------------------------
    def test_duplicate_rn_replaced_with_integers(self):
        """Exercises R-B11d: topTableF.R:34.

        After detecting duplicates, R sets ``rn <- 1:nrow(M)``. Output
        rownames must therefore be integers 1..N (post-subset).
        """
        expr_df, design, fit = _three_group_fit(seed=14, n_genes=10)
        # Force duplicate rownames at the coefficient level
        coef_df = pd.DataFrame(
            np.asarray(fit["coefficients"]),
            index=["g1"] * 5 + ["g2"] * 5,
        )
        fit["coefficients"] = coef_df

        py_tt = top_table_f(fit, number=10, sort_by="none")
        # R behaviour: rownames -> "1","2",...,"10" (integer-like strings)
        r_template = (
            _R_PRELUDE
            + "rownames(fit2$coefficients) <- c(rep('g1',5), rep('g2',5))\n"
            + "tt <- topTableF(fit2, number=10, sort.by='none')\n"
            + "rn <- rownames(tt)\n"
        )
        inputs = _save_inputs(expr_df, design)
        r_out = run_r_comparison(inputs, r_template, ["rn"])
        r_rn = [str(s).strip('"') for s in np.asarray(r_out["rn"]).ravel()]
        py_rn = [str(x) for x in py_tt.index]
        # Both should be 1..10 in some order matching sort_by='none'
        assert r_rn == py_rn, (
            f"Duplicated rownames -> integer rn (topTableF.R:34);\nR={r_rn}\nPy={py_rn}"
        )
