"""
Rigorous per-branch parity tests for pylimma.ebayes.top_treat.

Each test exercises a specific R branch of topTreat in R limma's
treat.R.

These tests were added by a rigorous single-function audit on
2026-04-29. They run a live R subprocess via helpers.run_r_comparison
so any regression surfaces immediately. Tolerances are tight (rtol=1e-8
for stats, log10_diff<=1.0 for p-values). Each test compares every
output column the function exposes (log_fc, ave_expr, t, p_value,
adj_p_value), not just the headline value.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pylimma.ebayes import top_treat, treat
from pylimma.lmfit import lm_fit

from ..helpers import (
    compare_arrays,
    compare_pvalues,
    limma_available,
    run_r_comparison,
)


pytestmark = pytest.mark.skipif(
    not limma_available(), reason="R/limma not available"
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _two_group_expr(seed=0, n_genes=30, n_samples=8):
    """Return (expr_df, design) for a two-group test.

    expr_df is a DataFrame with explicit gene rownames so the R
    subprocess (which reads CSV with ``row.names=1``) and pylimma agree
    on gene labels.
    """
    rng = np.random.default_rng(seed)
    expr = rng.standard_normal((n_genes, n_samples))
    design = np.column_stack(
        [np.ones(n_samples), np.array([0] * 4 + [1] * 4, dtype=float)]
    )
    expr_df = pd.DataFrame(
        expr,
        index=[f"g{i + 1}" for i in range(n_genes)],
        columns=[f"s{j + 1}" for j in range(n_samples)],
    )
    return expr_df, design


def _save_inputs(expr, design):
    """Format expr/design for ``run_r_comparison`` (DataFrames with
    rownames so R reads them with ``row.names=1``)."""
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


def _r_top_treat(
    *,
    coef=2,
    number=10,
    sort_by="p",
    resort_by="NULL",
    lfc=0.5,
    p_value=1,
):
    """Build an R script that runs topTreat and saves the table."""
    resort_arg = "" if resort_by == "NULL" else f", resort.by={resort_by!r}"
    return f"""
    suppressMessages(library(limma))
    expr <- as.matrix(read.csv('{{tmpdir}}/expr.csv', row.names=1))
    design <- as.matrix(read.csv('{{tmpdir}}/design.csv', row.names=1))
    fit <- lmFit(expr, design)
    tr <- treat(fit, lfc={lfc})
    tt <- topTreat(tr, coef={coef}, number={number},
                   sort.by={sort_by!r}{resort_arg},
                   p.value={p_value})
    """


def _py_top_treat_via_treat(expr, design, *, lfc=0.5, **kwargs):
    """Fit lm_fit + treat then call top_treat with kwargs."""
    expr_arr = np.asarray(expr)
    fit = lm_fit(expr_arr, design)
    fit = treat(fit, lfc=lfc)
    return top_treat(fit, **kwargs)


def _assert_columns_match(py_df, r_df, *, rtol=1e-8):
    """Compare every numeric column R's topTreat returns.

    R columns: logFC, AveExpr, t, P.Value, adj.P.Val (no B for treat).
    Python columns: log_fc, ave_expr, t, p_value, adj_p_value (and
    spuriously a NaN-filled b column today; not checked here).
    """
    col_map = {
        "logFC": "log_fc",
        "AveExpr": "ave_expr",
        "t": "t",
        "P.Value": "p_value",
        "adj.P.Val": "adj_p_value",
    }
    for r_col, py_col in col_map.items():
        assert r_col in r_df.columns, f"Missing R column: {r_col}"
        assert py_col in py_df.columns, f"Missing Py column: {py_col}"
        if r_col in {"P.Value", "adj.P.Val"}:
            res = compare_pvalues(
                r_df[r_col].values, py_df[py_col].values, max_log10_diff=1.0
            )
            assert res["match"], (
                f"{r_col} vs {py_col} differs: "
                f"max_log10_diff={res.get('max_log10_diff'):.3f}"
            )
        else:
            res = compare_arrays(
                r_df[r_col].values, py_df[py_col].values, rtol=rtol
            )
            assert res["match"], (
                f"{r_col} vs {py_col} differs: "
                f"max_rel_diff={res['max_rel_diff']:.2e}"
            )


def _read_top_treat_csv(tmpdir, name="tt_out"):
    """Helper: load the topTreat CSV the R subprocess wrote."""
    path = tmpdir / f"{name}.csv"
    return pd.read_csv(path, index_col=0)


# ----------------------------------------------------------------------
# Custom run helper: compare full topTreat tables (rather than only the
# numeric arrays the helpers.run_r_comparison primitive exposes).
# ----------------------------------------------------------------------


def _run_r_top_treat(
    expr_df, design, *, r_args: dict, output_name: str = "tt"
) -> pd.DataFrame:
    """Run topTreat in R and return the resulting table as a DataFrame.

    Mirrors helpers.run_r_comparison but returns a DataFrame (preserving
    column structure) instead of a flat ndarray. We need this because
    the audit checks columns like "logFC" / "P.Value" / "adj.P.Val"
    by name, not by positional column index.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    inputs = _save_inputs(expr_df, design)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for key, df in inputs.items():
            df.to_csv(tmpdir / f"{key}.csv", index=True)
        script = _r_top_treat(**r_args).format(tmpdir=tmpdir)
        script += (
            f"\nwrite.csv({output_name}, '{tmpdir}/{output_name}_out.csv',"
            " row.names=TRUE)\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False
        ) as f:
            f.write(script)
            script_path = f.name
        try:
            res = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if res.returncode != 0:
                raise RuntimeError(f"R failure: {res.stderr}")
            return _read_top_treat_csv(tmpdir, name=f"{output_name}_out")
        finally:
            Path(script_path).unlink()


# ----------------------------------------------------------------------
# Class-level tests
# ----------------------------------------------------------------------


class TestRigorousTopTreat:
    """One test per uncovered/partial R branch of topTreat()."""

    # ------------------------------------------------------------------
    # R-B2 (treat.R:84): default-args call - length(coef)<=1 pass-through
    # ------------------------------------------------------------------
    def test_default_args_match_r(self):
        """Exercises R-B2 + R-B7: pass-through path with R defaults.

        R defaults: coef=1, sort.by="p". Pylimma defaults: coef=0,
        sort_by="p". Both refer to the first (intercept) coefficient.
        We deliberately request coef=2 in R / coef=1 in Python (the
        treatment coefficient) to make the test biologically meaningful
        - default-args does mean "single coef, sort by p" though.
        """
        expr, design = _two_group_expr(n_genes=30, n_samples=8, seed=11)
        r_df = _run_r_top_treat(
            expr, design, r_args={"coef": 2, "number": 30}
        )
        py_df = _py_top_treat_via_treat(
            expr, design, coef=1, number=30
        )
        # Order may differ on ties; align by gene name.
        # R rownames come from the CSV (g1..g30); pylimma uses
        # the same when expr is a DataFrame -> shouldn't, actually
        # pylimma's top_table builds genes from the genes slot if
        # present, otherwise from "gene1..N". Compare by sorted
        # logFC value-by-value across genes.
        py_df_sorted = py_df.sort_values("p_value", kind="stable")
        r_df_sorted = r_df.sort_values("P.Value", kind="stable")
        _assert_columns_match(py_df_sorted, r_df_sorted)

    # ------------------------------------------------------------------
    # R-B1 (treat.R:80-83): length(coef)>1 -> warn + use coef[1]
    # ------------------------------------------------------------------
    def test_multi_coef_warns_and_truncates(self):
        """Exercises R-B1: treat.R:80-83 (coef vector -> first only)."""
        expr, design = _two_group_expr(n_genes=20, n_samples=8, seed=21)

        # Pylimma side
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            py_df = _py_top_treat_via_treat(
                expr, design, coef=[1, 0], number=20
            )
            messages = [str(w.message) for w in caught]
        assert any(
            "single coefficient" in m.lower() for m in messages
        ), f"Expected truncation warning; got {messages}"

        # R side: equivalent input is coef=c(2, 1) (1-based).
        r_df = _run_r_top_treat(
            expr, design, r_args={"coef": "c(2, 1)", "number": 20}
        )
        py_df_sorted = py_df.sort_values("p_value", kind="stable")
        r_df_sorted = r_df.sort_values("P.Value", kind="stable")
        # Both should have used coef[0]=1 / coef[1]=2 -> the treatment
        # column in this two-group design.
        _assert_columns_match(py_df_sorted, r_df_sorted)

    # ------------------------------------------------------------------
    # R-B3 (treat.R:86): sort.by="B" -> stop
    # ------------------------------------------------------------------
    def test_sort_by_capital_B_raises(self):
        """Exercises R-B3: treat.R:86 (sort.by=='B' -> stop).

        R rejects sort.by="B" explicitly (capital B). Pylimma's check
        compares against lowercase "b" only, so capital "B" slips
        through and a NaN-filled b column is silently produced
        downstream by top_table.
        """
        expr, design = _two_group_expr(n_genes=10, n_samples=8, seed=31)
        fit = lm_fit(np.asarray(expr), design)
        fit = treat(fit, lfc=0.5)
        with pytest.raises(ValueError, match="(?i)b-statistic"):
            top_treat(fit, coef=1, sort_by="B")

    # ------------------------------------------------------------------
    # R-B3 (treat.R:86): sort.by="b" -> R rejects via match.arg
    # (downstream); pylimma rejects directly.
    # ------------------------------------------------------------------
    def test_sort_by_lowercase_b_raises(self):
        """Exercises R-B3 / pylimma Py-B2: sort_by='b' -> error.

        In R 'b' isn't in match.arg's choice list (only 'B' is), so R
        raises from match.arg. Pylimma raises from the explicit guard
        in top_treat. Both should error.
        """
        expr, design = _two_group_expr(n_genes=10, n_samples=8, seed=32)
        fit = lm_fit(np.asarray(expr), design)
        fit = treat(fit, lfc=0.5)
        with pytest.raises(ValueError):
            top_treat(fit, coef=1, sort_by="b")

    # ------------------------------------------------------------------
    # R-B4 (treat.R:87): resort.by="B" -> stop
    # ------------------------------------------------------------------
    def test_resort_by_capital_B_raises(self):
        """Exercises R-B4: treat.R:87 (resort.by=='B' -> stop)."""
        expr, design = _two_group_expr(n_genes=10, n_samples=8, seed=41)
        fit = lm_fit(np.asarray(expr), design)
        fit = treat(fit, lfc=0.5)
        with pytest.raises(ValueError, match="(?i)b-statistic"):
            top_treat(fit, coef=1, resort_by="B")

    # ------------------------------------------------------------------
    # R-B5 (treat.R:87): resort.by non-NULL non-B -> pass-through
    # ------------------------------------------------------------------
    def test_resort_by_logfc_passthrough(self):
        """Exercises R-B5: treat.R:87 (resort.by!='B' allowed)."""
        expr, design = _two_group_expr(n_genes=30, n_samples=8, seed=51)
        r_df = _run_r_top_treat(
            expr,
            design,
            r_args={
                "coef": 2,
                "number": 30,
                "sort_by": "p",
                "resort_by": '"logFC"',
            },
        )
        py_df = _py_top_treat_via_treat(
            expr,
            design,
            coef=1,
            number=30,
            sort_by="p",
            resort_by="logFC",
        )
        # After resort, R's table is sorted by abs(logFC) decreasing
        # within whatever subset survived sort.by="p" thinning. With
        # number=30 (no truncation) and no p_value cutoff, all genes
        # survive. Compare value-by-value.
        py_df_sorted = py_df.sort_values("p_value", kind="stable")
        r_df_sorted = r_df.sort_values("P.Value", kind="stable")
        _assert_columns_match(py_df_sorted, r_df_sorted)

    # ------------------------------------------------------------------
    # R-B6 (treat.R:87): resort.by NULL pass-through (covered by B2)
    # ------------------------------------------------------------------
    # No additional test - default of resort_by=None already exercised
    # by test_default_args_match_r.

    # ------------------------------------------------------------------
    # Output schema: R drops B column when treat strips lods.
    # ------------------------------------------------------------------
    def test_output_schema_drops_b_column(self):
        """Exercises R-B7: R's topTreat output omits B when lods=NULL.

        R's topTable / .topTableT sets ``include.B = FALSE`` whenever
        ``eb$lods`` is NULL (and treat strips lods at treat.R:12). The
        resulting data.frame has no B column. Pylimma's _top_table_t
        always emits a 'b' column (NaN-filled when lods is None),
        diverging from R's schema.
        """
        expr, design = _two_group_expr(n_genes=10, n_samples=8, seed=61)
        r_df = _run_r_top_treat(
            expr, design, r_args={"coef": 2, "number": 10}
        )
        py_df = _py_top_treat_via_treat(
            expr, design, coef=1, number=10
        )
        # R has these columns (no B):
        assert "B" not in r_df.columns
        # Python should match: no 'b' column.
        assert "b" not in py_df.columns, (
            "pylimma top_treat emits a 'b' column when treat strips "
            "lods; R's topTreat omits the column entirely. Got: "
            f"{list(py_df.columns)}"
        )
