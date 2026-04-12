"""
Test helper utilities for pylimma R parity testing.

Provides comparison functions following edgePython's three-tier testing strategy:
1. Pre-computed CSV fixture comparison
2. Live R subprocess calls (optional)
3. Inline reference value comparison
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Tier 1: DataFrame comparison helpers
# -----------------------------------------------------------------------------


def compare_dataframes(
    r_df: pd.DataFrame,
    py_df: pd.DataFrame,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    check_names: bool = True,
) -> dict[str, Any]:
    """
    Compare two DataFrames column-by-column.

    Parameters
    ----------
    r_df : DataFrame
        Reference DataFrame (from R).
    py_df : DataFrame
        Python DataFrame to compare.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    check_names : bool
        Whether to check row/column names match.

    Returns
    -------
    dict
        Comparison results with keys:
        - match: bool, overall match status
        - max_rel_diff: float, maximum relative difference
        - max_abs_diff: float, maximum absolute difference
        - mismatched_cols: list, columns that differ
        - details: dict, per-column comparison details
    """
    results = {
        "match": True,
        "max_rel_diff": 0.0,
        "max_abs_diff": 0.0,
        "mismatched_cols": [],
        "details": {},
    }

    # Check shape
    if r_df.shape != py_df.shape:
        results["match"] = False
        results["details"]["shape"] = f"R: {r_df.shape}, Py: {py_df.shape}"
        return results

    # Check column names
    if check_names and list(r_df.columns) != list(py_df.columns):
        results["details"]["columns"] = {
            "r": list(r_df.columns),
            "py": list(py_df.columns),
        }

    # Compare each column
    for col in r_df.columns:
        if col not in py_df.columns:
            results["match"] = False
            results["mismatched_cols"].append(col)
            continue

        r_vals = pd.to_numeric(r_df[col], errors="coerce").values
        py_vals = pd.to_numeric(py_df[col], errors="coerce").values

        # Handle NaN
        r_nan = np.isnan(r_vals)
        py_nan = np.isnan(py_vals)
        if not np.array_equal(r_nan, py_nan):
            results["match"] = False
            results["mismatched_cols"].append(col)
            results["details"][col] = "NaN pattern differs"
            continue

        # Compare non-NaN values
        mask = ~r_nan
        if not np.any(mask):
            continue

        r_masked = r_vals[mask]
        py_masked = py_vals[mask]

        abs_diff = np.abs(r_masked - py_masked)
        rel_diff = abs_diff / (np.abs(r_masked) + atol)

        max_abs = float(np.max(abs_diff))
        max_rel = float(np.max(rel_diff))

        results["max_abs_diff"] = max(results["max_abs_diff"], max_abs)
        results["max_rel_diff"] = max(results["max_rel_diff"], max_rel)

        # Check tolerance
        close = np.allclose(r_masked, py_masked, rtol=rtol, atol=atol)
        if not close:
            results["match"] = False
            results["mismatched_cols"].append(col)

        results["details"][col] = {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "close": close,
        }

    return results


def compare_arrays(
    r_arr: np.ndarray,
    py_arr: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """
    Compare two numpy arrays.

    Returns
    -------
    dict
        Comparison results with max_rel_diff, max_abs_diff, match status.
    """
    r_arr = np.asarray(r_arr, dtype=np.float64)
    py_arr = np.asarray(py_arr, dtype=np.float64)

    if r_arr.shape != py_arr.shape:
        return {
            "match": False,
            "max_rel_diff": np.inf,
            "max_abs_diff": np.inf,
            "error": f"Shape mismatch: R={r_arr.shape}, Py={py_arr.shape}",
        }

    # Handle NaN
    r_nan = np.isnan(r_arr)
    py_nan = np.isnan(py_arr)
    if not np.array_equal(r_nan, py_nan):
        return {
            "match": False,
            "max_rel_diff": np.inf,
            "max_abs_diff": np.inf,
            "error": "NaN pattern differs",
        }

    mask = ~r_nan
    if not np.any(mask):
        return {"match": True, "max_rel_diff": 0.0, "max_abs_diff": 0.0}

    r_masked = r_arr[mask]
    py_masked = py_arr[mask]

    abs_diff = np.abs(r_masked - py_masked)
    rel_diff = abs_diff / (np.abs(r_masked) + atol)

    return {
        "match": np.allclose(r_masked, py_masked, rtol=rtol, atol=atol),
        "max_rel_diff": float(np.max(rel_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
    }


def compare_pvalues(
    r_pvals: np.ndarray,
    py_pvals: np.ndarray,
    max_log10_diff: float = 2.0,
) -> dict[str, Any]:
    """
    Compare p-values on log10 scale.

    P-values span many orders of magnitude, so direct comparison is not
    meaningful. Instead, compare on log10 scale.

    Parameters
    ----------
    r_pvals : array
        Reference p-values from R.
    py_pvals : array
        Python p-values.
    max_log10_diff : float
        Maximum allowed difference in log10(p-value). Default 2.0 means
        p-values can differ by up to 2 orders of magnitude.

    Returns
    -------
    dict
        Comparison results.
    """
    r_pvals = np.asarray(r_pvals, dtype=np.float64).ravel()
    py_pvals = np.asarray(py_pvals, dtype=np.float64).ravel()

    if len(r_pvals) != len(py_pvals):
        return {
            "match": False,
            "max_log10_diff": np.inf,
            "error": "Length mismatch",
        }

    # Clamp to avoid log(0)
    r_log = np.log10(np.maximum(r_pvals, 1e-300))
    py_log = np.log10(np.maximum(py_pvals, 1e-300))

    log_diff = np.abs(r_log - py_log)
    max_diff = float(np.max(log_diff))

    return {
        "match": max_diff <= max_log10_diff,
        "max_log10_diff": max_diff,
        "median_log10_diff": float(np.median(log_diff)),
    }


def compare_rankings(
    r_vals: np.ndarray,
    py_vals: np.ndarray,
    top_n: int | None = None,
) -> dict[str, Any]:
    """
    Compare rankings/ordering of values.

    Parameters
    ----------
    r_vals : array
        Reference values from R.
    py_vals : array
        Python values.
    top_n : int, optional
        Only compare top N rankings.

    Returns
    -------
    dict
        Comparison with Spearman correlation and top-N agreement.
    """
    from scipy.stats import spearmanr

    r_vals = np.asarray(r_vals).ravel()
    py_vals = np.asarray(py_vals).ravel()

    # Remove NaN for correlation
    mask = ~(np.isnan(r_vals) | np.isnan(py_vals))
    r_masked = r_vals[mask]
    py_masked = py_vals[mask]

    if len(r_masked) < 2:
        return {"match": False, "error": "Too few valid values"}

    corr, _ = spearmanr(r_masked, py_masked)

    # Top-N agreement
    if top_n is not None:
        r_order = np.argsort(r_masked)[::-1][:top_n]
        py_order = np.argsort(py_masked)[::-1][:top_n]
        top_agreement = len(set(r_order) & set(py_order)) / top_n
    else:
        top_agreement = None

    return {
        "match": corr > 0.99,
        "spearman_corr": float(corr),
        "top_n_agreement": top_agreement,
    }


# -----------------------------------------------------------------------------
# Tier 2: Live R subprocess helpers
# -----------------------------------------------------------------------------


def r_available() -> bool:
    """Check if Rscript is available."""
    return shutil.which("Rscript") is not None


def limma_available() -> bool:
    """Check if R and limma package are available."""
    if not r_available():
        return False
    try:
        cmd = ["Rscript", "-e", "cat(requireNamespace('limma', quietly=TRUE))"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip().lower() == "true"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def run_r_code(code: str, timeout: int = 60) -> str:
    """
    Run R code and return stdout.

    Parameters
    ----------
    code : str
        R code to execute.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    str
        stdout from R.

    Raises
    ------
    RuntimeError
        If R execution fails.
    """
    if not r_available():
        raise RuntimeError("Rscript not available")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["Rscript", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R error: {result.stderr}")
        return result.stdout
    finally:
        Path(script_path).unlink()


def run_r_comparison(
    py_data: dict[str, np.ndarray],
    r_code_template: str,
    output_vars: list[str],
    timeout: int = 60,
) -> dict[str, np.ndarray]:
    """
    Run R code with Python data and return results.

    Parameters
    ----------
    py_data : dict
        Python arrays to pass to R (saved as CSV).
    r_code_template : str
        R code template with {tmpdir} placeholder.
    output_vars : list
        Names of R variables to capture.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    dict
        R output variables as numpy arrays.
    """
    if not r_available():
        raise RuntimeError("Rscript not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save Python data
        for name, arr in py_data.items():
            if isinstance(arr, pd.DataFrame):
                arr.to_csv(tmpdir / f"{name}.csv", index=True)
            else:
                # Create DataFrame with row indices for R to read
                df = pd.DataFrame(arr)
                df.index = [f"row{i+1}" for i in range(len(df))]
                df.to_csv(tmpdir / f"{name}.csv", index=True)

        # Add output saving to R code
        r_code = r_code_template.format(tmpdir=tmpdir)
        for var in output_vars:
            r_code += f"\nwrite.csv({var}, '{tmpdir}/{var}_out.csv', row.names=TRUE)"

        # Run R
        run_r_code(r_code, timeout=timeout)

        # Load results
        results = {}
        for var in output_vars:
            out_path = tmpdir / f"{var}_out.csv"
            if out_path.exists():
                df = pd.read_csv(out_path, index_col=0)
                results[var] = df.values if df.shape[1] > 1 else df.iloc[:, 0].values

        return results


# -----------------------------------------------------------------------------
# Tier 3: Inline reference value helpers
# -----------------------------------------------------------------------------


def assert_close_to_reference(
    value: float,
    reference: float,
    tol: float = 0.001,
    name: str = "value",
) -> None:
    """
    Assert a value is close to a hard-coded R reference.

    Parameters
    ----------
    value : float
        Python computed value.
    reference : float
        R reference value (from R output, documented in comment).
    tol : float
        Absolute tolerance.
    name : str
        Name for error message.
    """
    diff = abs(value - reference)
    if diff > tol:
        raise AssertionError(
            f"{name}: Python={value:.6f}, R={reference:.6f}, diff={diff:.6f} > tol={tol}"
        )


def assert_pvalue_close(
    value: float,
    reference: float,
    max_log10_diff: float = 1.0,
    name: str = "p-value",
) -> None:
    """
    Assert a p-value is close to reference on log10 scale.

    Parameters
    ----------
    value : float
        Python p-value.
    reference : float
        R reference p-value.
    max_log10_diff : float
        Maximum allowed difference in log10 scale.
    name : str
        Name for error message.
    """
    log_diff = abs(np.log10(max(value, 1e-300)) - np.log10(max(reference, 1e-300)))
    if log_diff > max_log10_diff:
        raise AssertionError(
            f"{name}: Python={value:.2e}, R={reference:.2e}, "
            f"log10_diff={log_diff:.2f} > {max_log10_diff}"
        )
