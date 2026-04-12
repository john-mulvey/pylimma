"""
Time pylimma's pipelines on the four reference datasets.

Writes a JSON file ``results/python_<YYYYMMDD>_<platform>.json`` with
n_reps measurements per (dataset, pipeline) combination. ``run_r.R``
writes the same JSON schema so ``run_benchmarks.ipynb`` can consume
both without branching.

Reproducibility:
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import pylimma

import generate_data as gd


N_REPS = 5
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _rss_scale() -> int:
    # ru_maxrss is kB on Linux, bytes on macOS.
    return 1024 if sys.platform.startswith("linux") else 1


def time_and_memory(fn, *args, **kwargs):
    """Return (elapsed_seconds, peak_rss_bytes, return_value)."""
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = max(0, rss_after - rss_before) * _rss_scale()
    return elapsed, peak, out


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def pipeline_a(expr, design, contrasts):
    fit = pylimma.lm_fit(expr, design)
    fit = pylimma.contrasts_fit(fit, contrasts=contrasts)
    fit = pylimma.e_bayes(fit)
    return pylimma.top_table(fit, coef=0, number=np.inf)


def pipeline_b(counts, design, contrasts):
    v = pylimma.voom(counts, design)
    fit = pylimma.lm_fit(v["E"], design, weights=v["weights"])
    fit = pylimma.contrasts_fit(fit, contrasts=contrasts)
    fit = pylimma.e_bayes(fit)
    return pylimma.top_table(fit, coef=0, number=np.inf)


def pipeline_c(counts, design, contrasts, gene_sets):
    v = pylimma.voom(counts, design)
    fit = pylimma.lm_fit(v["E"], design, weights=v["weights"])
    fit = pylimma.contrasts_fit(fit, contrasts=contrasts)
    fit = pylimma.e_bayes(fit)
    return pylimma.camera(v["E"], index=gene_sets,
                          design=design, contrast=contrasts[:, 0])


def pipeline_d_splicing(expr, design, geneid):
    fit = pylimma.lm_fit(expr, design)
    fit = pylimma.e_bayes(fit)
    ds = pylimma.diff_splice(fit, geneid=geneid)
    return pylimma.top_splice(ds, number=np.inf)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _repeat(label, fn, *args, **kwargs):
    fn(*args, **kwargs)                                         # warm up
    elapsed, peak = [], []
    for _ in range(N_REPS):
        t, m, _ = time_and_memory(fn, *args, **kwargs)
        elapsed.append(t); peak.append(m)
    print(f"  {label:40s}  median={np.median(elapsed):.3f}s "
          f"min={min(elapsed):.3f}s max={max(elapsed):.3f}s")
    return {"elapsed_seconds": elapsed, "peak_rss_bytes": peak}


# ---------------------------------------------------------------------------
# Dataset-specific pipeline runners
# ---------------------------------------------------------------------------

def run_all(results, *, small):
    key = "all_small" if small else "all"
    data = gd.load_all(small=small)
    expr = data["expr"].values
    # Two-group BT: B-cell (B, B1-B4) vs T-cell (T, T1-T4).
    bt = data["targets"]["BT"].astype(str).str[0]               # "B" or "T"
    design, C = gd.build_two_group_design(bt)
    print(f"[{key}] shape={expr.shape}")
    results[key] = {
        "shape": list(expr.shape),
        "pipeline_a": _repeat("pipeline_a", pipeline_a, expr, design, C),
    }


def run_gse60450(results):
    data = gd.load_gse60450()
    counts = data["counts"].values.astype(float)
    # Collapse 6-level group to basal vs luminal for a clean 2-group comparison.
    celltype = data["targets"]["group"].astype(str).str.split(".").str[0]
    design, C = gd.build_two_group_design(celltype)
    print(f"[gse60450] shape={counts.shape}")
    rng = np.random.default_rng(7)
    n_genes = counts.shape[0]
    gene_sets = {
        f"set_{i}": rng.choice(n_genes, size=30, replace=False)
        for i in range(50)
    }
    results["gse60450"] = {
        "shape": list(counts.shape),
        "pipeline_a": _repeat("pipeline_a", pipeline_a,
                              np.log2(counts + 1), design, C),
        "pipeline_b": _repeat("pipeline_b (voom)", pipeline_b,
                              counts, design, C),
        "pipeline_c": _repeat("pipeline_c (voom+camera)", pipeline_c,
                              counts, design, C, gene_sets),
    }


def run_yoruba(results):
    data = gd.load_yoruba()
    counts = data["counts"].values.astype(float)
    design, C = gd.build_two_group_design(data["targets"]["gender"])
    print(f"[yoruba] shape={counts.shape}")
    results["yoruba"] = {
        "shape": list(counts.shape),
        "pipeline_a": _repeat("pipeline_a", pipeline_a,
                              np.log2(counts + 1), design, C),
        "pipeline_b": _repeat("pipeline_b (voom)", pipeline_b,
                              counts, design, C),
    }


def run_pasilla(results):
    data = gd.load_pasilla()
    counts = data["counts"].values.astype(float)
    design, C = gd.build_two_group_design(data["targets"]["condition"])
    # Synthetic gene grouping for splicing benchmark: 5 consecutive
    # rows = one gene. Real splicing workflows supply exon->gene
    # mapping from the GTF; this is a time-comparable stand-in.
    geneid = (np.arange(counts.shape[0]) // 5).astype(str)
    print(f"[pasilla] shape={counts.shape}")
    results["pasilla"] = {
        "shape": list(counts.shape),
        "pipeline_d": _repeat("pipeline_d (splicing)", pipeline_d_splicing,
                              np.log2(counts + 1), design, geneid),
    }


def main():
    results = {
        "runtime": "python",
        "python_version": platform.python_version(),
        "pylimma_version": pylimma.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "n_reps": N_REPS,
        "thread_counts": {
            k: os.environ.get(k, "") for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "datasets": {},
    }
    run_all(results["datasets"], small=True)
    run_all(results["datasets"], small=False)
    run_gse60450(results["datasets"])
    run_yoruba(results["datasets"])
    run_pasilla(results["datasets"])

    out = RESULTS_DIR / (
        f"python_{datetime.utcnow().strftime('%Y%m%d')}_"
        f"{platform.system().lower()}.json"
    )
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
