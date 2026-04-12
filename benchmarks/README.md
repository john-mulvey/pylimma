# pylimma benchmarks

End-to-end wall-clock and peak-RSS benchmarks for pylimma vs R limma
on four real datasets from the limma User's Guide. The harness is
designed to be reproducible locally and to feed results directly into
`docs/validation/benchmarks.rst`.

## Datasets (all real, no simulated data)

| Slot                      | Dataset              | Shape          | User's-guide section | Pipeline                                  |
|---------------------------|----------------------|----------------|----------------------|-------------------------------------------|
| Microarray, factorial     | Estrogen (affy)      | ~12,625 x 8    | 17.2                 | A (core) / factorial contrasts            |
| RNA-seq, two-group        | GSE60450 (mammary)   | ~27,000 x 12   | 15 / Law 2014        | A, B (voom), C (camera)                   |
| RNA-seq, scaling          | Yoruba HapMap        | ~52,000 x 69   | 18.1                 | A, B (voom)                               |
| Splicing                  | Pasilla              | ~70,000 x 7    | 18.2                 | D (diff_splice)                            |

The "overhead / fixed-cost floor" measurement is a 50-gene subset of
Estrogen; no synthetic dataset is needed.

Preprocessing (CEL-file reading, RMA normalisation, TMM-normalisation
of raw counts) is done in `generate_data.R` so both runtimes start
from an identical pre-normalised CSV. Benchmarks therefore time
limma-side code only, not preprocessing.

## Pipelines

```
A (core, all datasets):
    lm_fit -> contrasts_fit -> e_bayes -> top_table

B (voom, GSE60450 and Yoruba):
    voom -> lm_fit -> contrasts_fit -> e_bayes -> top_table

C (gene-set testing, GSE60450):
    voom -> lm_fit -> contrasts_fit -> e_bayes -> camera(50 sets)

D (splicing, Pasilla):
    diff_splice -> top_splice
```

Plus one-off single-function timings for `squeeze_var(trend=True,
robust=True)`, `duplicate_correlation`, and `array_weights` on
GSE60450.

## Layout

```
benchmarks/
  README.md                 # this file
  generate_data.R           # pulls / normalises datasets, writes CSVs
  generate_data.py          # Python counterpart reading the R-produced CSVs
  run_python.py             # runs pipelines in Python, writes JSON
  run_r.R                   # runs pipelines in R, writes JSON
  run_benchmarks.ipynb      # loads both JSONs, plots comparison
  data/                     # datasets (gitignored)
  results/                  # JSON and profile output (committed)
```

## Running

```bash
# Thread-count pinning (mandatory for fair comparison)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 1. Fetch / preprocess datasets (requires R + Bioconductor).
cd pylimma/benchmarks
Rscript generate_data.R

# 2. Symlink / copy CSVs into Python's view.
python generate_data.py

# 3. Time both runtimes.
python run_python.py      # writes results/python_<YYYYMMDD>_<platform>.json
Rscript run_r.R           # writes results/r_<YYYYMMDD>_<platform>.json

# 4. Analyse and plot.
jupyter nbconvert --to notebook --execute run_benchmarks.ipynb \
    --output run_benchmarks.ipynb
```

## Conventions

- **Repetitions**: `n_reps = 5` per dataset-pipeline combination.
  Trim the single fastest and slowest per group before reporting.
- **Statistic reported**: median, with 25th / 75th percentile error
  bars. Not the minimum - we want the typical case, not best-of-5.
- **Wall clock**: `time.perf_counter()` (Python) and
  `system.time(...)["elapsed"]` (R). Never `time.time()` or
  `Sys.time()` - those track the wall clock that NTP adjusts.
- **Memory**: `resource.getrusage(RUSAGE_SELF).ru_maxrss` (Python)
  and `gc()` + `Rprof(memory.profiling = TRUE)` (R). Report
  separately - cross-runtime memory is not directly comparable.
- **Seeds**: fixed in `generate_data.R` for reproducibility.
  Benchmark pipelines with stochastic components (roast, camera)
  seed with the same integer in both runtimes.

## Interpretation rules

- **>5x slower** than R on any dataset: open a `performance` issue
  referencing the relevant row of `results/` and the profile under
  `results/profile_*.txt`. Do not fix in the benchmarks phase.
- **>2x R memory** on any dataset: open a `memory` issue same way.
- **Within 2x** across all datasets: annotate the next CHANGELOG
  entry with "pylimma is within 2x of R limma across the reference
  benchmark suite."

Performance optimisation (numba, Cython, compiled extensions) is
downstream of these issues, not part of the benchmarks phase.

## Dataset provenance and licensing

Full provenance, citations, and licensing are in
[`data/DATA_PROVENANCE.md`](data/DATA_PROVENANCE.md). Summary:

| Dataset | Source | Licence | Canonical paper |
|---------|--------|---------|-----------------|
| ALL (Chiaretti adult leukemia) | Bioc `ALL` package 1.52.0 | Artistic-2.0 | Chiaretti 2004, Blood |
| GSE60450 (mouse mammary) | GEO supplementary | Public (NCBI) | Fu 2015, Nature Cell Biol |
| Pasilla (Drosophila splicing) | Bioc `pasilla` package 1.38.0 | LGPL | Brooks 2011, Genome Res |
| Yoruba HapMap (Pickrell) | Bioc `tweeDEseqCountData` 1.48.0 | MIT | Pickrell 2010, Nature |

All four upstream licences are compatible with pylimma's
GPL-3.0-or-later. If you publish work using these datasets, cite the
upstream papers (not pylimma).

The CSVs in `data/` were extracted once by `_setup_datasets.R` (R-side
data packages) and `_setup_gse60450.py` (GEO) on the maintainer's
machine. The data packages were uninstalled after extraction, so the
committed CSVs are the sole source of truth going forward. Users
running the benchmark do **not** need to install any Bioconductor data
packages.
