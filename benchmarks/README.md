# pylimma benchmarks

End-to-end wall-clock and peak-RSS benchmarks for pylimma vs R limma on
four reference datasets from the limma User's Guide. The harness is
designed to be reproducible locally and to feed results directly into
`docs/validation/benchmarks.rst`.

Input datasets and the dataset-loading helper (`generate_data.py`) live
in the sibling `../data/` directory alongside the R reference top-tables
used by the parity notebooks under `examples/`. See
[`../data/DATA_PROVENANCE.md`](../data/DATA_PROVENANCE.md) for the full
dataset table.

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

## Layout

```
benchmarks/
  README.md                 # this file
  run_python.py             # runs pipelines in Python, writes JSON
  run_r.R                   # runs pipelines in R, writes JSON
  run_benchmarks.ipynb      # loads both JSONs, plots comparison
  results/                  # JSON and profile output (committed)
```

## Running

```bash
# Thread-count pinning (mandatory for fair comparison)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd pylimma/benchmarks

# Time both runtimes.
python run_python.py      # writes results/python_<YYYYMMDD>_<platform>.json
Rscript run_r.R           # writes results/r_<YYYYMMDD>_<platform>.json

# Analyse and plot.
jupyter nbconvert --to notebook --execute run_benchmarks.ipynb \
    --output run_benchmarks.ipynb
```

The committed datasets in `../data/` are the source of truth; the
extraction scripts (`_setup_datasets.R`, `_setup_gse60450.py`,
`_setup_R_references.R`) are in `../data/` and only need to be re-run if
the pinned dataset versions change.

## Conventions

- **Repetitions**: `n_reps = 5` per dataset-pipeline combination. Trim
  the single fastest and slowest per group before reporting.
- **Statistic reported**: median, with 25th / 75th percentile error
  bars. Not the minimum - we want the typical case, not best-of-5.
- **Wall clock**: `time.perf_counter()` (Python) and
  `system.time(...)["elapsed"]` (R). Never `time.time()` or `Sys.time()`
  - those track the wall clock that NTP adjusts.
- **Memory**: `resource.getrusage(RUSAGE_SELF).ru_maxrss` (Python) and
  `gc()` + `Rprof(memory.profiling = TRUE)` (R). Report separately -
  cross-runtime memory is not directly comparable.

## Interpretation rules

- **>5x slower** than R on any dataset: open a `performance` issue
  referencing the relevant row of `results/` and the profile under
  `results/profile_*.txt`.
- **>2x R memory** on any dataset: open a `memory` issue same way.
- **Within 2x** across all datasets: the release is within the target
  band.
