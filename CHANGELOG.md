# Changelog

All notable changes to pylimma are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sphinx documentation under `docs/` (installation, quickstart, API
  reference, tutorials, validation pages).
- Four-dataset benchmark suite committed to `benchmarks/`: ALL
  (Chiaretti 2004), GSE60450 (Fu 2015), Yoruba HapMap (Pickrell
  2010), and Pasilla (Brooks 2011). Data and metadata are bundled
  in `benchmarks/data/` as gzipped CSVs, with provenance and
  licensing documented in `benchmarks/data/DATA_PROVENANCE.md`.
- Four R-vs-Python parity notebooks under `examples/`, one per
  dataset, reproducing the side-by-side comparison pattern used by
  edgePython. Each notebook loads a pre-computed R reference
  top-table and runs the pylimma pipeline live, then shows
  per-column relative differences and R-vs-Python scatter plots.

### Parity
- Reference toolchain: R 4.5.2, Bioconductor limma 3.66.0 + edgeR
  4.8.2. pylimma reproduces R limma's top-table output at
  floating-point precision across the four benchmark datasets.
  Median relative differences are 1.6e-14 (ALL, core), 6.9e-12
  (GSE60450, voom), 7.8e-11 (Yoruba, voom), and 6.3e-14 (Pasilla,
  splicing) - well inside pylimma's published `rtol=1e-6`
  tolerance. Full numbers in `docs/validation/benchmarks.rst`.

### Performance
- Across the reference benchmark suite, pylimma is within 2x of R
  limma on every pipeline-dataset combination. Core pipeline
  (`lm_fit` -> `e_bayes` -> `top_table`) is within 15% of R on all
  four datasets. `voom` is ~2x slower than R (LOWESS fit dominates)
  - the largest gap and the clearest follow-up candidate.

### Deferred
- GitHub Actions workflows (`tests.yml`, `lint.yml`, `docs.yml`,
  `r_parity.yml`, `publish.yml`) - will land once pylimma is on a
  public GitHub repository.
- PyPI release and ReadTheDocs hosting - follow the public-repo work.

## [0.1.0] - 2026-04

First release tag. Covers the complete port of R limma's linear
modelling, voom, normalisation, batch correction, gene-set testing,
differential splicing, and diagnostic plotting functions.

### Added
- **Linear modelling**: `lm_fit`, `lm_series`, `contrasts_fit`,
  `make_contrasts`, `model_matrix`, `e_bayes`, `treat`, `top_table`,
  `top_table_f`, `decide_tests`, `classify_tests_f`, `squeeze_var`,
  `fit_f_dist`, `fit_f_dist_robustly`.
- **RNA-seq**: `voom`, `voom_with_quality_weights`, `vooma`,
  `vooma_lm_fit`.
- **Duplicates and weights**: `duplicate_correlation`, `ave_dups`,
  `avereps`, `array_weights`, `array_weights_quick`.
- **Normalisation**: `normalize_between_arrays` (single-channel),
  `normalize_quantiles`, `normalize_median_values`,
  `normalize_cyclic_loess`, `background_correct`, `normexp_fit`,
  `normexp_signal`, `aver_arrays`.
- **Batch and surrogate variables**: `remove_batch_effect`, `wsva`.
- **Gene-set testing**: `ids2indices`, `roast`, `mroast`, `fry`,
  `camera`, `camera_pr`, `inter_gene_correlation`, `romer`,
  `gene_set_test`, `rank_sum_test_with_correlation`.
- **Statistical utilities**: `zscore_t`, `tricube_moving_average`,
  `convest`, `prop_true_null`, `detection_p_values`, `qqt`.
- **Differential splicing**: `diff_splice`, `top_splice`,
  `plot_splice`.
- **Plotting**: `plot_with_highlights`, `plot_ma`, `plot_md`,
  `volcano_plot`, `plot_sa`, `plot_densities`, `plot_mds`,
  `venn_counts`, `venn_diagram`, `coolmap`, `barcode_plot`.
- **Data classes**: `EList` and `MArrayLM` as `dict` subclasses, with
  AnnData round-trip via `get_eawp` / `put_eawp`.

### Known differences from R limma
See `docs/validation/known_differences.rst` for full details.
Summary:

- Monte-Carlo rotation tests (`roast`, `mroast`, `romer`,
  `gene_set_test`) use NumPy's PCG64 RNG and so differ from R's
  Mersenne-Twister-driven p-values within documented sampling
  tolerance (empirically ~0.3 log10 at `nrot=999`). Deterministic
  summaries match R to `rtol=1e-15`.
- `normexp_fit(method="saddle")` can drift up to ~2e-4 from R on
  the flat saddle-likelihood plateau (scipy Nelder-Mead vs R
  `nmmin` termination rules). Other normexp methods match R at
  floating-point precision.
- All other R-parity tests pass at `rtol=1e-6` or better.

### Licence
- GPL-3.0-or-later (derivative of R limma, "GPL (>=2)"). Per-file
  SPDX headers credit upstream limma / MASS / statmod / base R
  splines / affy contributors.
