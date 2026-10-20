# Changelog

All notable changes to pylimma. pylimma is a port of R limma, and its
parity target is limma 3.66.0 (Bioconductor 3.22, R 4.5). Before 1.0, a
minor version (0.x.0) may change results or behaviour; a patch version
(0.x.y) does not.

## [Unreleased]

### Changed results

- `lm_fit(ndups > 1)`: `fit["genes"]` kept one label per probe, so
  `top_table` attached probe i's annotation to fitted gene i (wrong genes
  for every row past the first); it is now `unique_genelist(genes, ndups,
  spacing)`, as R's `lmFit`.
- numpy masked arrays as expression data: the mask was ignored and the
  hidden values used as data; masked cells are now missing values (NaN),
  as R's NA.
- `cum_overlap`: p-values below ~1e-16 were lost (computed as
  `1 - cdf`), so `p_value`, `p_min` and `n_min` were wrong for strongly
  concordant lists; they now use the upper tail directly, as R.
- `duplicate_correlation` (and so `lm_fit(block=..., correlation=None)`,
  `voom(block=...)`): the per-gene REML fit allowed one iteration fewer than
  `statmod::glmgam.fit`, which runs up to `maxit + 1`. Genes whose fit hits
  the cap now match R; because the consensus correlation is a trimmed mean
  over genes, every gene's t-statistic in a blocked fit can shift (up to
  ~1e-2 relative on proteomics data with missing values).
- `fit_gamma_intercept` (used by `pred_fcm` and `genas`) follows R's
  iteration control: up to `maxit + 1` updates and a warning "iteration
  limit maxit exceeded" when the limit is reached. Results change only for
  fits that hit the limit; inputs with no root (e.g. all-zero `y`) now give
  `-inf` with the warning instead of `0.0` or a huge finite number.

### Changed behaviour

- `decide_tests(method="hierarchical")` / `"nestedF"` on a fit without
  F-statistics (rank-deficient design) raises a `ValueError` explaining
  why, instead of `KeyError: 'F_p_value'`. R returns all zeros by accident.
- `top_table(number=...)`: a negative `number` returned all but the last
  `|number|` rows and a non-integer raised `TypeError`; as in R, `number`
  below 1 now gives an empty table and a non-integer is truncated.
  `number=NaN` raises a clear `ValueError` (R fails obscurely).
- `fit_gamma_intercept` raises a clear `ValueError` for missing values, an
  undefined (0/0) Newton step, or an `offset` that is neither scalar nor the
  length of `y` (see `docs/validation/known_differences.rst`). An empty `y`
  gives NaN, as in R.

### Fixed

- `unique_genelist` on a DataFrame reset the row index; it keeps the
  selected rows' names, as R's `uniquegenelist`.
- scipy.sparse expression matrices, and AnnData whose `X` or layer is
  backed (`backed="r"`), lazy or dask-held, failed with "setting an
  array element with a sequence" or "got 0D" in `lm_fit`, `voom`,
  `remove_batch_effect`, the normalisation functions and others; they
  are densified as R's `as.matrix()` does for a sparse `Matrix`.
- `classify_tests_f` with one coefficient returned -9223372036854775808
  for a gene with NA t-statistic; it returns NaN (R's NA), as the
  multi-coefficient path already did.
- `fitmixture` with 2 arrays raised `ZeroDivisionError`; as in R, `A` and
  `M` are estimated and `stdev` is Inf or NaN (no residual degrees of
  freedom).
- `choose_lowess_span(n=0)` raised `ZeroDivisionError`; it returns 1, as
  in R.
- `duplicate_correlation(trim=)` follows R's `mean(x, trim=)`: `trim=0.5`
  returned NaN and larger or negative values raised; they now give the
  median and the untrimmed mean respectively.
- `top_table` with several coefficients no longer emits the
  "top_table_f is obsolete" `DeprecationWarning`; as in R, only a direct
  call to `top_table_f` does.
- `e_bayes(proportion=1)` and `pred_fcm(prop_true_null_method="hist")` when
  the estimated proportion of true nulls is 0 returned `ZeroDivisionError`;
  lods are now Inf, as in R (including `pred_fcm(all_de=False)`, where the
  probability of differential expression becomes 1).

## [0.2.0] - 2026-09-30

A branch-by-branch audit against R limma 3.66.0. Every fix below is paired
with an R fixture or a live-R test that forces the branch it corrects.
Results that differed from R now match it, so **re-running an analysis may
give different numbers** in the cases listed under "Changed results".

### Changed results

- `fit_f_dist` / `squeeze_var` / `e_bayes(trend=True)`: the spline trend now
  follows R's `ns()` and `lm.fit` - interior knots that coincide with a
  boundary knot are moved inwards (scale was off by up to 58% when a third
  or more of the covariate tied at an extreme, e.g. many all-zero genes),
  the trend is linear beyond the boundary knots, and a rank-deficient basis
  uses R's pivoting and residual effects (up to 51%). When every interior
  knot is on one boundary, "Problem with covariate" is raised as in R
  instead of a silent linear fit.
- `fit_f_dist_robustly` / `e_bayes(robust=True)`: unequal `df1` is mapped to
  `max(df1)` through the F quantiles (df2 was off by 31%), ranks average
  ties and orderings are stable (df2_shrunk off by up to 9%), log tail
  probabilities stay finite where the tail underflows, `trim >= 0.5` gives
  the median, and `prob_outlier` / `df2_outlier` are returned.
- `fit_f_dist_unequal_df1` (robust `e_bayes` with genewise df, e.g. data
  with missing values): ranks were inverted and flagged about half of the
  genes as outliers; lowess weight limits and the robust refit now follow R.
- `decide_tests`: both methods ported branch by branch (NA handling,
  hierarchical method for fits and for p-value matrices);
  `classify_tests_f` follows R on rank-deficient fits and at `df = Inf`.
- `top_table`: the F-test for a subset of coefficients (including the
  default call on a design with an intercept) is recomputed from those
  coefficients, as R does.
- `contrasts_fit`: statements now follow R's order (estimable coefficients
  before pruning all-zero rows, R's `cov2cor` and orthogonality test).
- `lm_fit`: rank decisions use a port of R's LINPACK `dqrdc2` pivoting (a
  large-scale covariate no longer drops columns).
- `loess_fit` / `weighted_lowess`: port of R's `weighted_lowess.c` and every
  `loessFit` branch (results differed by ~1e-5 when n > npts).
- `fry`: remaining branches ported; infinite prior df no longer gives NaN.

### Changed behaviour

- `contrasts_fit(fit, coefficients=...)` returns `fit[:, coefficients]` as
  R does: test statistics are kept and F is recomputed, so `top_table` works
  on the result. `MArrayLM` column subsetting (`fit[:, j]`) follows
  `[.MArrayLM`: it also subsets `cov_coefficients`, `contrasts` and
  `var_prior`, and recomputes F. The `contrasts` slot stores the matrix as
  supplied.
- `lm_fit(..., block=..., correlation=None)` and `voom(..., block=...)`
  estimate the correlation with `duplicate_correlation`, as R does;
  omitting `correlation` raises R's "the correlation must be set" error.
- `trigamma_inverse` raises TypeError for logical, character or complex
  input and returns `inf` at 0 without a warning.
- Several error messages now use R's wording (`contrasts_fit`, `lm_fit`,
  `fit_f_dist_robustly`).
- `lm_fit` and `gls_series` forward extra keyword arguments to
  `duplicate_correlation` / `mrlm`, as R's `...` does.
- `lm_fit` on AnnData no longer stores `adata.obs` as the fit's targets.
- `plot_rldf(..., plot=False)` no longer needs matplotlib.

### Added

- `get_fit(adata, key)`: a standalone `MArrayLM` from `adata.uns[key]`.
- `layer=` and `weights_layer=` for `roast`, `mroast`, `fry` and `camera`;
  `layer=` for `romer` and `wsva`; `top_table_f` accepts AnnData.
- Gene-set tests accept a contrast given as a design column name.
- `avereps` / `aver_arrays` on AnnData average X and every layer, as R's
  EList methods do.

### Fixed

- DataFrames with pandas nullable dtypes, a leading gene-id column and fits
  reloaded from `.h5ad` now give the same results as the other input types.
- `goana` / `kegga` take gene ids from the fit and raise R's error for a
  wrong-length `geneid`.

### Documentation and examples

- The four R-vs-pylimma notebooks are shown under Validation, with a
  provenance cell pinning R 4.5, Bioconductor 3.22 and limma 3.66.0;
  notebooks re-run without local paths in their outputs.
- `known_differences.rst` gains "Deliberate divergences" for limma bugs
  pylimma does not reproduce (`decide_tests` genewise p-values, `genas`
  re-centring, `fit_f_dist_unequal_df1` with two informative values).
- Preprint citation added.
- Parity tolerances tightened to rtol 1e-6 for deterministic statistics and
  1e-6 on the log10 scale for p-values; `tests/rigorous/` adds live-R tests.

## [0.1.0] - 2026-05-01

First release.
