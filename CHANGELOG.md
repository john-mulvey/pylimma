# Changelog

All notable changes to pylimma. pylimma is a port of R limma, and its
parity target is limma 3.66.0 (Bioconductor 3.22, R 4.5). Before 1.0, a
minor version (0.x.0) may change results or behaviour; a patch version
(0.x.y) does not.

## [Unreleased]

### Changed results

- `gene_set_test(ranks_only=False)` compares exactly rounded means, so a
  random set with the same values ties the observed set exactly (a set of
  all genes gave p 0.78 instead of 1; tie-heavy statistics were biased).
- `roast` / `mroast`: an empty set, a NaN gene weight or a negative
  `df_prior` gave small finite p-values (an empty set was even sorted
  first); as in R, the NaN propagates to NaN p-values, active proportions
  and FDR, and `mroast` sorts such sets last.
- `camera(trend_var=True)` with array weights computed the variance trend's
  average expression from the unweighted data; as R, it uses the data
  scaled by the square-root weights (p-values differed by up to ~3%).
- `inter_gene_correlation` averaged the scaled residual effects over the
  wrong axis, so `vif` and `correlation` were wrong for any data (e.g. vif
  0.672 vs R 0.349); it now matches R's `interGeneCorrelation`, including
  R's rank rule and NaN for an all-zero gene.
- `remove_batch_effect`: for a gene whose design-of-interest coefficient
  is not estimable (e.g. missing values), the whole row of coefficients
  was zeroed, so its batch effect was not removed; as R, only the
  non-estimable batch coefficients are zeroed.
- `vooma_by_group` fitted each group with an intercept-only design and
  ignored `block` and `correlation`; as R's `voomaByGroup`, each group now
  uses its rows of the design, its block and the correlation (weights
  differed by up to ~13% when the design varies within a group).
- `modify_weights` compared `str()` of status and values, so a float or
  logical status never matched (`"1.0"` vs `"1"`, `"True"` vs `"TRUE"`)
  and no weight was changed; both are now compared as R's
  `as.character()` renders them.
- `remove_batch_effect`: numeric batch values whose R labels coincide
  (0.3 and 0.1 + 0.2) are one level, as R's `as.factor()` merges them.
- `vooma` and `array_weights` decide rank as R's `qr()` does (LINPACK
  pivoting, tolerance 1e-7 relative to each column's norm) instead of an
  absolute 1e-10 cutoff on the diagonal of the QR factor. Designs with a
  near-collinear or tiny-scale column gave weights differing from R (by
  up to 9%); they now match.
- `array_weights(method="genebygene")` (and `method="auto"` with missing
  values): when missing values left a gene's design rank-deficient, the
  leverages and residuals came from an unpivoted QR and all design
  columns, giving weights up to ~3% off R or all NaN; the per-gene fit now
  uses R's pivoted QR (`lm.wfit`), and hat values its first `rank` columns.
- `voom_with_quality_weights` with as many genes as samples: the sample
  weights are always used as sample weights (R reads them as gene weights
  in that case; documented in `known_differences.rst`).
- `lm_fit` weights: whether weights are sample (array) weights now follows
  R's `asMatrixWeights`. A one-gene fit with a 1 x N weight matrix is fitted
  with per-observation weights (unweighted `cov_coefficients`, as R; was
  ~19% off), and when there are as many genes as samples a length-N
  vector is read as per-gene weights, as R (pylimma mixed both readings,
  giving `sigma` up to ~25% off).
- `lm_fit(adata, layer="voom_E")` (and other `*_E` layers) after samples
  were reordered or subset applied the design that `voom` / `vooma` stored
  in `adata.uns` row for row, i.e. to the wrong samples. The stored design
  now follows `adata.obs_names`, as R's `v[, idx]` subsets the design;
  a sample missing from it (e.g. renamed samples) or non-unique sample
  names raise a `ValueError` asking for `design=`.
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
- `goana` drops universe genes that have no annotation in `gene_pathway`,
  as R's goana keeps only annotated genes. Unannotated genes in an explicit
  universe (or among a fit's genes) inflated the gene count, so every
  p-value changed.

### Changed behaviour

- `nrot` / `nsim` in `roast`, `mroast`, `romer` and `gene_set_test` must be
  whole numbers of at least 1 (clear `ValueError`; was `ZeroDivisionError`,
  `TypeError`, or a silent p = 1 in `romer`). Whole-number floats such as
  `1e4` work.
- `camera(inter_gene_cor=...)` takes a single number (or None / NaN to
  estimate it), as R; a vector was silently used per set and now raises a
  `ValueError` (`camera_pr` still accepts one value per set, as R).
- `camera` and `romer` raise "All y values must be finite and non-NA" for
  missing or infinite values, as R errors; they returned NaN or arbitrary
  p-values.
- `wsva` with missing values raises R's "All y values must be finite and
  non-NA" (was numpy's "SVD did not converge"), and `remove_batch_effect`
  with a `batch`, `batch2` or `covariates` of the wrong length names the
  argument (was a raw numpy concatenation error).
- `remove_batch_effect` raises a clear `ValueError` for a missing `batch` /
  `batch2` label, where R fails; it treated the missing value as a batch of
  its own.
- `wsva` with a design that is not of full column rank raises R's "design
  must be full column rank" error; it returned meaningless surrogate
  variables.
- `remove_batch_effect` with a single-level `batch` or `batch2` raises a
  `ValueError`, as R's `contr.sum` does; it silently returned the data
  unchanged.
- `array_weights` warns when it uses an EList's `design` slot, which R's
  `arrayWeights` documents but its code ignores.
- `vooma_lm_fit` warns when it uses an EList's `weights` slot as
  `prior_weights` (R's `voomaLmFit` uses it only in its first fit), and
  `sample_weights=True` with such a slot raises a `ValueError` naming the
  slot.
- A fit stored in `adata.uns` is only used while `adata.var_names` match
  the fitted genes. After genes were subset, reordered or renamed,
  `top_table`, `decide_tests`, `e_bayes`, `treat`, `contrasts_fit`, `genas`,
  `goana`/`kegga` and `diff_splice` silently used the stale fit (rows for
  genes not in the object, annotation attached to the wrong genes); they
  now raise a `ValueError`. The vector-identifier workaround of
  `goana`/`kegga`/`diff_splice` on such an object is removed.
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
- `goana` / `kegga` raise `NotImplementedError` for `prior_prob` (R:
  `null.prob`), as they do for `covariate`: R then uses Wallenius'
  noncentral hypergeometric test (BiasedUrn), which is not yet ported, and
  pylimma silently ignored `prior_prob`. As in R, `covariate` and
  `prior_prob` are ignored with a warning when `universe` is not given
  (`covariate` raised), and `covariate` is ignored when no gene is DE.

### Fixed

- `camera`, `romer` and the gene-set effects behind `roast` / `mroast` /
  `fry` decide rank with R's `qr()` rule instead of an absolute 1e-10
  cutoff, so a design with a tiny-scale column is no longer rejected as
  rank-deficient; `romer` uses the pivoted QR as R.
- `romer` reforms the design for a contrast with `contrast_as_coef`, as R:
  an all-zero contrast raises R's "contrast is all zero" (it returned a
  table), a contrast matrix tests its last column (raised a shape error),
  and a float column number such as `1.0` works.
- `camera`, `fry`, `roast` and `mroast` take any numeric scalar contrast as
  a column number, as R: a float such as `1.0` raised a shape error. A
  fractional value is truncated, as R does.
- `top_romer` raised `KeyError('Up')` on every `romer` result; it reads
  `romer`'s `n_genes` / `up` / `down` / `mixed` columns.
- `camera` / `camera_pr` with a set of all genes or of no genes raised
  `ZeroDivisionError`; as in R, that set's row has NaN p-values.
- An empty list of gene sets raises R's "index is empty" in `mroast`, `fry`,
  `camera`, `camera_pr` and `romer` (it ran as a phantom empty set, or
  raised `ZeroDivisionError`). All gene-set functions now share one set
  normalisation; with ten or more unnamed sets, `camera`, `camera_pr` and
  `romer` zero-pad the names (`set01`) as `mroast` and `fry` do.
- `ids2indices` follows R's `which(identifiers %in% x)`: a duplicated
  identifier gives all its positions (was the first only) and a repeated
  member is counted once (was repeated); a list of sets, a single ID,
  numeric IDs and Python sets / generators work (raised errors).
- `gene_set_test`, `wilcox_gst` and `rank_sum_test_with_correlation` subscript
  the statistics as R's `statistics[index]`: a set of gene IDs selects by the
  names of a pandas Series (was `int('g1')`), and a boolean mask its True
  positions (was read as positions 0/1, giving wrong p-values).
- Gene sets in `roast`, `mroast`, `fry`, `camera` and `camera_pr` are resolved
  as R does: a set of gene IDs selects every row whose ID is in it (`roast`
  and `mroast` raised `int('g1')` errors and ignored `geneid`; `camera`
  matched only the first row of a duplicated ID), a boolean mask selects its
  True rows (`camera`/`camera_pr` treated it as positions 0/1), and `mroast`
  accepts (id, weight) data-frame sets. Python `set`s, dict keys and
  generators of IDs are accepted.
- `unique_genelist` on a DataFrame reset the row index; it keeps the
  selected rows' names, as R's `uniquegenelist`.
- `normalize_between_arrays(method="none")` and `remove_batch_effect`
  with no batch or covariates returned the input matrix itself (or wrote
  a layer sharing memory with `adata.X`), so editing the result edited the
  input; they return a copy, as R does.
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
- `goana` / `kegga` take a single gene ID as `de` (one set, as R; it
  raised) and a list of gene-ID lists as unnamed sets `DE1`, `DE2`, ...
  (it raised, or merged them into one set), and raise R's errors for a
  DataFrame `de` (its column labels were read as gene IDs) and for a
  `None` set (tested as the gene "None"). A dict view of IDs is one set
  (it was read as a single ID).
- `kegga` gains `restrict_universe` (R: `restrict.universe`), on both the
  gene-list and the fit path; it was silently ignored, or raised a
  `TypeError` on a fit. An empty `universe` raises R's "No genes in
  universe" (`kegga`) or "No annotated genes found in universe" (`goana`)
  instead of a `TypeError`.
- `goana` / `kegga` on a fit accept `coef` as a coefficient or contrast
  name, as R; a name always raised "Only one coef can be specified.".

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
