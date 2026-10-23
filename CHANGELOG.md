# Changelog

All notable changes to pylimma. pylimma is a port of R limma, and its
parity target is limma 3.66.0 (Bioconductor 3.22, R 4.5). Before 1.0, a
minor version (0.x.0) may change results or behaviour; a patch version
(0.x.y) does not.

## [Unreleased]

### Changed results

- `duplicate_correlation` returns R's NaN consensus correlation when no
  gene can be fitted (too few observations per gene); it returned 0,
  which reads as an estimated zero correlation.
- `duplicate_correlation` decides the design's rank as R does (relative
  to each column, R's `qr()` and statmod's `lm.fit`), so a design column
  on a tiny scale (e.g. 1e-12) is kept; an absolute 1e-10 cutoff dropped
  it and gave a different correlation.
- `pool_var` gives R's `NaN` df when every variance is 0 and `NaN` (R's
  NA) when a variance is missing, where it gave `inf`; an infinite df
  still comes only from infinite input df, as in R.
- `zscore_gamma` (and `zscore(..., "gamma")`) gave `inf` / `-inf` in the
  far tails, where the gamma tail probability underflows (q = 1000 with
  shape 2, rate 1.5; q = 1e-200); the log tail probability is now
  computed directly there, as R's `pgamma(log.p=TRUE)`, and gives R's
  finite z-scores.
- `qqt` and `qqf` place the theoretical quantiles at R's `ppoints(n)`
  (`(i - 3/8) / (n + 1/4)` for up to 10 values), not always at
  `(i - 0.5) / n`, and tied values keep their order as in R's
  `order()`. `qqf` with an infinite `df1` or `df2` gives R's quantiles
  where it gave NaN.
- `prop_true_null(method="hist")` counted a p-value lying on a bin edge
  (such as 0.05 or 1) in the bin above it; as R's `cut()`, the bins are
  (left, right], so it counts in the bin below.
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
- The same holds in the plot functions that take expression data (also
  for an EList's weights, and in `mdplot` and `plotlines`), which plotted
  the hidden values; they also accept scipy.sparse matrices, which raised
  "setting an array element with a sequence".
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
- `goana` / `kegga` compare gene and pathway IDs as R's `as.character()`
  does, so a numeric ID such as `1.0` matches `1` (it became "1.0" and
  never matched), and drop missing IDs (`None`, NaN, pandas NA) as R's
  `is.na` does. A pandas NA became the gene "<NA>"; the strings "NA", "nan"
  and "" are now IDs, as in R.
- `e_bayes(robust=True)` with unequal residual df converts t-statistics
  to the largest df as R does for very large |t| (beyond about 58 at
  1,000 arrays). The conversion's log tail probability underflowed to
  `-inf`, and its inverse used an asymptotic formula that is wrong unless
  t^2 is much larger than df (R 139.3, pylimma 4.6). This overestimated
  `var_prior` (by about 18% in a 1,000-array simulation) and shifted every
  B-statistic; t-statistics and p-values were unaffected.
- `fit_f_dist_robustly` with unequal `df1` and an infinite non-robust
  `df2` moves the variances to the largest `df1` with the chi-squared
  limit, as R's `pf` / `qf` with `df2 = Inf`. scipy's F distribution gave
  NaN there, so the scale was wrong (0.470, R 0.545) and `tail_p_value`
  was missing. `squeeze_var(robust=True)` was not affected.
- The far F tail behind `fit_f_dist_robustly`'s outlier p-values is exact
  for large `df2`: its series stopped at 1,000 terms, so at `df2 = 1e6`
  a log tail probability of R's -1185.08 came out as -1185.17.
- `make_contrasts` with a categorical (`pd.Categorical` or a categorical
  Series) as `levels` used its values, giving repeated rows; as R's
  `levels(factor)`, it uses the categories in category order.
- `normalize_vsn` sets up L-BFGS-B as vsn does: from vsn's
  `pstartHeuristic` (b = 1) with 5 correction pairs (scipy's default is
  10) and no cap on function evaluations. Output now matches R's
  `normalizeVSN` to rounding error (it differed by up to 2.4e-4, which was
  documented as an accepted difference), and data with many missing
  values no longer runs away to a worse optimum (up to 1.8 log2 units off
  R).
- `normalize_vsn` selects the rows for each robust (LTS) refit as vsn
  does: a row with a missing value is left out (outside the lowest
  intensity slice), slices are right-closed intervals of average ranks,
  and an all-missing slice selects nothing. Rows with missing values were
  kept, giving output up to ~0.1 log2 units off R for data with missing
  values; slice boundaries moved a few rows between slices.
- `normalize_vsn` leaves rows that are all missing out of the fit, with
  vsn's warning ("N rows were removed since they contained only NA
  elements."); they stay NaN in the output. They were ranked last in the
  robust selection, shifting its slice boundaries. An all-missing matrix
  raises an error.
- `voom` (so also `voom_with_quality_weights`), `vooma`, `vooma_lm_fit`,
  `vooma_by_group`, `loess_fit` (so also the trends of `e_bayes(trend=True)`
  and `squeeze_var(covariate=)`) and cyclic loess normalisation fit their
  lowess curves with a port of R's `lowess` (stats `lowess.c`) instead of
  statsmodels' lowess. statsmodels' version fitted tied x values in an
  order-dependent way when one value filled the window, fitted constant x
  with the first y value, depended on the scale of x and differed from R
  by about 2e-6 relative in voom weights on typical data. For example
  `squeeze_var(covariate=, robust=True)` with a two-valued covariate gave a
  prior variance of 2.16 against R's 0.84, `vooma` on 20 genes with span
  0.2 gave weights up to 82% from R's, `e_bayes(trend=True, robust=True)`
  on 12 genes gave prior variances up to 210% from R's and `loess_fit` with
  x scaled by 1e-6 was 0.48 off. The trend is interpolated as R's
  `approxfun`, which leaves out a gene with no residual df (NA sigma):
  `vooma_lm_fit` with such a gene gave weights up to 23% from R's. The
  smoother is about 3x slower than statsmodels' (0.38 s against 0.12 s for
  50,000 points at span 0.7): the voom pipeline on GSE60450 takes 1.91 s
  instead of 1.85 s, but cyclic loess, which is all lowess fits, takes
  about 3.3 times as long (20,000 x 4: 1.77 s instead of 0.55 s for
  `"fast"`, 2.49 s instead of 0.73 s for `"pairs"`).
- `plot_ma_3by2` on a matrix or EList plots what R's `plotMA3by2` plots:
  each array's M = E - rowMeans(E) against A = rowMeans(E), labelled "A"
  and "M", with common limits over all of A and M, the legend on the first
  plot of each page and the status from `genes`. It plotted each array
  against the mean of the other arrays, with M off by a factor n/(n-1) and
  limits from those values. Without column names there is no title (it was
  "Array i").
- `coolmap(cluster_by="de pattern")` clusters the columns on the unscaled
  matrix, as R does; it used the row z-scores, so the column order could
  differ from R's.
- `coolmap` with `linkage_row` / `linkage_col` of `"ward.D"`, `"centroid"`
  or `"median"` clusters as R's `hclust` does; `"ward.D"` ran scipy's ward
  (R's `"ward.D2"`) and centroid and median used scipy's squared-distance
  updates, so the dendrogram orders differed from R's.

### Changed behaviour

- Subsetting an `EList` or `MArrayLM` follows R: `head` / `tail` with a
  negative `n` leave out the last / first `-n` rows (they returned no
  rows) and a fractional `n` is truncated (it raised `TypeError`);
  `obj[True, :]` selects every row (it selected the first); a name that
  is not among the row or column names raises R's "Subscript not found in
  rownames" / "colnames", also when there are no names (it raised "invalid
  literal for int()"), which `contrasts_fit(coefficients=...)` reports
  too.
- `qqt` and `qqf` draw what R draws: the points only (pylimma added a
  dashed reference line), titled "Student's t Q-Q Plot" also for
  `df=np.inf` (it said "Normal Q-Q Plot"), and take R's `main`, `xlab`,
  `ylab` and `ylim`. Without matplotlib they raise the same `ImportError`
  as the other plot functions; they warned and returned.
- `gene_set_test`, `wilcox_gst`, `rank_sum_test_with_correlation` and
  `barcode_plot` raise for a set position outside 0..n-1; a negative
  position counted from the end and one past the end raised an
  IndexError (R: NA p-values, or a negative position drops the gene). A
  logical set must have one value per statistic (R recycles a shorter
  one; pylimma tested only the given values).
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
- `goana_trend` raises a clear error for negative, fractional or
  out-of-range `index_de` positions (a negative position counted from the
  end, a fraction was truncated and an out-of-range one raised
  `IndexError`).
- `make_contrasts` raises a clear error for levels that are not names: a
  design matrix without column names (previously given invented names
  `x0`, `x1`, ...) and numeric values. Duplicated level names raise an
  error (R gives the indicator to the last copy only). A single string is
  one level, not one level per character.
- `make_contrasts` rejects level names that R rejects, such as `_x`, and
  contrast expressions are limited to R arithmetic on the levels and
  numbers (`+ - * / ^` and brackets); other names, function calls and
  other Python syntax raise a clear error.
- `make_contrasts` raises an error when keyword contrasts are combined with
  `contrasts=`, as R does for named `...` arguments; they were silently
  combined.
- Formula strings containing `^` (`model_matrix`, `lm_fit`, `voom` and the
  other formula entry points) raise an error asking for `**`: patsy
  evaluated `I(age^2)` as bitwise XOR, giving silently wrong designs for
  integer columns (1, 4, 9, ... became 3, 0, 1, ...).
- Formula strings with an ordered categorical that would be coded with
  contrasts raise an error asking for `C(o, Poly)` (R's `contr.poly`) or
  `C(o, Treatment)`; patsy silently used treatment contrasts where R uses
  polynomial ones.
- Formula strings are evaluated with the columns of the data, patsy's
  built-ins, `np` and `pd` only. Names were also looked up among the
  imports of pylimma's own modules, so a missing column called `io` or
  `warnings` resolved to a Python module and gave an obscure error.
- `normalize_vsn` checks its input as vsn does: fewer rows than the new
  `min_data_points_per_stratum` (default 42, R's `minDataPointsPerStratum`)
  raise an error, `lts_quantile` must lie between 0.5 and 1, `factr` and
  `pgtol` must be 0 or more, `maxit` and `cvg_niter` must be integers of 1
  or more, and a likelihood that is not finite (infinite values, identical
  arrays) raises "L-BFGS-B needs finite values of 'fn'". Each of these
  returned numbers; `cvg_niter=0` returned the data transformed at the
  starting values, never fitted.
- The lowess fits accept a span above 1, as R does (all points), where
  statsmodels raised "Lowess `frac` must be in the range [0,1]!", and raise
  R's errors for `voom(span=0)` ("'f' must be finite and > 0") and
  `loess_fit(iterations=0)` ("'iter' must be finite and >= 0"); before,
  `iterations` below 1 returned uninitialised memory. A voom-family trend
  with fewer than two distinct points (one gene with a finite sigma, or all
  genes at one average) stops with R's "need at least two non-NA values to
  interpolate" where pylimma returned a constant trend.
- `normalize_cyclic_loess` returns rows that hold an infinite value as
  NaN, as R does, and warns; before, such a row was left unnormalised.
- `normalize_cyclic_loess` checks its input: `iterations` below 1 raises
  (R's `1:iterations` runs two passes for 0; pylimma returned x
  unnormalised), as do `"pairs"` / `"affy"` on a single column and `"fast"`
  on none (R: "subscript out of bounds"; pylimma returned x unchanged).
  Weights with more than one value per row give R's "y and weights have
  different lengths" (pylimma raised an unrelated numpy error); `loess_fit`,
  like R's `loessFit`, now accepts weights of any shape with one value per
  point, such as an n x 1 matrix.
- `plot_md` / `plot_ma` take R's defaults. For a fit they plot the last
  coefficient (they plotted the first, usually the intercept), titled with
  its name; `status` defaults to the `Status` column of `fit["genes"]` (or
  of an EList's genes), and `plot_md` reduces a status with one column per
  coefficient (`decide_tests` output) to the plotted column. For a matrix or
  EList the title is the column name and `ylab` is "Expression log-ratio
  (this sample vs others)"; columns can be given by name.
- `volcano_plot` labels the highlighted genes with the `ID` column of
  `fit["genes"]` by default, as R does; it used row numbers.
- `plot_mds` gains R's `plot` argument: `plot=False` returns the MDS result
  (coordinates for `dim_plot`, eigenvalues and -vectors, variance
  explained, distance matrix) without drawing, and that result can be
  passed back to plot it for other dimensions, as R's `plotMDS.MDS` does
  (`plot=False` used to be swallowed and the plot drawn). `dim_plot` must
  name two different dimensions and `top` must be 1 or more (`np.inf`
  for all genes), as in R (`dim_plot=(0, 0)` ran, `top=0` failed inside
  the eigendecomposition), and a degenerate dimension warns as in R, with
  R's 1-based dimension number.
- `volcano_plot` on an object without coefficients raises R's "fit must be
  an MArrayLM" (it raised `KeyError`), and `mdplot` with fewer than two
  columns and `venn_diagram(None)` raise clear errors where R fails
  obscurely (see known differences).
- The plot functions raise a clear error for an integer `pch` (R's codes
  such as 16 are not translated to matplotlib markers; see known
  differences), where matplotlib raised "Unrecognized marker style 16".
- `plot_rldf` follows R's `plotRLDF`: the design's rank comes from R's
  `qr()` rule (a design column on a tiny scale was dropped by an absolute
  1e-10 cutoff), R's argument checks apply (`labels_y` and `labels_z`
  lengths, rows of `z`, `show_dimensions`, `var_prior` / `df_prior`
  lengths), points are labelled with the column names or 1..n unless
  `pch_y` is given, and a design with fewer discriminant functions than
  asked for (two groups) raises a clear error where R pads an arbitrary
  dimension (it raised `IndexError`).
- `coolmap` draws and returns what R's `coolmap` (through
  `gplots::heatmap.2`) does: the rows from the bottom up in dendrogram
  order (they were upside down relative to R), the row and column
  dendrograms as `show_dendrogram` asks (none were drawn) and the colour
  key, and it returns a dict with the figure, the heatmap axes, `row_ind`
  / `col_ind` (R's `rowInd` / `colInd`, 0-based) and the linkages, where it
  returned the Figure. `ax=` and further arguments (which R passes to
  `heatmap.2`) raise errors; they were silently ignored.
- `barcode_plot` takes `index` / `index2` as pylimma's gene-set functions
  do: booleans, 0-based positions or names of a `statistics` Series. It
  guessed the base of integer positions (all between 1 and n were taken as
  1-based and shifted), so a 0-based index could mark the wrong genes, and
  `gene_weights` followed the position order instead of the order of
  `index`; an index with NA raises R's error.

### Added

- `normalize_vsn(cvg_eps=)`, vsn's `optimpar$cvg.eps`: the robust (LTS)
  iterations stop once the transformed data have changed by less than
  `cvg_eps` three times in a row. The default 0 runs all `cvg_niter`
  iterations, as before and as in R.
- `layer=` for `plot_md`, `plot_ma`, `plot_ma_3by2`, `plot_mds`,
  `plot_densities`, `coolmap` and `plot_rldf`: an AnnData's layer is
  plotted in place of `X`, as in `lm_fit`.

### Fixed

- `duplicate_correlation` takes a `block` given as a list of strings (it
  raised `AttributeError`) and a design given as a vector, which is one
  column as R's `as.matrix` (it raised `IndexError`); a design whose rows
  do not match the arrays raises R's error, and blocks all of size 1 are
  checked before the block-in-design check, as in R.
- `au_roc` takes a matrix `truth` (read column by column, as R's
  `as.vector`; it raised `IndexError`), a logical or character `stat`
  (ranked as R's `order()`; they raised `TypeError` / `ValueError`) and
  R's logical strings ("TRUE", "F", ...) as `truth`; another string as
  `truth` raises a clear error (see known differences).
- A fit from `lm_fit` on an AnnData or a DataFrame with row names keeps
  those names as its row names (`fit.dimnames`), as R's `rownames(fit)`,
  so `fit[["g3", "g1"], :]` selects genes by name; it raised "invalid
  literal for int()".
- `prop_true_null` takes a matrix of p-values as its values, as R does;
  methods `"lfdr"`, `"mean"` and `"convest"` raised a broadcasting error.
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
- `goana` / `kegga` on a fit raise "defines its own universe" for
  `universe=`, as R (it was silently ignored), and "defines its own
  covariate" for `covariate=` (R fails with an argument collision).
- `goana` / `kegga` accept a numpy boolean `trend` (`np.False_` raised
  `NotImplementedError`).
- `kegga` takes the first row of a pathway ID that appears more than once
  in `pathway_names` (R's `match`; it took the last), and a row with a
  missing name is dropped, so that pathway's name is NA (it was the text
  "nan").
- `top_go` / `top_kegg` sort a missing term or pathway name last among
  ties, as R's `order`; they raised `TypeError`.
- `top_go` takes `ontology` as one string and accepts abbreviations, as
  R's `match.arg` (a string was iterated by character and raised); an
  element that matches nothing still raises. `top_go` / `top_kegg` cap
  `number` at the row count, so `number=np.inf` returns every row (it
  raised `OverflowError`), and `number=NaN` raises a clear error, as in
  `top_table`.
- `goana` / `kegga` raise an error when two set names give the same
  lower-case output column (for example "A" and "a", or a set named "N"
  next to `n`); one of the sets was silently dropped.
- `goana_trend(plot=True)` draws R's trend plot (it was silently ignored),
  and `goana_trend` gains R's `xlab`, `ylab` and `main`.
- `squeeze_var(covariate=...)`, and so `e_bayes(trend=True)` and
  `treat(trend=True)`, fit the variance trend with an economy-size QR. The
  full QR built a genes x genes matrix: 8,000 genes needed 1.2 GB and
  22,000 genes over 10 GB; 30,000 genes now take 0.02 s and 170 MB.
  Results are unchanged.
- `zscore_t` (default quantile method) returns R's z-scores for very large
  |t|: it returned Inf once the t tail probability underflowed (for
  example |t| = 100 at 988 df, where R gives 48.77).
- `squeeze_var`, `fit_f_dist`, `fit_f_dist_robustly` and
  `fit_f_dist_unequal_df1` take a single variance as a length-1 vector, as
  R does; a Python scalar raised `TypeError`.
- `make_contrasts` accepts every level name that R accepts (any name equal
  to its `make.names()`), including dotted names such as `WT.Ctrl` (as in
  the limma User's Guide) and Python keywords such as `not` or `True`,
  which raised errors or, for `None`/`True`, evaluated to Python constants.
  `^` is a power as in R (it was Python's XOR), and R number forms such as
  `2L` and `0x1F` work.
- `make_contrasts(contrasts=...)` with a dict evaluated the keys as
  expressions; as R's named character vector, the keys of a dict or the
  index of a Series (other than a default RangeIndex) are the column names
  (an empty name falls back to the expression). A single string is one contrast, not one per character.
- `normalize_between_arrays(method="vsn")` says, as R does, that the
  method is no longer supported and to use `normalize_vsn` instead; it
  said the method was not applicable to single-channel data.
- `normalize_cyclic_loess` (so also `normalize_between_arrays(method=
  "cyclicloess")`) takes each fit from `loess_fit`, as R does. Rows with a
  zero weight were left unadjusted (up to 2.16 off R on a 60-row example)
  and a column with one observed value in a row was not fitted, where R
  fits it exactly (two genes with missing values: 6.58 against R's 6.00).
- `normalize_cyclic_loess(adaptive_span=True)` chooses its span as R does,
  with `choose_lowess_span(nrow, small_n=200, min_span=0.6)`; it used
  voom's settings (`small_n=50, min_span=0.3`), a span of 0.69 instead of
  0.95 for 300 rows (results up to 1.02 off R).
- `plot_with_highlights` (so also `plot_md` / `plot_ma` with a status)
  reads the status as R does: a missing value (None, NaN) is background, not
  a highlighted group of its own, and is no longer counted when choosing the
  most frequent value as background; a float status matches integer
  `values` (`1.0` is `"1"`), where nothing was highlighted; tied
  frequencies keep R's level order (numeric for numbers).
- `plot_with_highlights` (so also `plot_md` / `plot_ma`) draws limma's
  legend: the background value first ("Other" when the background holds
  several values), then every highlighted value, also one with no points.
  The legend listed only the highlighted values that had points.
- `venn_counts` leaves out rows with a missing value, as R's `vennCounts`
  does; such a row was counted as all zeros (with any `include`). Unnamed
  columns are named "Group 1", "Group 2", ... as in R (they were "col0",
  "col1", ...), which also gives `venn_diagram` R's default set names.
- `venn_diagram` places the counts as R's `vennDiagram` does: with two
  sets the counts for "first set only" and "second set only" were drawn in
  each other's circles, and with one set the counts inside and outside the
  circle were swapped. A scalar `cex` works (it raised `TypeError`), and a
  given `circle_col` also fills the circles lightly, as in R.
- `heat_diagram` labels rows by their position in the full matrix, as R
  does (it renumbered the significant rows 1..k, so the labels did not
  identify the genes), and orders rows on the coefficients before they are
  truncated at `limit`, with a stable sort, so truncated values keep R's
  order. Names are R's `as.character` (`1.0` is "1"), and `names` must
  give one name per row (R labels the rows past its end NA).
- `plot_densities` orders the groups as R's factor does (numbers
  numerically, otherwise sorted), so the legend and the colour of each
  group follow R; a `group` shorter than the number of arrays is recycled
  over the arrays as in R, where it raised `IndexError`. A missing group
  (None, pandas NA, or NaN among strings) gets no line, as R's NA; a NaN
  among numbers is a group of its own, as in R.
- `plotlines` recycles a list of colours over the rows, as R does; a list
  whose length was not the number of rows raised `ValueError`.
- Without matplotlib, `plotlines`, `heat_diagram`, `plot_exons`,
  `plot_exon_junc`, `plot_ma_3by2` and `plot_rldf(plot=True)` raise the
  same `ImportError` as the other plot functions, saying to install
  `pylimma[plot]`; they raised a bare `ModuleNotFoundError`.
- `plot_md`, `plot_ma`, `plot_ma_3by2`, `plot_mds` and `plot_densities`
  accept an AnnData (read through `get_eawp`, so samples x genes is turned
  to genes x samples and `obs_names` name the samples), as the other
  functions that take expression data do; they raised an obscure
  `ValueError`. `plot_densities(log=False)` plots `2 ** E` of an AnnData,
  as of an EList.
- `plot_exons` draws what R's `plotExons` draws: `rank` counts distinct
  genes (`rank=2` plotted the first gene again when it had several exons),
  exons are ordered and labelled by `exoncolname` (which was ignored) and
  joined by a line, DE exons are marked red or blue by the sign of the
  logFC and sized by significance, a dashed line marks the mean logFC (it
  was a line at 0) and the title lists the gene's annotation and strand. A
  missing `exoncolname` raises R's error, and `coef` takes a coefficient
  name, as in R.
- `coolmap` clusters a matrix with missing values as R's `dist()` does
  (over the coordinates two rows share, scaled up by the number missing);
  it raised "The condensed distance matrix must contain only finite
  values".
- `coolmap` accepts an EList (its `E`, as R's `as.matrix`) and an AnnData,
  where an EList raised `TypeError`, and a single row raises R's "Need at
  least 2 rows and 2 columns" instead of scipy's error.
- `barcode_plot` draws R's barcode plot: the bars for each gene of the set
  (below the band for a second set or negative weights, with heights and
  transparency for unequal weights), the shaded band (light blue / pink
  beyond the quantiles), the second worm and both reference lines, and R's
  labels (quantiles of the statistics along x, enrichment and weight scales,
  "Down" / "Up"). It drew only the first worm. `main` sets the title; other
  extra arguments raise `TypeError` (they were ignored).

### Documentation and examples

- `model_matrix` and `known_differences.rst` document how formula designs
  built with patsy differ from R's `model.matrix` in column names and
  order (interactions of numeric and categorical terms, `~ g:h`).

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
