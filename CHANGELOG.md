# Changelog

All notable changes to pylimma. pylimma is a port of R limma, and its
parity target is limma 3.66.0 (Bioconductor 3.22, R 4.5). Before 1.0, a
minor version (0.x.0) may change results or behaviour; a patch version
(0.x.y) does not.

## [Unreleased]

### Changed results

- `vooma_lm_fit` leaves a gene with no residual degrees of freedom (NaN
  `sigma`, from missing values) out of its mean-variance trends, as
  `vooma` now does. It followed R's `lowess`, which computes with the NaN
  and so shifts the trend and every other gene's weights
  (`known_differences.rst`).
- `classify_tests_f` (and `decide_tests(method="nestedF")`) takes its
  thresholds from R's `qf` rules, which use the chi-squared limit for a
  residual df above 4e5; scipy's exact F quantile was up to about 5e-6
  higher, so a gene just above R's threshold was not classified.
- `zscore(q, "hyper", m, n, k)` maps the plain hypergeometric tail
  probabilities, as R's `zscore`; it gave the mid-p values of
  `zscore_hyper` (R's `zscoreHyper`), which differ.
- `normalize_quantiles` (and `normalize_between_arrays(method="quantile")`)
  interpolates as R's `approx()`, so a value interpolated next to an
  infinite one is NaN, as in R (tied `-Inf` values, or a column with
  missing values and `-Inf`); `np.interp` gave the infinity.
- `diff_splice` orders genes and exons as R's `order()`: numeric IDs
  numerically (gene 12 after gene 2), where it sorted their text (12
  before 2); with missing gene IDs the IDs become text, as in R.
- `avereps` forms its groups as R does: IDs are compared as R's
  `as.character` (1 and 1.0 are one probe), missing IDs form one group
  with R's "missing values for 'group'" warning (each was a row of NaN of
  its own), and an ID of the wrong length raises R's "incorrect length
  for 'group'" (it raised `IndexError`). Grouping no longer loops over the
  IDs, so large inputs average faster.
- `squeeze_var(robust=True)` with unequal residual df takes its F
  quantiles from R's `qf` rules, including the chi-squared limits for a
  prior df above 4e5, where scipy's exact quantile was about 1e-6 from R.
- Selecting columns of a rank-deficient fit (`fit[:, ["C", "A"]]`,
  `contrasts_fit(coefficients=...)`, `genas(coef=...)`) subsets
  `cov_coefficients` in the order of the selected columns. As R, it kept
  pivot order, so the covariance did not line up with the coefficients
  for columns out of that order (see known differences).
- `duplicate_correlation` keeps a gene's correlation whatever the sign of
  its variance components, clipped to R's bounds, as R does; genes whose
  components summed to a non-positive value were left out of the
  consensus.
- `diff_splice` gives R's NaN gene variance (`gene_s2`) for a gene with
  no residual degrees of freedom (R's 0/0), where it gave 0. As in R,
  `squeeze_var` gives such a gene the prior variance, so its moderated
  statistics are unchanged unless too few genes are left to estimate the
  prior.
- `normexp_fit(method="mle")` (and `background_correct(method="normexp",
  normexp_method="mle")`) refines with R's `nlminb` parameter tolerance
  (1.5e-8); on Illumina-like arrays with a flat likelihood scipy's default
  stopped early, up to 2e-3 from R's parameters.
- `normexp_signal` is exact to rounding when the normal part is narrow
  next to the exponential mean (large `sigma^2 / alpha`), where the
  signal is a small difference of large terms; it was off by up to 9%
  (R's own value is off by up to ~3% there; see known differences).
- `background_correct` on an `EList` with an `Eb` slot corrects with it,
  as R's `backgroundCorrect` does for an EListRaw, and drops it from the
  result; it ignored the slot, so `method="auto"` ran normexp and
  `method="subtract"` returned the foreground unchanged.
- `genas` fits its likelihoods with R's `optim()` Nelder-Mead settings
  (initial simplex, stopping rule, 500 evaluations). With a start of 0
  for the correlation parameter, scipy's default simplex barely moved it
  and the search stopped at the null fit: a strongly correlated data set
  gave p = 0.99999981 where R gives 9.2e-12. The deviance and p-value now
  agree with R to about 2e-6 (see known differences).
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
  names raise a `ValueError` asking for `design=`. A design saved by
  pylimma 0.2.0 or earlier (e.g. read back from an `.h5ad` file) has no
  sample names, so a reorder cannot be detected; it also raises, and
  `design=` must be passed or `voom` / `vooma` re-run. These errors come
  only from functions that use the stored design, and not when `design=`
  is passed.
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
- `e_bayes` with unequal total df (genes with missing values, or
  `robust=True`) converts t-statistics to the largest df as R does for
  very large |t| (beyond about 58 at
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

- `lm_fit` with `ndups > 1` and `block` raises a clear error. R fits every
  probe separately when `block` is given but reduces the gene annotation to
  one row per gene, so the annotation was attached to the wrong rows
  (`top_table` on an EList failed with `IndexError`).
- `top_table` on a `treat()` fit with the default `sort_by="B"` raises
  R's "Trying to sort.by B, but B-statistic (lods) not found" (a treat
  fit has no B-statistic); it sorted by p-value instead. Use
  `top_treat`, which sorts by p-value, or pass `sort_by="p"`.
- `top_table` (and `top_table_f`, `top_treat`) of a fit without row
  names numbers the rows 1..N, as R; without a genelist they were named
  `gene1`..`geneN`, and with a DataFrame genelist after its index.
- `lm_fit` with a design without column names (a matrix, a DataFrame
  with pandas' default columns, or no design) names the
  coefficients `x1`..`xp`, as R's `lmFit` does for least squares and the
  usual GLS path (R leaves them unnamed for `method="robust"` and for GLS
  with missing values or probe weights; pylimma names them for every
  method, see `known_differences.rst`); they had no names, so `top_table(fit, coef="x2")` failed, the F-test
  table named its columns `Coef1`.., and `as_dataframe()` numbered them.
  Contrasts written for such a design use these names
  (`make_contrasts("x2-x1", levels=["x1", "x2"])`); other row names give
  R's "row names of contrasts don't match" warning.
- `vooma_by_group` returns R's `voom_xy` (each group's trend data, one
  column per group level) and `span`, names its default design
  `group<level>` as R's `model.matrix(~0+group)`, and keeps the column
  names of a DataFrame design (it returned a bare matrix).
- `voom`, `voom_with_quality_weights`, `vooma` and `vooma_lm_fit` name
  their default intercept-only design "GrandMean", as R, so the design is
  a DataFrame and a fit of it has the coefficient "GrandMean"; it was an
  unnamed matrix.
- `vooma_lm_fit` returns its output as R's `voomaLmFit`: with sample
  weights, `targets` is the input's targets (or a new DataFrame named by
  the samples) with a `sample_weight` column, where it was a dict
  `{"sample_weights": ...}`, and the saved plot's `xlab` is "Average log2
  expression" or "Combined predictor", where it was always "Average
  log-expression". AnnData input still stores `sample_weights` in
  `adata.uns`.
- `zscore` takes a distribution's parameters with R's names, order and
  defaults (`df1`/`df2` for "f", `shape1`/`shape2` for "beta", `rate` for
  "exp", `size`/`prob` for "binom", positional arguments as R); it passed
  them to scipy under scipy's names, contrary to its docstring. It also
  supports R's nbinom, geom, unif, logis, cauchy, lnorm and weibull and
  the noncentral t, F and chi-squared (`ncp`).
- `top_table` with a `genelist` puts its columns first, as R (they came
  after the statistics for a single coefficient), and names the rows
  after the fit's gene names, as R: an explicit genelist replaced them
  with its own index or 1..N, and duplicated fit names were not moved to
  an `ID` / `ID0` column. A Series genelist is a vector, as R (it was
  silently dropped), and a vector genelist of the wrong length raises.
- A pandas DataFrame input gives a DataFrame result with its index and
  columns, as R keeps dimnames, where pylimma returned a bare ndarray:
  `normalize_quantiles`, `normalize_median_values`,
  `normalize_cyclic_loess`, `normalize_between_arrays`,
  `background_correct`, `remove_batch_effect`, the `E` of `voom`, `vooma`
  and `voom_with_quality_weights`, `ave_dups` (rows named after the first
  spot of each duplicate set) and `aver_arrays` (the unique IDs as
  columns). ndarray input still gives an ndarray.
- `fitmixture` raises a clear error for one array or one `mixprop` value,
  where it returned a minimum-norm fit (R fails), and takes a whole-number
  float `niter` (it raised `TypeError`; a fraction raises).
- `genas` raises clear errors for a `coef` that is out of range, negative,
  fractional or repeated, and for two coefficients with a singular
  covariance matrix (identical contrasts); it returned results (R fails in
  `fit[, coef]` or `chol()`; see known differences).
- `decide_tests(p_value=NaN)` and `classify_tests_f` with a `p_value`
  outside 0..1, a `df` of 0 or NaN or an unusable `cor_matrix` raise clear
  errors; they silently returned all zeros (R gives NA or fails obscurely;
  see known differences). `decide_tests` takes a single p-value as a 1 x 1
  matrix, as R's `as.matrix` does (it raised `IndexError`), and
  `summarize_test_results` checks its input and `coef_names`.
- Formula designs (`model_matrix`, `lm_fit("~ ...")`, the voom family)
  raise R's "contrasts can be applied only to factors with 2 or more
  levels" for a factor with one level, where patsy gave it a constant
  column, and a clear error for a formula that is not a string (it raised
  `AttributeError`).
- `MArrayLM.as_dataframe` names columns as R's `as.data.frame.MArrayLM`:
  `coefficients.<coefficient name>` (it numbered them,
  `coefficients.1`), and `genes.<column>` for a genes data frame.
- `lm_fit(method="robust")` / `mrlm` raise a clear error for `k <= 0`; `k=0`
  returned zero coefficients and a negative `k` a LAPACK error (R fails
  obscurely; see known differences).
- `lm_fit` raises R's "NAs not allowed in design matrix" (scipy raised "array
  must not contain infs or NaNs") and a clear error for a matrix with no
  samples (it returned an all-NaN fit); `is_fullrank` raises for missing or
  infinite values, where it returned `False` (see known differences).
- `lm_fit`, `gls_series`, `voom` and `duplicate_correlation` raise a clear
  error for a missing `block` label (NaN, None); a NaN label became a block
  of its own and `None` raised `TypeError` (R fails obscurely; see known
  differences).
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
  polynomial ones. This includes `pd.cut()` columns, which pandas makes
  ordered: `.cat.as_unordered()` gives the treatment coding of R's `cut()`,
  as 0.2.0 did.
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

- `kegga` names pathways whose IDs are numbers in `pathway_names` (e.g. a
  column read by `read_csv` with a gap, so 10.0): IDs are compared as R's
  `as.character()` gives them ("10"), as R's `match()` does; they were left
  unnamed.
- `lm_fit`'s "Coefficients not estimable" warning separates the names
  with spaces, as R's message (it used commas); R prints the message
  rather than warning, which is now recorded in `known_differences.rst`.
- `gls_series(spacing=None)` reads the spacing as 1, as R's
  `gls.series`; it raised "spacing must be a whole number". With the
  correlation left to be estimated R fails on the NULL; pylimma reads it
  as 1 there too (documented in `known_differences.rst`).
- `treat` removes the `lods` slot, as R's `fit$lods <- NULL`; it set it
  to None, so `"lods" in fit` was true.
- `top_table` and `top_table_f` read a fit whose coefficients (and, for
  `top_table`, t, p-values or lods) are a vector as one coefficient, as
  R's `as.matrix`; they raised `ValueError` / `IndexError`.
- `top_table(adjust_method=None)` (and `top_treat`) adjusts by "BH", as
  R's `topTable`; it raised `AttributeError`.
- Functions that take a fit (`e_bayes`, `treat`, `contrasts_fit`,
  `top_table`, `diff_splice` and others) raise a clear `TypeError` for
  input that is not a fit, as R's "fit is not a valid MArrayLM object"
  (they raised obscure errors from `dict()`); `e_bayes` and `treat` raise
  a clear error when `stdev_unscaled` and `coefficients` differ in shape
  (R: "non-conformable arrays"; pylimma broadcast them and raised
  `IndexError`), read a vector of either as one column, as R (it raised
  `IndexError`), read integer coefficients as numbers (`treat`
  truncated its t-statistics) and name the slots a fit lacks; `top_table_f` and
  `top_table` without coefficients raise R's "Coefficients not found in
  fit" and "coefficients not found in fit object" (they raised
  `KeyError`, or failed later for coefficients set to None).
- `e_bayes(stdev_coef_lim=...)` of a length other than 2 raises a clear
  error, as R's "v0.lim must have length 2"; three values used the first
  two and one value raised `IndexError`.
- `contrasts_fit` with a DataFrame contrast that has pandas' default
  0..n-1 index or columns treats it as having no row or column names, as
  R does a matrix without them; it warned "row names of contrasts don't
  match col names of coefficients" and named the contrasts 0, 1, ...
- The EList or dict returned by a shape-preserving function holds copies
  of the input's other slots (weights, genes, targets ...), and `voom`'s
  `lib_size` is a copy of the one given, as R's copy-on-modify; they were
  the caller's objects.
- `classify_tests_f` with one coefficient compares the p-values with any
  `p_value`, as R, whose one-coefficient branch never reaches `qf`; this
  group's new check raised for a `p_value` outside 0..1 there too.
- A one-element array, list or Series is accepted where a single number
  is expected, as R's length-1 vectors: `decide_tests` and
  `classify_tests_f` `p_value`,
  `background_correct` `offset`, `array_weights` `prior_n` and `lm_fit`
  robust `k` (the new argument checks rejected them).
- A fit rebuilt by hand from a reloaded h5ad (`MArrayLM(adata.uns[key])`)
  with contrasts can be subset by column name and flattened with
  `as_dataframe`; the contrast names came back as an array, which raised
  "truth value of an array is ambiguous".
- `select_model` with a 1-D `y` fits it as one gene, as `lm_fit` does
  (it raised an IndexError); the result equals R's
  `selectModel(matrix(y, 1), ...)` (see known differences).
- `modify_weights` defaults `weights` to 1 for each `status`, as R's
  `modifyWeights` (`weights` was required, and None raised an
  IndexError); a missing `status`, `values` or `multipliers` raises R's
  "argument is missing" error.
- `array_weights` requires `maxiter` to be a whole number of at least 1
  and `prior_n` a finite number of 0 or more; `prior_n=Inf` gave NaN
  weights or scipy's error and `maxiter=0` no iteration (R's `1:0`
  iterates).
- `array_weights` stops as R's scoring iteration does: a singular
  information matrix (for example `prior_n=0` with 2 residual df) raises a
  clear error, where pylimma skipped the step and could return weights of
  1, and REML warns "iteration limit reached" or "convergence tolerance
  not achievable, stopping prematurely" when it stops unconverged, as R.
- `vooma` accepts missing values in `y`, as limma intends (NEWS 3.64.0)
  although R's `lm.fit` fails on them: each gene is fitted on its
  observed values, as R's `voomaLmFit` does (`known_differences.rst`). It
  returned NaN weights for every gene with a missing value. Infinite
  values raise a clear error (it failed in lowess). `vooma_by_group`, which
  calls `vooma` for each group, accepts them too, as long as no gene is
  entirely missing within a group.
- Clear errors for invalid shapes where R fails obscurely or recycles:
  rows that are not a multiple of `ndups * spacing` (`lm_fit`,
  `unwrap_dups`, `ave_dups`; numpy's reshape error, R's "dims do not
  match the length of object"), a design with the wrong number of rows
  for `array_weights` (numpy's matmul error, R's "incompatible
  dimensions") and a `lib_size` of the wrong length for `voom` (numpy's
  broadcast error; R recycles it).
- `voom`, `vooma`, `voom_with_quality_weights` and `convest` with
  `plot=True` raise the same clear ImportError as the plotting functions
  when matplotlib is not installed; voom warned and drew nothing, and
  convest drew nothing silently.
- `prop_true_null` raises R's convest error "Missing values in p not
  allowed" for missing p-values with `method="lfdr"` or `"mean"` (it
  returned NaN; R recycles into an arbitrary number, see known
  differences), and `nbins` must be a whole number of at least 1 (0 and
  fractions raised numpy errors).
- `choose_lowess_span` takes an array `n` and returns a span for each
  value, as R's `pmin`, and gives NaN for a negative `n`, as R; an array
  raised numpy's "truth value is ambiguous" error and a negative `n` a
  TypeError.
- `logsumexp(x, y)` with `x` and `y` of different lengths (neither a
  single value) raises a clear error; it raised numpy's broadcast error
  (R recycles the shorter, see known differences).
- `make_unique` compares values as R's `as.character` (1 and 1.0 are the
  same value) and keeps missing values missing, as R's NA; they became
  the text "None" or "nan" and were numbered.
- `is_numeric` is False for categorical values (R's factors), as for
  logical and complex values, as R's `is.numeric`; a categorical with
  numeric categories was reported numeric.
- `background_correct` raises R's "non-conformable arrays" for a
  `background` of another shape (numpy broadcast a column, row or scalar,
  or gave its own error), a clear error for `method="minimum"` when an
  array has no value above background (it raised numpy's "zero-size
  array" error; R gives Inf) and for a NaN or vector `offset` (a NaN
  offset returned all NaN).
- Outputs no longer share memory with their inputs, as R's
  copy-on-modify: `lm_fit`'s `design` slot, the `E` of `vooma` and
  `vooma_by_group` (every expression matrix returned through `put_eawp`
  is now a copy), the voom family's `design`, `vooma_lm_fit`'s kept
  `EList`, every slot of the fits returned by `e_bayes`, `treat` and
  `contrasts_fit`, and the inputs that `normalize_median_values` (one
  column), `ave_dups` and `unwrap_dups` (`ndups=1`) returned unchanged.
  Editing one in place changed the other.
- `select_model(criterion=None)` uses "aic", as R's `match.arg(NULL)` picks
  the first choice (it raised `AttributeError`), and an empty
  `design_list` raises a clear error.
- `as_matrix_weights(None, dim)` raises a clear error; it returned a matrix
  of NaN weights (R fails in `as.matrix(NULL)`).
- `wsva` raises R's "design matrix not specified" and "row dimension of
  design matrix must match column dimension of data" (it raised
  `IndexError` or a matrix-multiplication error). As R, an EList's design
  is not used (R's `as.matrix(y)` drops it).
- `get_eawp` (and so every function taking expression data) reads input as
  R's `getEAWP` does: an EList's vector `E` is one column, a bare vector
  one row; complex values raise R's "Data object doesn't contain numeric
  expression values" (the imaginary part was dropped with a warning); an
  `E` of `None` and an AnnData whose `X` is `None` raise clear errors
  (they raised `AxisError`); an all-NA gene no longer warns "Mean of empty
  slice" (its `Amean` is NaN, as in R).
- `array_weights` takes a design given as a vector, as one column (R's
  `as.matrix`); it raised "not enough values to unpack".
- `genas` takes `coef` as coefficient names (`coef=("A", "B")`), as R's
  `fit[, coef]` does; it raised "invalid literal for int()".
- `genas` with a `subset` rule keeps the selected genes as R's
  `fit[genes, ]` does: on a fit from AnnData it raised `TypeError`, and on
  a fit with two genes it also sliced `cov_coefficients` (of the same
  size) and failed with "cor_matrix does not match the number of tests".
- `lm_fit(method="robust")` and `mrlm` take a whole-number float
  `maxit` (10.0), as R does; it raised `TypeError`. A fraction raises
  "maxit must be a whole number" (see known differences).
- `squeeze_var` takes a matrix of variances as its values, read column by
  column as R's `length(var)` does (it raised a broadcasting error),
  raises R's "var is empty" for `None` (it returned a result) and R's "x
  and df1 have different lengths" for a `df` of the wrong length (it
  raised `IndexError`); a single `df` of 0 no longer sets every variance
  to 0, which R does only for a `df` vector.
- `e_bayes(trend=np.True_)` treats a numpy logical as logical (it raised
  `TypeError`); `e_bayes` and `treat` on a fit with no genes raise R's
  "No finite residual standard deviations" (numpy's "zero-size array"
  error before); `treat(fc=None)` without `lfc` raises a clear error
  (R: "non-numeric argument to mathematical function").
- `top_table` takes a float `coef` (1.0) or an array of coefficients, as
  R truncates a double subscript (it raised `IndexError`), and raises a
  clear error for an empty `coef`, where it raised `IndexError` (R returns
  an empty table; see known differences).
- `contrasts_fit` takes a float scalar `coefficients` (1.0), as R does
  (it raised `TypeError`), raises R's "fit must contain coefficients
  component" for a missing fit (it raised `AttributeError`), and raises a
  clear error for a `cov_coefficients` whose size does not match the
  coefficients, where it returned a result silently (see known
  differences).
- `avereps` on an `EList` averages every matrix in `other` and weights
  given as a vector (as one column, R's `as.matrix`), as R does; they were
  left with one row per probe while `E` was averaged. Weights or `other`
  matrices whose rows do not match the probes (such as per-sample array
  weights) raise R's "incorrect length for 'group'"; they were left out.
- `unwrap_dups`, `ave_dups`, `unique_genelist` and `duplicate_correlation`
  take whole-number floats for `ndups` and `spacing` (2.0), as R does;
  they raised `TypeError`. A fraction raises "ndups must be a whole
  number" (R fails in `dim<-`).
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
  `ZeroDivisionError`; as in R, that set's row has NaN p-values. With
  `use_ranks=True` such a set also has NaN p-values, where R gives a
  p-value of 2 (`known_differences.rst`), and the FDR is adjusted over the
  other sets.
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

- `known_differences.rst` and the `fry` docstring document that with
  observation weights `fry`'s p-values can differ from R's for designs
  where the residual basis is set by the sign of a rounding-level number
  (e.g. `~0 + group` with a contrast between two groups); both bases are
  valid. A fixture checks the contrast effects and residual sums of
  squares, which do not depend on the basis, against R. For a contrast
  vector, `fry`, `roast` and `mroast` now reform the design with
  `contrast_as_coef`, as R's `.lmEffects` calls `contrastAsCoef` (the
  contrast column was not scaled; only rounding changes).
- `known_differences.rst` records the df2 root of
  `fit_f_dist_robustly` (scipy's `brentq` against R's `uniroot`, up to
  1.3e-8 relative), the F-test table's unmangled coefficient names, the
  gene names `lm_fit(ndups=2)` keeps, the `x1`..`xp` names on every
  fitting path, the one-column results of `e_bayes` for a vector fit and
  `lm_series` being an internal helper without R's `design=NULL`,
  `ndups` and `spacing`.
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
