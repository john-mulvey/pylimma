# pylimma

A Python port of [R limma](https://bioconductor.org/packages/limma/) for differential expression and abundance analysis.

[![PyPI version](https://img.shields.io/pypi/v/pylimma)](https://pypi.org/project/pylimma/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
<!-- [![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](https://doi.org/PREPRINT_DOI_PENDING) -->

## Scope

pylimma is a comprehensive port of limma, with the exception of some microarray specific functionality that we do not anticipate is commonly used.

pylimma covers linear modelling with empirical Bayes moderation (`lm_fit`, `contrasts_fit`, `e_bayes`, `treat`, `top_table`), voom and RNA-seq precision weights, normalisation and background correction, batch correction and surrogate variables, duplicate correlation and array weights, competitive and self-contained gene-set testing (camera, roast, mroast, fry, romer, geneSetTest), differential splicing, diagnostics plots, and the supporting statistical utilities.

It accepts three input idioms for the same analysis:

- **Flat array-like objects** - a NumPy `ndarray` or a pandas `DataFrame` of genes x samples
- **`EList`** - a Python port of R limma's `EList` class, implemented as a `dict` subclass in `pylimma.classes` with slots `E`, `weights`, `genes`, `targets`, `design`
- **`AnnData`** - read from `adata.X` or `adata.layers[layer]`, with shape-preserving outputs written back to `adata.layers[...]` and fits stored in `adata.uns[...]`

Input and output dispatch lives in a single place (`pylimma/classes.py`); all public functions route through it.

## Installation

```bash
pip install pylimma
pip install pylimma[plot]   # matplotlib for diagnostics
pip install pylimma[dev]    # pytest, ruff, mypy
pip install pylimma[docs]   # sphinx + nbsphinx
```

Requires Python >= 3.10.

## Quick start

The canonical limma workflow in Python mirrors the R workflow one-for-one: `lm_fit` then `contrasts_fit` then `e_bayes` then `top_table`. Parameter names and defaults follow R limma.

```python
import numpy as np
import pandas as pd
from pylimma import lm_fit, contrasts_fit, e_bayes, top_table, make_contrasts

# 100 genes x 6 samples, two groups of three, real signal in the first 10 genes
rng = np.random.default_rng(0)
expr = rng.normal(size=(100, 6))
expr[:10, 3:] += 2.0

# design matrix - intercept plus a GroupB indicator
design = pd.DataFrame(
    np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]),
    columns=["Intercept", "GroupB"],
)

# contrasts - test GroupB against the intercept; equivalent to R's
#   makeContrasts(GroupBvsA = GroupB, levels = design)
contrasts = make_contrasts(GroupBvsA="GroupB", levels=design)

fit = lm_fit(expr, design)
fit = contrasts_fit(fit, contrasts=contrasts)
fit = e_bayes(fit)
print(top_table(fit, coef="GroupBvsA").head())
```

The result is a pandas `DataFrame` with the standard limma columns (`log_fc`, `ave_expr`, `t`, `p_value`, `adj_p_value`, `B`), sorted by B-statistic.

## Data structures

pylimma offers three equivalent idioms for the same analysis - pick whichever matches your workflow.

### AnnData (scverse-native)

```python
import pylimma as pl
import anndata as ad

adata = ad.read_h5ad("expression.h5ad")   # obs x var = samples x genes
pl.voom(adata, design)                    # writes adata.layers["voom_E"]
pl.lm_fit(adata, design)                  # stores fit in adata.uns["pylimma"]
pl.e_bayes(adata)
results = pl.top_table(adata, coef=1)     # returns a DataFrame
```

### EList (R-style)

```python
from pylimma import EList, voom, lm_fit, e_bayes, top_table

el  = EList({"E": counts, "design": design})   # genes x samples
v   = voom(el)
fit = lm_fit(v, design)
fit = e_bayes(fit)
results = top_table(fit, coef=1)
```

### Flat arrays

NumPy arrays and pandas DataFrames are accepted directly (genes x samples). Fit objects are returned as `MArrayLM` (a `dict` subclass), so `fit["coefficients"]` and `fit.coefficients` both work.

## Features

- **Linear modelling**: `lm_fit`, `contrasts_fit`, `make_contrasts`, `e_bayes`, `treat`, `top_table`, `decide_tests`, `squeeze_var`.
- **RNA-seq**: `voom`, `voom_with_quality_weights`, `vooma`, `vooma_lm_fit`.
- **Normalisation**: `normalize_between_arrays` (single-channel), `normalize_quantiles`, `normalize_cyclic_loess`, `background_correct`, `normexp_fit`, `normexp_signal`, `aver_arrays`.
- **Batch and surrogate variables**: `remove_batch_effect`, `wsva`.
- **Duplicates and replication**: `duplicate_correlation`, `ave_dups`, `avereps`, `array_weights`, `array_weights_quick`.
- **Gene-set testing**: `ids2indices`, `roast`, `mroast`, `fry`, `camera`, `camera_pr`, `inter_gene_correlation`, `romer`, `gene_set_test`, `rank_sum_test_with_correlation`.
- **Splicing**: `diff_splice`, `top_splice`, `plot_splice`.
- **Plotting**: `plot_ma`, `plot_md`, `volcano_plot`, `plot_sa`, `plot_densities`, `plot_mds`, `venn_counts`, `venn_diagram`, `coolmap`, `barcode_plot`.
- **Stats utilities**: `zscore_t`, `tricube_moving_average`, `convest`, `prop_true_null`, `detection_p_values`.
- **Data classes**: `EList`, `MArrayLM` (both are `dict` subclasses, so `fit["coefficients"]` and `fit.coefficients` both work).

## Validation and R parity

The port is validated against R limma as follows:

- Every public function has a CSV fixture generated by [`tests/fixtures/generate_all_fixtures.R`](tests/fixtures/generate_all_fixtures.R) and a parity test in `tests/test_r_parity.py`.
- Comparison tolerance is `rtol=1e-6` for numeric output; p-values are compared on the log-10 scale.
- Fixture CSVs are checked into the repository. Running `pytest` does not require R installed; only regenerating the fixtures does.
- Two documented gaps - `normexp_fit(method="saddle")` (flat-plateau drift against R's `nmmin`) and rotation-test Monte Carlo p-values (`roast`, `mroast`, `romer`) - are quantified in [`docs/validation/known_differences.rst`](docs/validation/known_differences.rst). All other public functions match R output to the stated tolerance.
- Each ported module carries an SPDX header naming the R source files it ports from and the contributors listed in those files' R header comments. The full upstream attribution, including ports of algorithms from `MASS`, `statmod`, base R `splines`, and `affy`, is summarised in [`NOTICE`](NOTICE).

## Authorship and credit

pylimma is a port, not a new method. All statistical credit belongs to Gordon Smyth and the limma contributors listed in [`NOTICE`](NOTICE). The core methodology is set out in Smyth (2004), Ritchie et al. (2015), and Law et al. (2014); the full methodology citation list is below.

The Python implementation was written primarily by [Claude Code](https://www.anthropic.com/claude-code), working under my direction and review. I (John Mulvey) am named as the package author and maintainer and carry responsibility for correctness as the auditor of the port, but the large majority of the line-level Python was produced by Claude Code.

The porting workflow was as follows, and is the concrete basis on which readers should evaluate the port:

- The R limma source was read as the sole specification. Each Python module was ported from its R counterpart at the function level, preserving parameter semantics and defaults.
- Every public function has a CSV fixture generated from R limma and a parity test at `rtol=1e-6`. A function was not accepted into the library until it passed its parity test.
- Each ported module carries a per-file SPDX header naming the R source files and contributors; `NOTICE` lists the full upstream attribution.

## Related work: edgePython

I became aware of Lior Pachter's [edgePython](https://github.com/pachterlab/edgePython) - a python port of edgeR - partway through pylimma's implementation. The two projects target different upstream packages (limma for pylimma, edgeR for edgePython) and were developed in parallel, but share a methodology: an LLM-implemented port of a widely used R Bioconductor package, audited by a domain scientist.

## Citation

pylimma is a port, not a new method. If you use it in published work, please cite the original limma methodology papers:

- Smyth, G. K. (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology* 3(1), Article 3. [doi:10.2202/1544-6115.1027](https://doi.org/10.2202/1544-6115.1027)

- Ritchie, M. E., Phipson, B., Wu, D., Hu, Y., Law, C. W., Shi, W., and Smyth, G. K. (2015). limma powers differential expression analyses for RNA-sequencing and microarray studies. *Nucleic Acids Research* 43(7), e47. [doi:10.1093/nar/gkv007](https://doi.org/10.1093/nar/gkv007)

- Law, C. W., Chen, Y., Shi, W., and Smyth, G. K. (2014). voom: precision weights unlock linear model analysis tools for RNA-seq read counts. *Genome Biology* 15, R29. [doi:10.1186/gb-2014-15-2-r29](https://doi.org/10.1186/gb-2014-15-2-r29)

- Phipson, B., Lee, S., Majewski, I. J., Alexander, W. S., and Smyth, G. K. (2016). Robust hyperparameter estimation protects against hypervariable genes and improves power to detect differential expression. *Annals of Applied Statistics* 10(2), 946-963. [doi:10.1214/16-AOAS920](https://doi.org/10.1214/16-AOAS920)

- Smyth, G. K., and Speed, T. P. (2003). Normalization of cDNA microarray data. *Methods* 31, 265-273. [doi:10.1016/S1046-2023(03)00155-5](https://doi.org/10.1016/S1046-2023(03)00155-5)

- Ritchie, M. E., Silver, J., Oshlack, A., Holmes, M., Diyagama, D., Holloway, A., and Smyth, G. K. (2007). A comparison of background correction methods for two-colour microarrays. *Bioinformatics* 23(20), 2700-2707. [doi:10.1093/bioinformatics/btm412](https://doi.org/10.1093/bioinformatics/btm412)

- Silver, J. D., Ritchie, M. E., and Smyth, G. K. (2009). Microarray background correction: maximum likelihood estimation for the normal-exponential convolution. *Biostatistics* 10(2), 352-363. [doi:10.1093/biostatistics/kxn042](https://doi.org/10.1093/biostatistics/kxn042)

- Smyth, G. K., Michaud, J., and Scott, H. S. (2005). Use of within-array replicate spots for assessing differential expression in microarray experiments. *Bioinformatics* 21(9), 2067-2075. [doi:10.1093/bioinformatics/bti270](https://doi.org/10.1093/bioinformatics/bti270)

- Ritchie, M. E., Diyagama, D., Neilson, J., van Laar, R., Dobrovic, A., Holloway, A., and Smyth, G. K. (2006). Empirical array quality weights in the analysis of microarray data. *BMC Bioinformatics* 7, 261. [doi:10.1186/1471-2105-7-261](https://doi.org/10.1186/1471-2105-7-261)

- Liu, R., Holik, A. Z., Su, S., Jansz, N., Chen, K., Leong, H. S., Blewitt, M. E., Asselin-Labat, M.-L., Smyth, G. K., and Ritchie, M. E. (2015). Why weight? Modelling sample and observational level variability improves power in RNA-seq analyses. *Nucleic Acids Research* 43(15), e97. [doi:10.1093/nar/gkv412](https://doi.org/10.1093/nar/gkv412)

- McCarthy, D. J., and Smyth, G. K. (2009). Testing significance relative to a fold-change threshold is a TREAT. *Bioinformatics* 25(6), 765-771. [doi:10.1093/bioinformatics/btp053](https://doi.org/10.1093/bioinformatics/btp053)

- Wu, D., and Smyth, G. K. (2012). Camera: a competitive gene set test accounting for inter-gene correlation. *Nucleic Acids Research* 40(17), e133. [doi:10.1093/nar/gks461](https://doi.org/10.1093/nar/gks461)

- Wu, D., Lim, E., Vaillant, F., Asselin-Labat, M.-L., Visvader, J. E., and Smyth, G. K. (2010). ROAST: rotation gene set tests for complex microarray experiments. *Bioinformatics* 26(17), 2176-2182. [doi:10.1093/bioinformatics/btq401](https://doi.org/10.1093/bioinformatics/btq401)

- Majewski, I. J., Ritchie, M. E., Phipson, B., Corbin, J., Pakusch, M., Ebert, A., Busslinger, M., Koseki, H., Hu, Y., Smyth, G. K., Alexander, W. S., Hilton, D. J., and Blewitt, M. E. (2010). Opposing roles of polycomb repressive complexes in hematopoietic stem and progenitor cells. *Blood* 116(5), 731-739. [doi:10.1182/blood-2009-12-260760](https://doi.org/10.1182/blood-2009-12-260760)

- Phipson, B., and Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39. [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585)

- Baldoni, P. L., Chen, L., Li, M., Chen, Y., and Smyth, G. K. (2025). Dividing out quantification uncertainty enables assessment of differential transcript usage with limma and edgeR. *Nucleic Acids Research* 53(22), gkaf1305. [doi:10.1093/nar/gkaf1305](https://doi.org/10.1093/nar/gkaf1305)

## Licence

pylimma is a derivative work of limma. It is therefore licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See [`LICENSE`](LICENSE) for the full text.

pylimma additionally ports algorithms from the R packages `MASS`, `statmod`, base R `splines`, and `affy`. Per-module attribution to upstream contributors is recorded in each ported module's SPDX header and summarised in [`NOTICE`](NOTICE).

## Contributing

Contributions are welcome and wanted!

The kinds of contribution that are especially useful:

- Bug reports, particularly numerical divergences from R limma output. Fixture-reproducible reports (a small input that demonstrates the divergence) are the quickest to resolve.
- Additional R-parity fixtures for functions or parameter combinations not yet covered.
- Worked examples and tutorials, especially for proteomics and scverse workflows.
- Documentation improvements.
- Performance work on the hotter routines (for example `voom`, `camera`, `duplicate_correlation`), where we currently are less performant than the c code contained within the limma R pacakge.

The issue tracker at [github.com/john-mulvey/pylimma/issues](https://github.com/john-mulvey/pylimma/issues) is the primary entry point. A `CONTRIBUTING.md` with code style, commit conventions, and the PR process will land alongside the first tagged release. Until then, please open an issue to discuss larger changes before sending a pull request.
