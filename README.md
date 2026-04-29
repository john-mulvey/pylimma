# pylimma

A Python port of [R limma](https://bioconductor.org/packages/limma/) for differential expression and abundance analysis.

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

## Scope

pylimma is a comprehensive port of limma with demonstrated parity in output to the R package.

pylimma covers linear modelling with empirical Bayes moderation (`lm_fit`, `contrasts_fit`, `e_bayes`, `treat`, `top_table`), voom and RNA-seq precision weights, normalisation and background correction, batch correction and surrogate variables, duplicate correlation and array weights, competitive and self-contained gene-set testing (camera, roast, mroast, fry, romer, geneSetTest), differential splicing, diagnostics plots, and the supporting statistical utilities. Some microarray specific functionality that we do not anticipate is commonly used has not been ported.

It accepts three input idioms for any analysis:

- **Flat array-like objects** - a NumPy `ndarray` or a pandas `DataFrame` of genes x samples
- **`EList`** - a Python port of R limma's `EList` class, implemented as a `dict` subclass in `pylimma.classes` with slots `E`, `weights`, `genes`, `targets`, `design`
- **`AnnData`** - read from `adata.X` or `adata.layers[layer]`, with shape-preserving outputs written back to `adata.layers[...]` and fits stored in `adata.uns[...]`.

## Installation

pylimma is not yet on PyPI. Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/john-mulvey/pylimma.git
```

Optional extras (use the same `git+` URL with the appropriate marker):

```bash
pip install "pylimma[plot] @ git+https://github.com/john-mulvey/pylimma.git"   # matplotlib for diagnostics
pip install "pylimma[dev]  @ git+https://github.com/john-mulvey/pylimma.git"   # pytest, ruff, mypy
pip install "pylimma[docs] @ git+https://github.com/john-mulvey/pylimma.git"   # sphinx + nbsphinx
```

For an editable development install:

```bash
git clone https://github.com/john-mulvey/pylimma.git
cd pylimma
pip install -e ".[dev,plot]"
```

Requires Python >= 3.10.

## Quick start

The canonical limma workflow in Python mirrors the R workflow one-for-one: `lm_fit`, `contrasts_fit`, `e_bayes` then `top_table`. Parameter names and defaults follow R limma.

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

The result here is a pandas `DataFrame` with the standard limma columns (`log_fc`, `ave_expr`, `t`, `p_value`, `adj_p_value`, `b`), sorted by B-statistic (as is the case for the R package).

## Validation and R parity

pylimma's output is validated against R limma on a reference corpus covering every public function. In many cases we demonstrate parity close to the precision limit of double-precision floating-point arithmetic. Agreement with the limma release ported (3.66.0) is tighter than the variation between different versions of limma itself: on voom-based pipelines, pylimma matches 3.66.0 to floating-point precision, whereas earlier limma releases (e.g. 3.54.2) diverge from 3.66.0 by up to a few percent.

## AnnData and the scverse ecosystem

Beyond the R-parity port, pylimma has been extended to operate natively on `AnnData` objects so that limma workflows slot into the scverse ecosystem alongside scanpy and muon. Any public function that consumes expression data (`voom`, `lm_fit`, `normalize_between_arrays`, `array_weights`, `duplicate_correlation`, ...) accepts an `AnnData` in place of a matrix. Shape-preserving outputs are written to `adata.layers[...]`, fits to `adata.uns[...]`, and results retrieved with the usual `top_table` / `decide_tests` calls:

```python
import pylimma as pl
import anndata as ad

adata = ad.read_h5ad("expression.h5ad")   # obs x var = samples x genes
pl.voom(adata, design)                    # writes adata.layers["voom_E"]
pl.lm_fit(adata, design)                  # stores fit in adata.uns["pylimma"]
pl.e_bayes(adata)
results = pl.top_table(adata, coef=1)     # returns a DataFrame
```

## Examples

The [`examples/`](examples/) directory contains tutorial notebooks organised by workflow. Each tutorial is self-contained, and together they cover the three input idioms (flat array/DataFrame, `EList`, and `AnnData`) across threee different workflows. Side-by-side R-vs-Python notebooks are provided for most datasets.

### Microarray differential expression

The Chiaretti et al. acute lymphoblastic leukaemia microarray dataset ([Chiaretti et al. 2004](https://doi.org/10.1182/blood-2003-09-3243)) demonstrates the classical limma microarray pipeline (`lm_fit` -> `contrasts_fit` -> `e_bayes` -> `top_table`) on the `EList` input idiom:

- [`all_chiaretti_tutorial.ipynb`](examples/all_chiaretti/all_chiaretti_tutorial.ipynb) - pylimma-only tutorial
- [`all_R_vs_Python.ipynb`](examples/all_chiaretti/all_R_vs_Python.ipynb) - side-by-side R limma vs pylimma comparison

### Bulk RNA-seq with voom

The GSE60450 mouse mammary RNA-seq dataset ([Fu et al. 2015](https://doi.org/10.1038/ncb3117)) demonstrates the canonical voom pipeline (`voom` -> `lm_fit` -> `e_bayes` -> `top_table`) on a plain count matrix:

- [`gse60450_tutorial.ipynb`](examples/gse60450/gse60450_tutorial.ipynb) - pylimma-only tutorial
- [`gse60450_R_vs_Python.ipynb`](examples/gse60450/gse60450_R_vs_Python.ipynb) - side-by-side R limma vs pylimma comparison

The Yoruba HapMap RNA-seq dataset ([Pickrell et al. 2010](https://doi.org/10.1038/nature08872)) demonstrates voom in combination with `duplicate_correlation` for handling technical replicates:

- [`yoruba_tutorial.ipynb`](examples/yoruba/yoruba_tutorial.ipynb) - pylimma-only tutorial
- [`yoruba_R_vs_Python.ipynb`](examples/yoruba/yoruba_R_vs_Python.ipynb) - side-by-side R limma vs pylimma comparison

### Single-cell pseudobulk differential expression

The Kang 2018 PBMC dataset ([Kang et al. 2018](https://doi.org/10.1038/nbt.4042)) demonstrates the `AnnData` input idiom end-to-end - pseudobulk aggregation per cell type followed by per-cluster differential expression, with fits stored in `adata.uns`:

- [`kang_pbmc_tutorial.ipynb`](examples/kang_pbmc/kang_pbmc_tutorial.ipynb) - pylimma-only tutorial

### Differential splicing

The pasilla *Drosophila* exon-level dataset ([Brooks et al. 2011](https://doi.org/10.1101/gr.108662.110)) demonstrates the differential splicing workflow (`diff_splice`, `top_splice`, `plot_splice`):

- [`pasilla_tutorial.ipynb`](examples/pasilla/pasilla_tutorial.ipynb) - pylimma-only tutorial
- [`pasilla_R_vs_Python.ipynb`](examples/pasilla/pasilla_R_vs_Python.ipynb) - side-by-side R limma vs pylimma comparison

## Authorship

The Python implementation was written primarily by Claude Code (Opus 4.5-4.7), working under the direction and review of John Mulvey. I carry responsibility for correctness as the auditor of the port, but the large majority of the line-level Python was produced by Claude Code.

## Citing pylimma

pylimma is a port, not a new method: no statistical methodology is introduced here. If you use pylimma in published work, please include two distinct citations:

1. **Software citation** - the pylimma implementation itself. <!-- Preprint citation to be added here once available: Mulvey, J. F. <title> (2026). bioRxiv. doi:<DOI> -->
2. **Methodology citation(s)** - the relevant original limma paper(s) listed under [limma](#limma) below, for the statistical methods underlying your analysis.

### limma

pylimma is a port of the [limma](https://bioconductor.org/packages/limma/) Bioconductor package. All statistical credit belongs to Gordon Smyth and the limma contributors listed in [`NOTICE`](NOTICE).

The limma publications are:

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

pylimma additionally ports algorithms from the R packages `MASS`, `statmod`, base R `splines`, and `affy`. Per-module attribution to upstream contributors is recorded in each ported module's SPDX header and summarised in [`NOTICE`](NOTICE).

## Licence

pylimma is a derivative work of limma. It is therefore licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

## Contributing

Contributions are welcome and wanted!

The kinds of contribution that are especially useful:

- Bug reports. We have taken great effort to comprehensively assess pylimma against limma. If there are differences in parity of outputs/fucntionality/interface, we would like to know about it!
- Additional R-parity fixtures for functions or parameter combinations not yet covered.
- Performance work on the hotter routines (for example `voom`, `camera`, `duplicate_correlation`), where we currently are less performant than the C code contained within the limma R package.

The issue tracker at [github.com/john-mulvey/pylimma/issues](https://github.com/john-mulvey/pylimma/issues) is the primary entry point. Please open an issue to discuss larger changes before sending a pull request.
