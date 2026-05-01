# Benchmark dataset provenance and licensing

pylimma's benchmark suite redistributes five publicly available
biological datasets. The CSVs in this directory were extracted from
their upstream Bioconductor data packages, GEO supplementary files,
or the published article's supporting data on the date noted in
each section. None of the upstream data licences conflict with
pylimma's own GPL-3.0-or-later licence.

Dataset redistribution is deliberate (it pins the benchmark inputs
to a specific version and removes the link-rot failure mode from
downstream users), and follows the precedent set by edgePython which
bundles its tutorial datasets inside the repository.

Please cite the original publications, not pylimma, when using these
datasets for anything other than running the benchmark suite.

---

## 1. ALL (Chiaretti 2004) - `all_expr.csv.gz`, `all_targets.csv`

**What**: RMA-normalised Affymetrix HG-U95Av2 expression matrix from
bone-marrow or peripheral-blood samples of 128 adult acute lymphoblastic
leukemia patients, with B-cell / T-cell assignment and molecular-biology
subtype (BCR/ABL, ALL1/AF4, NEG, ...) in the phenotype table.
12,625 genes x 128 samples.

**Upstream paper**. Cite this if you use the data in a study:

> Chiaretti, S., Li, X., Gentleman, R., Vitale, A., Vignetti, M.,
> Mandelli, F., Ritz, J., and Foa, R. (2004). Gene expression profile
> of adult T-cell acute lymphocytic leukemia identifies distinct
> subsets of patients with different response to therapy and
> survival. *Blood* 103(7), 2771-2778. doi:10.1182/blood-2003-09-3243

**Source**. Bioconductor `ALL` package version 1.52.0 (Bioc 3.22).
  - Author: Xiaochun Li
  - Maintainer: Robert Gentleman
  - Package URL: https://bioconductor.org/packages/ALL/
  - Licence: **Artistic-2.0**

**Extraction**. The `_setup_datasets.R` script installs the `ALL`
package via BiocManager, calls `data("ALL")` to load the
ExpressionSet, then writes:

- `all_expr.csv.gz` from `exprs(ALL)`
- `all_targets.csv` from the `BT`, `mol.biol`, `sex`, `age` columns
  of `pData(ALL)`

The first 50 rows are additionally written to `all_small_expr.csv.gz`
for the "fixed-cost floor" benchmark slot.

**Extracted**: 2026-04-17.

---

## 2. GSE60450 (Fu et al. 2015) - `gse60450_counts.csv.gz`, `gse60450_targets.csv`

**What**: Mouse mammary gland epithelial RNA-seq, three cell types
(basal, luminal progenitor, mature luminal) crossed with two
developmental conditions (virgin, pregnant/lactate). 27,179 genes x
12 samples, raw Entrez-gene-level counts.

**Upstream paper**:

> Fu, N.Y., Rios, A.C., Pal, B., Soetanto, R., Lun, A.T., Liu, K.,
> Beck, T., Best, S.A., Vaillant, F., Bouillet, P., Strasser, A.,
> Preiss, T., Smyth, G.K., Lindeman, G.J., and Visvader, J.E. (2015).
> EGF-mediated induction of Mcl-1 at the switch to lactation is
> essential for alveolar cell survival. *Nature Cell Biology* 17,
> 365-375. doi:10.1038/ncb3117

**Source**. NCBI Gene Expression Omnibus accession GSE60450,
supplementary file
`GSE60450_Lactation-GenewiseCounts.txt.gz`. Downloaded by
`_setup_gse60450.py` via HTTPS from
`ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60450/suppl/`.

**Licence**. Data deposited in NCBI GEO is a public resource under
NCBI's terms of use; GEO data is treated as freely re-usable with
attribution to the depositors. See
https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html for the full
terms.

**Extracted**: 2026-04-17.

---

## 3. Pasilla (Brooks 2011) - `pasilla_counts.csv.gz`, `pasilla_targets.csv`

**What**: Drosophila melanogaster S2-DRSC RNA-seq with knockdown of
the pasilla splicing regulator, 7 samples. 14,599 genes x 7 samples,
raw counts. The canonical splicing-benchmark dataset for
limma / DEXSeq / edgeR's `diffSplice`.

**Upstream paper**:

> Brooks, A.N., Yang, L., Duff, M.O., Hansen, K.D., Park, J.W.,
> Dudoit, S., Brenner, S.E., and Graveley, B.R. (2011). Conservation
> of an RNA regulatory map between Drosophila and mammals.
> *Genome Research* 21, 193-202. doi:10.1101/gr.108662.110

**Source**. Bioconductor `pasilla` package version 1.38.0. The
counts ship as `inst/extdata/pasilla_gene_counts.tsv`; we extract
them directly from the source tarball via `_setup_datasets.R`.

  - Authors: Wolfgang Huber, Alejandro Reyes
  - Package URL: https://bioconductor.org/packages/pasilla/
  - Licence: **LGPL**

**Extracted**: 2026-04-17.

---

## 4. Yoruba / Pickrell (Pickrell 2010) - `yoruba_counts.csv.gz`, `yoruba_targets.csv`

**What**: Population-scale RNA-seq of lymphoblastoid cell lines from
69 individuals in the Yoruba HapMap cohort (YRI). 38,415 genes x 69
samples, raw counts. Classic RNA-seq scaling-test dataset.

**Upstream paper**:

> Pickrell, J.K., Marioni, J.C., Pai, A.A., Degner, J.F., Engelhardt,
> B.E., Nkadori, E., Veyrieras, J.-B., Stephens, M., Gilad, Y., and
> Pritchard, J.K. (2010). Understanding mechanisms underlying human
> gene expression variation with RNA sequencing. *Nature* 464,
> 768-772. doi:10.1038/nature08872

**Source**. Bioconductor `tweeDEseqCountData` package version
1.48.0. The counts ship as the `pickrell1` (really named
`pickrell1.eset`) ExpressionSet; `_setup_datasets.R` extracts both
`exprs(...)` and `pData(...)`.

  - Maintainer: Dolors Pelegri-Siso (ISGlobal, Barcelona)
  - Package URL: https://github.com/isglobal-brge/tweeDEseqCountData/
  - Licence: **MIT** (the package's DESCRIPTION declares MIT, with a
    stub `+ file LICENSE` that is absent in the tarball; standard
    Bioconductor metadata convention).

**Extracted**: 2026-04-17.

---

## 5. Mulvey 2026 cardiac proteomics - `mulvey_proteome.csv.gz`, `mulvey_targets.csv`

**What**: Tissue proteomics from a human cardiac cohort comprising 6
non-failing controls and 9 heart-failure samples (5 peripartum
cardiomyopathy + 4 non-peripartum cardiomyopathy, collapsed into a
single `heart_failure` group for the demo). FragPipe proteingroup
matrix, log2-transformed and median-subtraction-normalised. ~5,000
proteins x 15 samples, indexed by UniProt accession with gene
symbol, PSM count, max peptide probability and reference intensity
as leading metadata columns.

This is the proteomics-flavoured worked example for the AnnData
input idiom; the limma microarray-style pipeline (no `voom`) applied
to a log-abundance proteomics matrix.

**Upstream paper**. Cite this if you use the data in a study:

> Mulvey, J. F., Sailer, C., Achter, J. S., Milburn, G. N.,
> Bretherton, R. C., Kahnert, K., Erbil Bilir, S., Hvid, H., Pyke, C.,
> Gustafsson, F., Adamo, L., Campbell, K. S., Herum, K. M., and
> Lundby, A. (2026). An Unbiased Molecular Characterization of
> Peripartum Cardiomyopathy Hearts Identifies Mast Cell Chymase as
> a New Diagnostic Candidate. *Molecular & Cellular Proteomics*
> 25(2), 101510. doi:10.1016/j.mcpro.2026.101510

**Source**. The raw spectra and FragPipe processing outputs are
deposited in ProteomeXchange via PRIDE (accession TBD - confirm
from the published article's Data Availability statement before
release). The CSVs in this directory are produced by
`extract_mulvey_proteomics.py` at the project root from the
median-subtraction-normalised proteingroup matrix.

**Licence**. *Molecular & Cellular Proteomics* has been a fully
open-access journal under **CC BY 4.0** since 2021, so the published
article and its supporting data are GPL-compatible. Confirm the
specific licence stamp on the article landing page before tagging
a release.

PRIDE deposits are released under the depositor's choice of CC0 or
CC BY 4.0; both are GPL-compatible.

**Author standing**. The lead author of pylimma (John Mulvey) is
the first author of Mulvey 2026, so redistribution of the derived
matrix in this repository is consented to by an author of the
source work.

**Extracted**: 2026-04-29.

---

## Licence compatibility summary

All five upstream licences are compatible with pylimma's
GPL-3.0-or-later licence:

| Dataset          | Upstream licence | Compatible with GPL-3.0-or-later? |
|------------------|------------------|-----------------------------------|
| ALL              | Artistic-2.0     | Yes (per FSF's GPL compatibility list) |
| GSE60450         | NCBI public data | Yes (no restrictions beyond attribution) |
| Pasilla          | LGPL             | Yes (LGPL is a strict subset of GPL terms) |
| Pickrell/Yoruba  | MIT              | Yes (MIT is GPL-compatible)         |
| Mulvey 2026      | CC BY 4.0 (MCP)  | Yes (CC BY 4.0 is GPL-compatible)   |

pylimma redistributes the data values only, not the Bioconductor
package source code or the journal's typeset article. The
attribution above fulfils the attribution requirements of each
upstream licence.

If you are an upstream author and object to pylimma's
redistribution, please open an issue on the pylimma repository and
we will remove the dataset or replace it with an equivalent.
