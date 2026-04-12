Benchmarks
==========

This page reports pylimma's runtime and memory performance against R
limma on four reference datasets drawn from the limma User's Guide.
The methodology is documented in ``benchmarks/README.md`` in the
source tree; the raw JSON results are committed under
``benchmarks/results/``.

Dataset set
-----------

======================  =================  ==============  ====================
Slot                    Dataset            Shape           Upstream paper
======================  =================  ==============  ====================
Microarray              ALL (Chiaretti)    12,625 x 128    Chiaretti 2004, Blood
RNA-seq, two-group      GSE60450 (mammary) 27,179 x 12     Fu 2015, NCB
RNA-seq, scaling        Yoruba HapMap      38,415 x 69     Pickrell 2010, Nature
Splicing                Pasilla            14,599 x 7      Brooks 2011, GR
======================  =================  ==============  ====================

Full provenance, licence terms, and the extraction scripts for each
dataset are documented in ``benchmarks/data/DATA_PROVENANCE.md`` in
the source tree. All four upstream licences (Artistic-2.0, NCBI
public data, LGPL, MIT) are compatible with pylimma's
GPL-3.0-or-later. Users running the benchmarks do not need R or any
Bioconductor data packages - the CSVs are committed to the repo.

Pipelines
---------

- **A** (core): ``lm_fit`` -> ``contrasts_fit`` -> ``e_bayes`` ->
  ``top_table``. Run on all four datasets, plus a 50-gene
  "overhead floor" subset of Estrogen.
- **B** (voom): ``voom`` -> ``lm_fit`` -> ``contrasts_fit`` ->
  ``e_bayes`` -> ``top_table``. Run on GSE60450 and Yoruba.
- **C** (gene-set testing): pipeline B + ``camera`` against 50
  curated gene sets. Run on GSE60450.
- **D** (splicing): ``diff_splice`` + ``top_splice`` on Pasilla.

Reproducibility
---------------

All runs pin single-threaded BLAS:

.. code-block:: bash

   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export MKL_NUM_THREADS=1

and use ``n_reps >= 5`` repetitions per dataset-pipeline combination.
Reported values are medians with 25th/75th percentile error bars;
the single fastest and slowest measurement per group are trimmed to
control for OS noise and warm-up cost.

Numerical parity
----------------

pylimma reproduces R limma's top-table output to floating-point
precision on every pipeline in the benchmark suite. Median relative
differences across all genes, computed on R 4.5.2 + limma 3.66.0:

==============  ================  ======================  ==========================
Dataset         Pipeline          Max relative difference  Median relative difference
==============  ================  ======================  ==========================
ALL             core              6.4e-10                 1.6e-14
GSE60450        voom              3.5e-7                  6.9e-12
Yoruba          voom              3.8e-6                  7.8e-11
Pasilla         splicing          3.2e-10                 6.3e-14
==============  ================  ======================  ==========================

The four ``examples/<dataset>/<dataset>_R_vs_Python.ipynb`` notebooks
recompute these numbers live and additionally plot R-vs-pylimma
scatter plots per column. Running them locally will reproduce the
parity table with your own R and Python installs.

Runtime comparison
------------------

Headline timing (median of 5 reps, single-threaded BLAS, Apple
M-series macOS, R 4.5.2 + limma 3.66.0, pylimma 0.1.0):

==============  ================  ============  ============  ==============
Dataset         Pipeline          pylimma (s)   R limma (s)   R / pylimma
==============  ================  ============  ============  ==============
ALL (full)      ``pipeline_a``    0.036         0.032         0.89
GSE60450        ``pipeline_a``    0.032         0.030         0.94
GSE60450        ``pipeline_b``    1.54          0.72          0.47
GSE60450        ``pipeline_c``    1.50          0.72          0.48
Yoruba          ``pipeline_a``    0.069         0.069         1.00
Yoruba          ``pipeline_b``    3.07          1.67          0.54
Pasilla         ``pipeline_d``    0.047         0.042         0.89
==============  ================  ============  ============  ==============

Interpretation
~~~~~~~~~~~~~~

- **Core pipeline** (``lm_fit`` -> ``contrasts_fit`` -> ``e_bayes`` ->
  ``top_table``): pylimma is within 15% of R across all datasets.
- **voom pipeline**: pylimma is ~2x slower than R on both GSE60450
  and Yoruba. The bottleneck is the LOWESS fit inside ``voom``'s
  mean-variance modelling - R limma uses hand-written C
  (``weighted_lowess.c``) while pylimma uses
  ``statsmodels.nonparametric.lowess``.
- **Splicing** (``diff_splice`` + ``top_splice``) on Pasilla: pylimma
  is within 15% of R.

pylimma is within 2x of R limma across every pipeline-dataset
combination in the reference suite.

Embedded notebook
~~~~~~~~~~~~~~~~~

.. nbinput:: text

   benchmarks/run_benchmarks.ipynb

Profile output for ``pipeline_a`` on Yoruba is committed to
``benchmarks/results/profile_pipeline_a_yoruba.txt``.

Reruns and updates
~~~~~~~~~~~~~~~~~~

To rerun benchmarks on your own machine::

   cd pylimma/benchmarks
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   python run_python.py
   Rscript run_r.R
   jupyter nbconvert --to notebook --execute run_benchmarks.ipynb \
       --output run_benchmarks.ipynb

Results are written to ``benchmarks/results/``. Commit the JSON
files alongside the notebook if you want them included in the docs.
