pylimma documentation
=====================

pylimma is a faithful Python port of R limma, the most widely used
Bioconductor package for differential expression analysis. It provides
the full linear-modelling pipeline (``lm_fit``, ``contrasts_fit``,
``e_bayes``, ``top_table``), voom for RNA-seq, gene-set testing
(camera, roast, fry, romer), normalisation, batch correction, and
differential splicing - all validated to match R limma output within
``rtol=1e-6`` on fixture parity tests.

pylimma accepts numpy arrays, pandas DataFrames, AnnData objects, or
limma-style ``EList`` dict subclasses, with centralised polymorphic
dispatch so the same code works across the scverse ecosystem and
R-style workflows.

Quickstart
----------

.. code-block:: python

   import numpy as np
   from pylimma import lm_fit, contrasts_fit, e_bayes, top_table

   expr   = np.random.normal(size=(100, 6))
   design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

   fit = lm_fit(expr, design)
   fit = contrasts_fit(fit, contrasts=np.array([[0], [1]]))
   fit = e_bayes(fit)
   print(top_table(fit, coef=0).head())

See :doc:`quickstart` for the full walk-through.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api
   tutorials/voom_rnaseq
   tutorials/proteomics
   tutorials/gene_set_testing
   validation/known_differences
   validation/fixtures
   validation/benchmarks

Citation
--------

Please cite the original limma papers when using pylimma:

- Smyth, G. K. (2004). Linear models and empirical Bayes methods for
  assessing differential expression in microarray experiments.
  *Statistical Applications in Genetics and Molecular Biology* 3,
  Article 3.
- Ritchie, M. E., Phipson, B., Wu, D., Hu, Y., Law, C. W., Shi, W., and
  Smyth, G. K. (2015). limma powers differential expression analyses
  for RNA-sequencing and microarray studies. *Nucleic Acids Research*
  43(7), e47.
- Law, C. W., Chen, Y., Shi, W., and Smyth, G. K. (2014). voom:
  precision weights unlock linear model analysis tools for RNA-seq
  read counts. *Genome Biology* 15, R29.

A pylimma preprint / Zenodo DOI will be listed here once published.
Until then, cite the GitHub repository and the version tag.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
