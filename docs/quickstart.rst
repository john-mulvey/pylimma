Quickstart
==========

The minimum useful pylimma analysis: fit a linear model per gene,
moderate variances with empirical Bayes, and rank genes by evidence
for differential expression. The example below mirrors R limma's
introductory vignette.

Basic two-group comparison
--------------------------

.. code-block:: python

   import numpy as np
   from pylimma import lm_fit, contrasts_fit, e_bayes, top_table

   np.random.seed(0)
   expr   = np.random.normal(size=(100, 6))
   expr[:10, 3:] += 2.0                               # spike the first 10 genes
   design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])

   fit = lm_fit(expr, design)
   fit = contrasts_fit(fit, contrasts=np.array([[0], [1]]))
   fit = e_bayes(fit)
   print(top_table(fit, coef=0).head())

The output is a pandas DataFrame with the same columns as R limma's
``topTable`` - ``log_fc``, ``ave_expr``, ``t``, ``p_value``,
``adj_p_value``, and ``B`` (log-odds) - matching R output within
``rtol=1e-6``.

AnnData workflow
----------------

pylimma functions accept AnnData objects directly. Fit objects are
written to ``adata.uns[key]`` (default ``key="pylimma"``):

.. code-block:: python

   import anndata as ad
   import pylimma as pl

   adata = ad.read_h5ad("expression.h5ad")            # samples x genes
   pl.lm_fit(adata, design="~ group", key="fit")
   pl.e_bayes(adata, key="fit")
   results = pl.top_table(adata, coef=1, key="fit")   # DataFrame

EList workflow
--------------

For users migrating from R, the ``EList`` dict subclass mirrors R
limma's ``EList``:

.. code-block:: python

   from pylimma import EList, voom, lm_fit, e_bayes, top_table

   el  = EList({"E": counts, "design": design})      # genes x samples
   v   = voom(el)
   fit = lm_fit(v, design)
   fit = e_bayes(fit)
   results = top_table(fit, coef=1)

Next steps
----------

- :doc:`tutorials/gse60450` - voom pipeline on the GSE60450 mouse
  mammary dataset, the standard limma / voom tutorial.
- :doc:`tutorials/mulvey` - log-abundance proteomics workflow with
  missing-value handling.
- :doc:`tutorials/yoruba` - voom plus gene-set testing with
  ``camera`` and ``roast``.
- :doc:`api` - full API reference.
- :doc:`validation/known_differences` - documented numerical
  differences from R limma (all within published tolerances).
