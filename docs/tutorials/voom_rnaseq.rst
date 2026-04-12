RNA-seq differential expression with voom
=========================================

This tutorial reproduces the RNA-seq voom workflow from the R limma
User's Guide using pylimma. It walks the same analysis path as
Chapter 15 of the user's guide and Law et al. (2014) using the
GSE60450 mouse mammary dataset.

Dataset
-------

**GSE60450** (Fu et al. 2015): mouse mammary gland epithelial
populations, three cell types (basal, luminal progenitor, mature
luminal) in two conditions (virgin, pregnant/lactating) with two
replicates each. The analysis matrix is ``n_genes = 27,179`` by
``n_samples = 12``.

Full pipeline
-------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   import pylimma as pl

   counts = pd.read_csv("gse60450_counts.csv", index_col=0)  # genes x samples
   group  = pd.Categorical([
       "basal.virgin",     "basal.virgin",
       "basal.lactating",  "basal.lactating",
       "lp.virgin",        "lp.virgin",
       "lp.lactating",     "lp.lactating",
       "ml.virgin",        "ml.virgin",
       "ml.lactating",     "ml.lactating",
   ])

   # Filter low-count genes (edgeR::filterByExpr-style threshold).
   keep   = (counts.values >= 10).sum(axis=1) >= 2
   counts = counts.loc[keep]

   # Build design matrix.
   design = pd.get_dummies(group, drop_first=False).astype(float).values

   # voom: transform counts to log-CPM with precision weights.
   v = pl.voom(counts.values, design)

   # Linear model fit.
   fit = pl.lm_fit(v["E"], design, weights=v["weights"])

   # Contrast: basal.lactating vs basal.virgin.
   C = np.zeros((design.shape[1], 1))
   C[0, 0] = -1.0     # basal.virgin
   C[1, 0] =  1.0     # basal.lactating
   fit2 = pl.contrasts_fit(fit, contrasts=C)
   fit2 = pl.e_bayes(fit2)

   top = pl.top_table(fit2, coef=0, n=10)
   print(top[["log_fc", "ave_expr", "t", "p_value", "adj_p_value"]])

Parity check against R
----------------------

The intended tutorial notebook in ``examples/rnaseq/voom_tutorial.ipynb``
hard-codes the first voom weight and the top-table rank-1 gene taken
from an R limma run on the same CSV, so the notebook serves as a
published smoke test. A full R-vs-Python side-by-side comparison is
in ``examples/rnaseq/voom_R_vs_Python.ipynb`` (development notebook,
not part of the user docs).
