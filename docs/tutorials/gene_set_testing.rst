Gene set testing with camera, fry, and roast
============================================

pylimma ports the six gene-set testing functions from R limma:
``camera``, ``cameraPR``, ``fry``, ``roast``, ``mroast``, and
``romer``, plus the helper ``ids2indices`` for building gene-set
index lists from symbol/ID mappings.

Pick the right test
-------------------

- ``camera`` / ``camera_pr`` - competitive test adjusting for
  inter-gene correlation. Fast; best default for most workflows.
- ``fry`` - self-contained test (no inter-gene correlation).
  Deterministic and fast; equivalent to ``roast`` with infinite
  rotations.
- ``roast`` / ``mroast`` - rotation-based self-contained test. Use
  when you need a self-contained null and want to assess Monte-Carlo
  variability.
- ``romer`` - rotation-based competitive test, the complement of
  ``roast``.

Example: camera on the mammary dataset
--------------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   import pylimma as pl

   expr   = pd.read_csv("logcpm.csv", index_col=0)             # genes x samples
   design = pd.read_csv("design.csv").values
   contrast = np.array([[-1.0], [1.0]])

   # Fit base model.
   fit = pl.lm_fit(expr.values, design)
   fit = pl.contrasts_fit(fit, contrasts=contrast)
   fit = pl.e_bayes(fit)

   # Translate a gene-symbol-keyed pathway dict into row indices.
   # msigdbr, GO.db, or any curated source will do.
   pathways = {
       "HALLMARK_MYOGENESIS":  ["Myog", "Myod1", "Mef2c", ...],
       "HALLMARK_GLYCOLYSIS":  ["Hk2", "Pfkl", "Aldoa", ...],
       # ...
   }
   gene_sets = pl.ids2indices(pathways, list(expr.index))

   res = pl.camera(expr.values, index=gene_sets,
                   design=design, contrast=contrast)
   print(res.sort_values("p_value").head())

Monte-Carlo caveat for rotation tests
-------------------------------------

``roast``, ``mroast``, and ``romer`` draw rotations from NumPy's
random number generator, which does not produce byte-identical
streams to R's Mersenne-Twister. Deterministic summaries
(``ngenes_in_set``, observed test statistics, active proportions)
match R limma to ``rtol=1e-6``. Monte-Carlo p-values agree within
sampling error (empirically ~0.3 log10 at the default ``nrot=999``).
See :doc:`../validation/known_differences` for the tolerance
specification and reproducer.
