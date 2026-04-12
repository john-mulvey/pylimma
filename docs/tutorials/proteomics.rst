Proteomics differential abundance
=================================

Proteomics abundance matrices are typically log2-transformed and
quantile- or median-normalised at the protein level before analysis.
Unlike RNA-seq counts, there is no mean-variance trend to model with
voom - use ``lm_fit`` directly, optionally with ``eBayes(trend=TRUE)``
to model a residual mean-variance trend.

Dataset
-------

A 1,500-protein log2-abundance matrix with missing values from a
cardiac-tissue proteomics experiment. Samples: 6 control, 6 heart
failure.

Full pipeline
-------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   import pylimma as pl

   abund = pd.read_csv("cardiac_proteomics.csv", index_col=0)  # proteins x samples
   group = ["control"] * 6 + ["hf"] * 6
   design = pd.get_dummies(pd.Categorical(group), drop_first=False).astype(float).values

   # Drop proteins with too many missing values (R limma convention:
   # keep proteins quantified in >= 50% of samples).
   keep  = abund.notna().sum(axis=1) >= (abund.shape[1] // 2)
   abund = abund.loc[keep]

   # Fit per-protein linear model - NaN values are propagated into per-protein
   # residual degrees of freedom inside lm_fit.
   fit = pl.lm_fit(abund.values, design)

   # Contrast: heart-failure minus control.
   C = np.array([[-1.0], [1.0]])
   fit2 = pl.contrasts_fit(fit, contrasts=C)

   # Trend-based empirical Bayes handles abundance-dependent variance.
   fit2 = pl.e_bayes(fit2, trend=True, robust=True)

   top = pl.top_table(fit2, coef=0, n=20)
   print(top[["log_fc", "ave_expr", "t", "p_value", "adj_p_value"]])

Notes
-----

- ``e_bayes(trend=True)`` fits a LOWESS mean-variance trend across
  proteins, matching R's ``eBayes(trend=TRUE)``. Use this when
  abundance and variance covary (almost always, for proteomics).
- ``e_bayes(robust=True)`` downweights outlier proteins during the
  variance-shrinkage step, matching R's ``eBayes(robust=TRUE)``.
- Missing values (``NaN``) are handled per-protein: the design matrix
  is subset to non-missing samples before QR decomposition, so
  proteins with different missing-value patterns can be analysed in
  the same call.
