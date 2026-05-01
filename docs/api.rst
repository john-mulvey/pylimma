API reference
=============

Every public name exported from :mod:`pylimma` has an entry below;
every entry below resolves to a public attribute on the top-level
``pylimma`` module. If the two lists ever disagree that is a bug -
file an issue.

Linear modelling
----------------

.. autosummary::
   :toctree: generated/

   pylimma.lm_fit
   pylimma.lm_series
   pylimma.gls_series
   pylimma.mrlm
   pylimma.contrasts_fit
   pylimma.make_contrasts
   pylimma.model_matrix
   pylimma.e_bayes
   pylimma.treat
   pylimma.top_table
   pylimma.top_table_f
   pylimma.top_treat
   pylimma.decide_tests
   pylimma.classify_tests_f
   pylimma.squeeze_var
   pylimma.fit_f_dist
   pylimma.fit_f_dist_robustly
   pylimma.fit_f_dist_unequal_df1
   pylimma.is_fullrank
   pylimma.non_estimable

Voom and RNA-seq
----------------

.. autosummary::
   :toctree: generated/

   pylimma.voom
   pylimma.voom_with_quality_weights
   pylimma.vooma
   pylimma.vooma_lm_fit

Normalisation and batch
-----------------------

.. autosummary::
   :toctree: generated/

   pylimma.normalize_between_arrays
   pylimma.normalize_quantiles
   pylimma.normalize_median_values
   pylimma.normalize_cyclic_loess
   pylimma.background_correct
   pylimma.normexp_fit
   pylimma.normexp_signal
   pylimma.aver_arrays
   pylimma.remove_batch_effect
   pylimma.wsva

Duplicates, weights, correlation
--------------------------------

.. autosummary::
   :toctree: generated/

   pylimma.duplicate_correlation
   pylimma.ave_dups
   pylimma.avereps
   pylimma.array_weights
   pylimma.array_weights_quick

Gene set testing
----------------

.. autosummary::
   :toctree: generated/

   pylimma.ids2indices
   pylimma.roast
   pylimma.mroast
   pylimma.fry
   pylimma.camera
   pylimma.camera_pr
   pylimma.inter_gene_correlation
   pylimma.romer
   pylimma.gene_set_test
   pylimma.rank_sum_test_with_correlation

GO / KEGG enrichment
--------------------

.. autosummary::
   :toctree: generated/

   pylimma.goana
   pylimma.top_go
   pylimma.kegga
   pylimma.top_kegg
   pylimma.goana_trend

Statistical utilities
---------------------

.. autosummary::
   :toctree: generated/

   pylimma.qqt
   pylimma.zscore_t
   pylimma.tricube_moving_average
   pylimma.convest
   pylimma.prop_true_null
   pylimma.detection_p_values
   pylimma.weighted_lowess
   pylimma.au_roc

Model selection and mixture models
----------------------------------

.. autosummary::
   :toctree: generated/

   pylimma.select_model
   pylimma.fitmixture
   pylimma.genas
   pylimma.pred_fcm

Splicing
--------

.. autosummary::
   :toctree: generated/

   pylimma.diff_splice
   pylimma.top_splice
   pylimma.plot_splice

Plotting
--------

.. autosummary::
   :toctree: generated/

   pylimma.plot_with_highlights
   pylimma.plot_ma
   pylimma.plot_md
   pylimma.volcano_plot
   pylimma.plot_sa
   pylimma.plot_densities
   pylimma.plot_mds
   pylimma.venn_counts
   pylimma.venn_diagram
   pylimma.coolmap
   pylimma.barcode_plot

Data classes and dispatchers
----------------------------

.. autosummary::
   :toctree: generated/

   pylimma.EList
   pylimma.MArrayLM
   pylimma.get_eawp
   pylimma.put_eawp
