"""
pylimma: Python port of R limma for differential expression analysis.
"""

from pylimma._version import __version__
from pylimma.auroc import au_roc
from pylimma.batch import remove_batch_effect, wsva
from pylimma.classes import EList, MArrayLM, as_matrix_weights, get_eawp, put_eawp
from pylimma.contrasts import (
    contrast_as_coef,
    contrasts_fit,
    make_contrasts,
    model_matrix,
)
from pylimma.decide_tests import classify_tests_f, decide_tests
from pylimma.dups import (
    ave_dups,
    avereps,
    duplicate_correlation,
    unique_genelist,
    unwrap_dups,
)
from pylimma.ebayes import e_bayes, pred_fcm, top_treat, treat
from pylimma.enrichment import goana, goana_trend, kegga, top_go, top_kegg
from pylimma.fitmixture_mod import fitmixture
from pylimma.genas import genas
from pylimma.geneset import (
    camera,
    camera_pr,
    fry,
    gene_set_test,
    ids2indices,
    inter_gene_correlation,
    mroast,
    rank_sum_test_with_correlation,
    roast,
    romer,
    top_romer,
    wilcox_gst,
)
from pylimma.lmfit import (
    gls_series,
    is_fullrank,
    lm_fit,
    mrlm,
    non_estimable,
)
from pylimma.lmfit import (
    lm_series as lm_series,
)
from pylimma.normalize import (
    aver_arrays,
    background_correct,
    normalize_between_arrays,
    normalize_cyclic_loess,
    normalize_median_values,
    normalize_quantiles,
    normalize_vsn,
    normexp_fit,
    normexp_signal,
)
from pylimma.plotting import (
    barcode_plot,
    coolmap,
    heat_diagram,
    mdplot,
    plot_densities,
    plot_exon_junc,
    plot_exons,
    plot_ma,
    plot_ma_3by2,
    plot_md,
    plot_mds,
    plot_rldf,
    plot_sa,
    plot_with_highlights,
    plotlines,
    venn_counts,
    venn_diagram,
    volcano_plot,
)
from pylimma.selmod import select_model
from pylimma.splicing import diff_splice, plot_splice, top_splice
from pylimma.squeeze_var import (
    fit_f_dist,
    fit_f_dist_robustly,
    fit_f_dist_unequal_df1,
    squeeze_var,
)
from pylimma.toptable import top_table, top_table_f
from pylimma.utils import (
    block_diag,
    bwss,
    bwss_matrix,
    choose_lowess_span,
    convest,
    cum_overlap,
    detection_p_values,
    fit_gamma_intercept,
    is_numeric,
    loess_fit,
    logcosh,
    logsumexp,
    make_unique,
    pool_var,
    prop_true_null,
    propexpr,
    qqf,
    qqt,
    tricube_moving_average,
    trigamma_inverse,
    weighted_lowess,
    zscore,
    zscore_gamma,
    zscore_hyper,
    zscore_t,
)
from pylimma.voom import voom, voom_with_quality_weights, vooma, vooma_by_group, vooma_lm_fit
from pylimma.weights import array_weights, array_weights_quick, modify_weights

__all__ = [
    "__version__",
    # Data classes and dispatchers
    "EList",
    "MArrayLM",
    "as_matrix_weights",
    "get_eawp",
    "put_eawp",
    # Core pipeline
    "lm_fit",
    "gls_series",
    "mrlm",
    # lm_series is intentionally not in __all__ - it is an
    # implementation detail of lm_fit. R limma exports lm.series via
    # NAMESPACE pattern but lm_fit's dispatcher already handles R's
    # ndups/spacing kwargs, so adding them to lm_series would be
    # redundant API surface. lm_series remains importable as
    # pylimma.lm_series for advanced users; it just isn't in *.
    "contrasts_fit",
    "contrast_as_coef",
    "e_bayes",
    "treat",
    "top_treat",
    "top_table",
    "top_table_f",
    # RNA-seq (voom)
    "voom",
    "voom_with_quality_weights",
    "vooma",
    "vooma_lm_fit",
    "array_weights",
    "array_weights_quick",
    "modify_weights",
    # Duplicate probes / replicates
    "duplicate_correlation",
    "ave_dups",
    "avereps",
    "unique_genelist",
    "unwrap_dups",
    # Normalization
    "normalize_between_arrays",
    "normalize_quantiles",
    "normalize_median_values",
    "normalize_cyclic_loess",
    "normalize_vsn",
    "normexp_fit",
    "normexp_signal",
    "background_correct",
    "aver_arrays",
    "remove_batch_effect",
    "wsva",
    # Visualisation
    "plot_with_highlights",
    "plot_ma",
    "plot_ma_3by2",
    "plot_md",
    "volcano_plot",
    "plot_sa",
    "plot_densities",
    "plot_mds",
    "venn_counts",
    "venn_diagram",
    "coolmap",
    "barcode_plot",
    "plotlines",
    "mdplot",
    "heat_diagram",
    "plot_rldf",
    "plot_exons",
    "plot_exon_junc",
    "vooma_by_group",
    # Splicing
    "diff_splice",
    "top_splice",
    "plot_splice",
    # GO / KEGG enrichment
    "goana",
    "top_go",
    "kegga",
    "top_kegg",
    "goana_trend",
    # Decision procedures
    "decide_tests",
    "classify_tests_f",
    # Contrast/design construction
    "make_contrasts",
    "model_matrix",
    # Variance shrinkage
    "squeeze_var",
    "fit_f_dist",
    "fit_f_dist_robustly",
    "fit_f_dist_unequal_df1",
    # Gene-set testing
    "ids2indices",
    "roast",
    "mroast",
    "fry",
    "camera",
    "camera_pr",
    "inter_gene_correlation",
    "romer",
    "gene_set_test",
    "rank_sum_test_with_correlation",
    "wilcox_gst",
    "top_romer",
    # Utilities
    "is_fullrank",
    "non_estimable",
    "qqt",
    "qqf",
    "choose_lowess_span",
    "loess_fit",
    "trigamma_inverse",
    "zscore",
    "zscore_t",
    "zscore_gamma",
    "zscore_hyper",
    "tricube_moving_average",
    "convest",
    "prop_true_null",
    "detection_p_values",
    "weighted_lowess",
    "is_numeric",
    "block_diag",
    "make_unique",
    "logcosh",
    "logsumexp",
    "bwss",
    "bwss_matrix",
    "pool_var",
    "cum_overlap",
    "propexpr",
    "fit_gamma_intercept",
    "au_roc",
    "select_model",
    "fitmixture",
    "genas",
    "pred_fcm",
]
