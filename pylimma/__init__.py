"""
pylimma: Python port of R limma for differential expression analysis.
"""

from pylimma._version import __version__
from pylimma.classes import EList, MArrayLM, as_matrix_weights, get_eawp, put_eawp
from pylimma.lmfit import lm_fit, lm_series, mrlm, is_fullrank, non_estimable
from pylimma.contrasts import (
    make_contrasts,
    contrasts_fit,
    contrast_as_coef,
    model_matrix,
)
from pylimma.ebayes import e_bayes, treat, top_treat, pred_fcm
from pylimma.toptable import top_table, top_table_f
from pylimma.squeeze_var import (
    squeeze_var,
    fit_f_dist,
    fit_f_dist_robustly,
    fit_f_dist_unequal_df1,
)
from pylimma.decide_tests import decide_tests, classify_tests_f
from pylimma.utils import (
    qqt,
    qqf,
    choose_lowess_span,
    loess_fit,
    trigamma_inverse,
    zscore,
    zscore_t,
    zscore_gamma,
    zscore_hyper,
    tricube_moving_average,
    convest,
    prop_true_null,
    detection_p_values,
    weighted_lowess,
    is_numeric,
    block_diag,
    make_unique,
    logcosh,
    logsumexp,
    bwss,
    bwss_matrix,
    pool_var,
    cum_overlap,
    propexpr,
    fit_gamma_intercept,
)
from pylimma.auroc import au_roc
from pylimma.selmod import select_model
from pylimma.fitmixture_mod import fitmixture
from pylimma.genas import genas
from pylimma.geneset import (
    ids2indices,
    roast,
    mroast,
    fry,
    camera,
    camera_pr,
    inter_gene_correlation,
    romer,
    gene_set_test,
    rank_sum_test_with_correlation,
    wilcox_gst,
    top_romer,
)
from pylimma.voom import voom, voom_with_quality_weights, vooma, vooma_lm_fit, vooma_by_group
from pylimma.weights import array_weights, array_weights_quick, modify_weights
from pylimma.dups import (
    duplicate_correlation,
    ave_dups,
    avereps,
    unique_genelist,
    unwrap_dups,
)
from pylimma.normalize import (
    normalize_between_arrays,
    normalize_quantiles,
    normalize_median_values,
    normalize_cyclic_loess,
    normalize_vsn,
    normexp_fit,
    normexp_signal,
    background_correct,
    aver_arrays,
)
from pylimma.batch import remove_batch_effect, wsva
from pylimma.plotting import (
    plot_with_highlights,
    plot_ma,
    plot_ma_3by2,
    plot_md,
    volcano_plot,
    plot_sa,
    plot_densities,
    plot_mds,
    venn_counts,
    venn_diagram,
    coolmap,
    barcode_plot,
    plotlines,
    mdplot,
    heat_diagram,
    plot_rldf,
    plot_exons,
    plot_exon_junc,
)
from pylimma.splicing import diff_splice, top_splice, plot_splice
from pylimma.enrichment import goana, top_go, kegga, top_kegg, goana_trend

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
    "mrlm",
    # lm_series intentionally not exported - it is an implementation
    # detail of lm_fit and its public R counterpart takes ndups/spacing
    # kwargs that pylimma handles via lm_fit's dispatcher instead.
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
