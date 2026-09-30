#!/usr/bin/env Rscript
# Comprehensive fixture generation for pylimma R parity tests
#
# Run with: Rscript generate_all_fixtures.R
#
# This script generates reference values for all major limma functions
# with multiple parameter combinations following edgePython's pattern.
#
# R version and limma version are logged for reproducibility.

library(limma)

cat("=== pylimma Fixture Generation ===\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("limma version: %s\n", packageVersion("limma")))
cat(sprintf("Date: %s\n\n", Sys.time()))

set.seed(42)

# -----------------------------------------------------------------------------
# Test Dataset 1: Basic two-group comparison (50 genes, 8 samples)
# -----------------------------------------------------------------------------
cat("Generating Dataset 1: Basic two-group...\n")

n_genes <- 50
n_samples <- 8

expr1 <- matrix(rnorm(n_genes * n_samples), nrow = n_genes)
rownames(expr1) <- paste0("gene", 1:n_genes)
colnames(expr1) <- paste0("sample", 1:n_samples)

# Add true effects to first 10 genes (group B higher)
expr1[1:10, 5:8] <- expr1[1:10, 5:8] + 2

# Design
group1 <- factor(rep(c("A", "B"), each = 4))
design1 <- model.matrix(~ group1)
colnames(design1) <- c("Intercept", "groupB")

# Save input data
write.csv(expr1, "R_data1_expr.csv", row.names = TRUE)
write.csv(design1, "R_data1_design.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# lmFit tests
# -----------------------------------------------------------------------------
cat("  lmFit (basic)...\n")
fit1 <- lmFit(expr1, design1)

write.csv(fit1$coefficients, "R_lmfit_basic_coef.csv", row.names = TRUE)
write.csv(fit1$stdev.unscaled, "R_lmfit_basic_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(
    sigma = fit1$sigma,
    df_residual = fit1$df.residual,
    Amean = fit1$Amean
  ),
  "R_lmfit_basic_stats.csv",
  row.names = TRUE
)
write.csv(fit1$cov.coefficients, "R_lmfit_basic_cov.csv", row.names = TRUE)

# lmFit with weights (array weights)
cat("  lmFit (array weights)...\n")
weights1 <- rep(c(1, 2, 1, 2, 1, 2, 1, 2), 1)  # Sample weights
fit1_wt <- lmFit(expr1, design1, weights = weights1)

write.csv(fit1_wt$coefficients, "R_lmfit_weights_coef.csv", row.names = TRUE)
write.csv(fit1_wt$stdev.unscaled, "R_lmfit_weights_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(sigma = fit1_wt$sigma, df_residual = fit1_wt$df.residual),
  "R_lmfit_weights_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# eBayes tests
# -----------------------------------------------------------------------------
cat("  eBayes (basic)...\n")
eb1 <- eBayes(fit1)

write.csv(
  data.frame(
    t_1 = eb1$t[, 1],
    t_2 = eb1$t[, 2],
    p_value_1 = eb1$p.value[, 1],
    p_value_2 = eb1$p.value[, 2],
    lods_1 = eb1$lods[, 1],
    lods_2 = eb1$lods[, 2],
    s2_post = eb1$s2.post,
    df_total = eb1$df.total
  ),
  "R_ebayes_basic_stats.csv",
  row.names = TRUE
)

write.csv(
  data.frame(
    s2_prior = eb1$s2.prior,
    df_prior = eb1$df.prior,
    F_stat = eb1$F,
    F_p_value = eb1$F.p.value
  ),
  "R_ebayes_basic_global.csv",
  row.names = FALSE
)

# eBayes with trend
cat("  eBayes (trend=TRUE)...\n")
eb1_trend <- eBayes(fit1, trend = TRUE)

write.csv(
  data.frame(
    t_2 = eb1_trend$t[, 2],
    p_value_2 = eb1_trend$p.value[, 2],
    lods_2 = eb1_trend$lods[, 2],
    s2_post = eb1_trend$s2.post,
    s2_prior = eb1_trend$s2.prior  # Array when trend=TRUE
  ),
  "R_ebayes_trend_stats.csv",
  row.names = TRUE
)

write.csv(
  data.frame(df_prior = eb1_trend$df.prior),
  "R_ebayes_trend_global.csv",
  row.names = FALSE
)

# eBayes with robust
cat("  eBayes (robust=TRUE)...\n")
eb1_robust <- eBayes(fit1, robust = TRUE)

write.csv(
  data.frame(
    t_2 = eb1_robust$t[, 2],
    p_value_2 = eb1_robust$p.value[, 2],
    lods_2 = eb1_robust$lods[, 2],
    s2_post = eb1_robust$s2.post,
    df_prior = eb1_robust$df.prior  # Can be array when robust
  ),
  "R_ebayes_robust_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# treat tests
# -----------------------------------------------------------------------------
cat("  treat (lfc=0.5)...\n")
treat1 <- treat(fit1, lfc = 0.5)

write.csv(
  data.frame(
    t_2 = treat1$t[, 2],
    p_value_2 = treat1$p.value[, 2],
    s2_post = treat1$s2.post,
    df_total = treat1$df.total
  ),
  "R_treat_lfc05_stats.csv",
  row.names = TRUE
)

cat("  treat (lfc=1.0)...\n")
treat1_lfc1 <- treat(fit1, lfc = 1.0)

write.csv(
  data.frame(
    t_2 = treat1_lfc1$t[, 2],
    p_value_2 = treat1_lfc1$p.value[, 2]
  ),
  "R_treat_lfc10_stats.csv",
  row.names = TRUE
)

# treat with upshot=TRUE (tests integration for one-sided hypotheses)
cat("  treat (lfc=0.5, upshot=TRUE)...\n")
treat1_upshot <- treat(fit1, lfc = 0.5, upshot = TRUE)

write.csv(
  data.frame(
    t_2 = treat1_upshot$t[, 2],
    p_value_2 = treat1_upshot$p.value[, 2],
    s2_post = treat1_upshot$s2.post,
    df_total = treat1_upshot$df.total
  ),
  "R_treat_upshot_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# topTable tests
# -----------------------------------------------------------------------------
cat("  topTable (coef=2, various options)...\n")

# Basic
tt1 <- topTable(eb1, coef = 2, number = Inf, sort.by = "P")
write.csv(tt1, "R_toptable_basic.csv", row.names = TRUE)

# With confint
tt1_ci <- topTable(eb1, coef = 2, number = 20, confint = TRUE)
write.csv(tt1_ci, "R_toptable_confint.csv", row.names = TRUE)

# Sort by B
tt1_sortB <- topTable(eb1, coef = 2, number = 20, sort.by = "B")
write.csv(tt1_sortB, "R_toptable_sortB.csv", row.names = TRUE)

# Filter by p-value
tt1_filter <- topTable(eb1, coef = 2, number = Inf, p.value = 0.05)
write.csv(tt1_filter, "R_toptable_pfilter.csv", row.names = TRUE)

# F-test (multiple coefficients)
tt1_F <- topTable(eb1, coef = NULL, number = 20)
write.csv(tt1_F, "R_toptable_Ftest.csv", row.names = TRUE)

# Duplicated rownames handling
cat("  topTable (duplicated rownames)...\n")
# Create expression matrix with duplicated gene names
expr_dup <- matrix(rnorm(10 * 8), nrow = 10)
rownames(expr_dup) <- c("GeneA", "GeneA", "GeneB", "GeneC", "GeneC",
                        "GeneD", "GeneE", "GeneE", "GeneF", "GeneG")
colnames(expr_dup) <- paste0("sample", 1:8)
# Save expression and design for Python parity test
write.csv(expr_dup, "R_toptable_duplicated_expr.csv", row.names = TRUE)
write.csv(design1, "R_toptable_duplicated_design.csv", row.names = FALSE)
fit_dup <- lmFit(expr_dup, design1)
eb_dup <- eBayes(fit_dup)
tt_dup <- topTable(eb_dup, coef = 2, number = Inf, sort.by = "none")
write.csv(tt_dup, "R_toptable_duplicated.csv", row.names = TRUE)

# topTreat
cat("  topTreat...\n")
tt_treat <- topTreat(treat1, coef = 2, number = 20)
write.csv(tt_treat, "R_toptreat_basic.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# contrasts.fit tests
# -----------------------------------------------------------------------------
cat("  contrasts.fit...\n")

# Cell means model for contrast testing
design1_cm <- model.matrix(~ 0 + group1)
colnames(design1_cm) <- levels(group1)
fit1_cm <- lmFit(expr1, design1_cm)

# Contrast matrix
cont <- makeContrasts(
  BvsA = B - A,
  levels = design1_cm
)

fit1_cont <- contrasts.fit(fit1_cm, cont)
eb1_cont <- eBayes(fit1_cont)

write.csv(fit1_cont$coefficients, "R_contrasts_coef.csv", row.names = TRUE)
write.csv(fit1_cont$stdev.unscaled, "R_contrasts_stdev.csv", row.names = TRUE)
write.csv(fit1_cont$cov.coefficients, "R_contrasts_cov.csv", row.names = TRUE)
write.csv(
  data.frame(t = eb1_cont$t, p_value = eb1_cont$p.value, lods = eb1_cont$lods),
  "R_contrasts_ebayes.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# squeezeVar / fitFDist tests
# -----------------------------------------------------------------------------
cat("  squeezeVar / fitFDist...\n")

# Basic squeezeVar
sv1 <- squeezeVar(fit1$sigma^2, fit1$df.residual)
write.csv(
  data.frame(
    var_post = sv1$var.post,
    var_prior = sv1$var.prior,
    df_prior = sv1$df.prior
  ),
  "R_squeezevar_basic.csv",
  row.names = TRUE
)

# squeezeVar with covariate (trend)
sv1_trend <- squeezeVar(fit1$sigma^2, fit1$df.residual, covariate = fit1$Amean)
write.csv(
  data.frame(
    var_post = sv1_trend$var.post,
    var_prior = sv1_trend$var.prior,
    df_prior = sv1_trend$df.prior
  ),
  "R_squeezevar_trend.csv",
  row.names = TRUE
)

# squeezeVar robust
sv1_robust <- squeezeVar(fit1$sigma^2, fit1$df.residual, robust = TRUE)
write.csv(
  data.frame(
    var_post = sv1_robust$var.post,
    var_prior = sv1_robust$var.prior,
    df_prior = sv1_robust$df.prior
  ),
  "R_squeezevar_robust.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# Test Dataset 2: Three-group comparison (100 genes, 12 samples)
# -----------------------------------------------------------------------------
cat("\nGenerating Dataset 2: Three-group...\n")

n_genes2 <- 100
n_samples2 <- 12

expr2 <- matrix(rnorm(n_genes2 * n_samples2), nrow = n_genes2)
rownames(expr2) <- paste0("gene", 1:n_genes2)

# Add effects
expr2[1:20, 5:8] <- expr2[1:20, 5:8] + 1.5   # Group B effect
expr2[11:30, 9:12] <- expr2[11:30, 9:12] + 2  # Group C effect

group2 <- factor(rep(c("A", "B", "C"), each = 4))
design2_cm <- model.matrix(~ 0 + group2)
colnames(design2_cm) <- levels(group2)

write.csv(expr2, "R_data2_expr.csv", row.names = TRUE)
write.csv(design2_cm, "R_data2_design.csv", row.names = FALSE)

# Fit
fit2 <- lmFit(expr2, design2_cm)

# Multiple contrasts
cont2 <- makeContrasts(
  BvsA = B - A,
  CvsA = C - A,
  CvsB = C - B,
  levels = design2_cm
)

fit2_cont <- contrasts.fit(fit2, cont2)
eb2 <- eBayes(fit2_cont)

write.csv(eb2$coefficients, "R_multicontrast_coef.csv", row.names = TRUE)
write.csv(
  data.frame(
    t_BvsA = eb2$t[, 1],
    t_CvsA = eb2$t[, 2],
    t_CvsB = eb2$t[, 3],
    p_BvsA = eb2$p.value[, 1],
    p_CvsA = eb2$p.value[, 2],
    p_CvsB = eb2$p.value[, 3],
    F = eb2$F,
    F_p = eb2$F.p.value
  ),
  "R_multicontrast_stats.csv",
  row.names = TRUE
)

# decideTests
cat("  decideTests...\n")
dt2 <- decideTests(eb2, method = "separate", adjust.method = "BH", p.value = 0.05)
write.csv(as.data.frame(dt2), "R_decidetests_separate.csv", row.names = TRUE)

dt2_global <- decideTests(eb2, method = "global", adjust.method = "BH", p.value = 0.05)
write.csv(as.data.frame(dt2_global), "R_decidetests_global.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# Test Dataset 3: With missing values (30 genes, 6 samples)
# -----------------------------------------------------------------------------
cat("\nGenerating Dataset 3: With missing values...\n")

n_genes3 <- 30
n_samples3 <- 6

expr3 <- matrix(rnorm(n_genes3 * n_samples3), nrow = n_genes3)
rownames(expr3) <- paste0("gene", 1:n_genes3)

# Add some NA values
expr3[1, 1] <- NA
expr3[2, c(1, 2)] <- NA
expr3[3, 1:3] <- NA  # Many missing

group3 <- factor(rep(c("A", "B"), each = 3))
design3 <- model.matrix(~ group3)

write.csv(expr3, "R_data3_expr_na.csv", row.names = TRUE)
write.csv(design3, "R_data3_design.csv", row.names = FALSE)

fit3 <- lmFit(expr3, design3)
eb3 <- eBayes(fit3)

write.csv(
  data.frame(
    coef_1 = fit3$coefficients[, 1],
    coef_2 = fit3$coefficients[, 2],
    sigma = fit3$sigma,
    df_residual = fit3$df.residual,
    t_2 = eb3$t[, 2],
    p_value_2 = eb3$p.value[, 2]
  ),
  "R_na_handling_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# Test Dataset 4: Unequal df (for fitFDistUnequalDF1)
# -----------------------------------------------------------------------------
cat("\nGenerating Dataset 4: Unequal df...\n")

# Different df per gene (simulated by having different missing patterns)
n_genes4 <- 50
n_samples4 <- 10

expr4 <- matrix(rnorm(n_genes4 * n_samples4), nrow = n_genes4)
rownames(expr4) <- paste0("gene", 1:n_genes4)

# Create varying missing patterns
for (i in 1:10) {
  expr4[i, sample(1:n_samples4, i %% 5)] <- NA
}

group4 <- factor(rep(c("A", "B"), each = 5))
design4 <- model.matrix(~ group4)

write.csv(expr4, "R_data4_expr_unequaldf.csv", row.names = TRUE)
write.csv(design4, "R_data4_design.csv", row.names = FALSE)

fit4 <- lmFit(expr4, design4)
eb4 <- eBayes(fit4)

write.csv(
  data.frame(
    df_residual = fit4$df.residual,
    sigma = fit4$sigma,
    t_2 = eb4$t[, 2],
    p_value_2 = eb4$p.value[, 2],
    df_prior = eb4$df.prior,
    s2_prior = eb4$s2.prior,
    df_total = eb4$df.total
  ),
  "R_unequaldf_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# Test Dataset 5: duplicateCorrelation and block correlation
# -----------------------------------------------------------------------------
cat("\nGenerating Dataset 5: duplicateCorrelation...\n")

# Dataset with technical replicates (4 subjects, 3 replicates each)
n_genes5 <- 40
n_samples5 <- 12
expr5 <- matrix(rnorm(n_genes5 * n_samples5), nrow = n_genes5)
rownames(expr5) <- paste0("gene", 1:n_genes5)
colnames(expr5) <- paste0("sample", 1:n_samples5)

# Add subject-level effects (correlation structure)
subject_effects <- matrix(rep(rnorm(n_genes5 * 4), each = 3), nrow = n_genes5, byrow = FALSE)
expr5 <- expr5 + subject_effects * 0.5

# Add group effect to first 10 genes
expr5[1:10, 7:12] <- expr5[1:10, 7:12] + 1.5

# Block structure: 4 subjects, 3 replicates each
block5 <- factor(rep(1:4, each = 3))
group5 <- factor(rep(c("A", "B"), each = 6))
design5 <- model.matrix(~ group5)
colnames(design5) <- c("Intercept", "groupB")

write.csv(expr5, "R_dupcor_expr.csv", row.names = TRUE)
write.csv(design5, "R_dupcor_design.csv", row.names = FALSE)
write.csv(data.frame(block = as.integer(block5)), "R_dupcor_block.csv", row.names = FALSE)

# duplicateCorrelation
cat("  duplicateCorrelation...\n")
dupcor <- duplicateCorrelation(expr5, design5, block = block5)

# Per-gene correlations (atanh transformed in R, we save the correlation)
write.csv(
  data.frame(
    consensus_correlation = dupcor$consensus.correlation,
    atanh_consensus = dupcor$consensus
  ),
  "R_dupcor_consensus.csv",
  row.names = FALSE
)
write.csv(
  data.frame(cor = dupcor$cor),
  "R_dupcor_pergene.csv",
  row.names = TRUE
)

# lmFit with block correlation (gls.series path)
cat("  lmFit with block correlation...\n")
fit5_block <- lmFit(expr5, design5, block = block5,
                    correlation = dupcor$consensus.correlation)

write.csv(fit5_block$coefficients, "R_lmfit_block_coef.csv", row.names = TRUE)
write.csv(fit5_block$stdev.unscaled, "R_lmfit_block_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(
    sigma = fit5_block$sigma,
    df_residual = fit5_block$df.residual,
    Amean = fit5_block$Amean
  ),
  "R_lmfit_block_stats.csv",
  row.names = TRUE
)
write.csv(fit5_block$cov.coefficients, "R_lmfit_block_cov.csv", row.names = TRUE)

# eBayes on block-correlated fit
eb5_block <- eBayes(fit5_block)
write.csv(
  data.frame(
    t_2 = eb5_block$t[, 2],
    p_value_2 = eb5_block$p.value[, 2],
    s2_post = eb5_block$s2.post
  ),
  "R_ebayes_block_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# Gene weights (matrix weights, not just array weights)
# -----------------------------------------------------------------------------
cat("  lmFit (gene-level weights matrix)...\n")

# Create gene x sample weight matrix
set.seed(43)
gene_weights <- matrix(runif(n_genes * n_samples, 0.5, 2), nrow = n_genes)
rownames(gene_weights) <- rownames(expr1)

write.csv(gene_weights, "R_lmfit_geneweights_weights.csv", row.names = TRUE)

fit1_gw <- lmFit(expr1, design1, weights = gene_weights)
write.csv(fit1_gw$coefficients, "R_lmfit_geneweights_coef.csv", row.names = TRUE)
write.csv(fit1_gw$stdev.unscaled, "R_lmfit_geneweights_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(sigma = fit1_gw$sigma, df_residual = fit1_gw$df.residual),
  "R_lmfit_geneweights_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# mrlm (robust fitting)
# -----------------------------------------------------------------------------
cat("  mrlm (robust fitting with outliers)...\n")

# Add outliers to expression data
expr_outlier <- expr1
expr_outlier[1, 1] <- 50   # Extreme high outlier
expr_outlier[2, 8] <- -30  # Extreme low outlier
expr_outlier[3, 4] <- 25   # Another outlier

write.csv(expr_outlier, "R_mrlm_expr_outlier.csv", row.names = TRUE)

fit_robust <- lmFit(expr_outlier, design1, method = "robust")
write.csv(fit_robust$coefficients, "R_mrlm_coef.csv", row.names = TRUE)
write.csv(fit_robust$stdev.unscaled, "R_mrlm_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(sigma = fit_robust$sigma, df_residual = fit_robust$df.residual),
  "R_mrlm_stats.csv",
  row.names = TRUE
)

# Compare robust vs non-robust on same data
fit_nonrobust <- lmFit(expr_outlier, design1, method = "ls")
write.csv(fit_nonrobust$coefficients, "R_mrlm_ls_coef.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# Duplicate probe averaging (avereps, avedups)
# -----------------------------------------------------------------------------
cat("  avereps / avedups...\n")

# Expression with duplicate probes (60 probes mapping to 20 genes)
set.seed(44)
n_probes <- 60
expr_probes <- matrix(rnorm(n_probes * n_samples), nrow = n_probes)
colnames(expr_probes) <- paste0("sample", 1:n_samples)
rownames(expr_probes) <- paste0("probe", 1:n_probes)

# 3 probes per gene
probe_to_gene <- rep(paste0("gene", 1:20), each = 3)

write.csv(expr_probes, "R_avereps_expr.csv", row.names = TRUE)
write.csv(data.frame(ID = probe_to_gene), "R_avereps_ids.csv", row.names = FALSE)

# avereps
expr_averaged <- avereps(expr_probes, ID = probe_to_gene)
write.csv(expr_averaged, "R_avereps_output.csv", row.names = TRUE)

# avedups with ndups and spacing
# Create data where duplicates are in consecutive rows
n_genes_dup <- 20
expr_consec <- matrix(rnorm(n_genes_dup * 2 * n_samples), nrow = n_genes_dup * 2)
rownames(expr_consec) <- paste0("spot", 1:(n_genes_dup * 2))

write.csv(expr_consec, "R_avedups_expr.csv", row.names = TRUE)

expr_avedups <- avedups(expr_consec, ndups = 2, spacing = 1)
write.csv(expr_avedups, "R_avedups_output.csv", row.names = TRUE)

# unwrapdups
expr_unwrap <- unwrapdups(expr_consec, ndups = 2, spacing = 1)
write.csv(expr_unwrap, "R_unwrapdups_output.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# eBayes with proportion parameter
# -----------------------------------------------------------------------------
cat("  eBayes (proportion parameter)...\n")

eb_prop01 <- eBayes(fit1, proportion = 0.1)
write.csv(
  data.frame(
    t_2 = eb_prop01$t[, 2],
    p_value_2 = eb_prop01$p.value[, 2],
    lods_2 = eb_prop01$lods[, 2],
    s2_post = eb_prop01$s2.post
  ),
  "R_ebayes_prop01_stats.csv",
  row.names = TRUE
)

eb_prop05 <- eBayes(fit1, proportion = 0.5)
write.csv(
  data.frame(
    t_2 = eb_prop05$t[, 2],
    p_value_2 = eb_prop05$p.value[, 2],
    lods_2 = eb_prop05$lods[, 2],
    s2_post = eb_prop05$s2.post
  ),
  "R_ebayes_prop05_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# classifyTestsF
# -----------------------------------------------------------------------------
cat("  classifyTestsF...\n")

# Use multi-contrast fit from Dataset 2
ctf_p05 <- classifyTestsF(eb2, p.value = 0.05)
write.csv(as.data.frame(ctf_p05), "R_classifytestsf_p05.csv", row.names = TRUE)

ctf_p01 <- classifyTestsF(eb2, p.value = 0.01)
write.csv(as.data.frame(ctf_p01), "R_classifytestsf_p01.csv", row.names = TRUE)

# With fstat.only = FALSE (default) vs TRUE
ctf_fstat <- classifyTestsF(eb2, p.value = 0.05, fstat.only = TRUE)
write.csv(as.data.frame(ctf_fstat), "R_classifytestsf_fstat.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# makeContrasts (direct testing)
# -----------------------------------------------------------------------------
cat("  makeContrasts...\n")

# Simple contrasts
cont_simple <- makeContrasts(
  BvsA = B - A,
  CvsA = C - A,
  levels = c("A", "B", "C")
)
write.csv(cont_simple, "R_makecontrasts_simple.csv", row.names = TRUE)

# Complex contrasts (averages, interactions)
cont_complex <- makeContrasts(
  BvsA = B - A,
  CvsA = C - A,
  CvsB = C - B,
  AvgBCvsA = (B + C)/2 - A,
  levels = c("A", "B", "C")
)
write.csv(cont_complex, "R_makecontrasts_complex.csv", row.names = TRUE)

# Contrasts from design with intercept
cont_intercept <- makeContrasts(
  groupB = groupB,
  levels = colnames(design1)
)
write.csv(cont_intercept, "R_makecontrasts_intercept.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# decideTests additional methods
# -----------------------------------------------------------------------------
cat("  decideTests (additional methods)...\n")

# method = "hierarchical"
dt_hier <- decideTests(eb2, method = "hierarchical", adjust.method = "BH", p.value = 0.05)
write.csv(as.data.frame(dt_hier), "R_decidetests_hierarchical.csv", row.names = TRUE)

# method = "nestedF"
dt_nestedF <- decideTests(eb2, method = "nestedF", adjust.method = "BH", p.value = 0.05)
write.csv(as.data.frame(dt_nestedF), "R_decidetests_nestedF.csv", row.names = TRUE)

# Different adjust methods
dt_bonf <- decideTests(eb2, method = "separate", adjust.method = "bonferroni", p.value = 0.05)
write.csv(as.data.frame(dt_bonf), "R_decidetests_bonferroni.csv", row.names = TRUE)

dt_holm <- decideTests(eb2, method = "separate", adjust.method = "holm", p.value = 0.05)
write.csv(as.data.frame(dt_holm), "R_decidetests_holm.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# topTable additional edge cases
# -----------------------------------------------------------------------------
cat("  topTable (additional edge cases)...\n")

# resort.by parameter
tt_resortP <- topTable(eb1, coef = 2, number = 20, sort.by = "B", resort.by = "P")
write.csv(tt_resortP, "R_toptable_resortP.csv", row.names = TRUE)

# lfc filter
tt_lfc <- topTable(eb1, coef = 2, number = Inf, lfc = 1)
write.csv(tt_lfc, "R_toptable_lfc1.csv", row.names = TRUE)

# adjust.method variations
tt_bonf <- topTable(eb1, coef = 2, number = 20, adjust.method = "bonferroni")
write.csv(tt_bonf, "R_toptable_bonferroni.csv", row.names = TRUE)

tt_none <- topTable(eb1, coef = 2, number = 20, adjust.method = "none")
write.csv(tt_none, "R_toptable_noadjust.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# squeezeVar edge cases
# -----------------------------------------------------------------------------
cat("  squeezeVar (edge cases)...\n")

# Very small df
small_df <- rep(2, n_genes)
sv_smalldf <- squeezeVar(fit1$sigma^2, small_df)
write.csv(
  data.frame(
    var_post = sv_smalldf$var.post,
    var_prior = sv_smalldf$var.prior,
    df_prior = sv_smalldf$df.prior
  ),
  "R_squeezevar_smalldf.csv",
  row.names = TRUE
)

# Mixed df (some genes with more df than others)
mixed_df <- c(rep(3, 25), rep(6, 25))
sv_mixeddf <- squeezeVar(fit1$sigma^2, mixed_df)
write.csv(
  data.frame(
    var_post = sv_mixeddf$var.post,
    var_prior = sv_mixeddf$var.prior,
    df_prior = sv_mixeddf$df.prior
  ),
  "R_squeezevar_mixeddf.csv",
  row.names = TRUE
)

# With winsor.tail.p
sv_winsor <- squeezeVar(fit1$sigma^2, fit1$df.residual, robust = TRUE, winsor.tail.p = c(0.01, 0.01))
write.csv(
  data.frame(
    var_post = sv_winsor$var.post,
    var_prior = sv_winsor$var.prior,
    df_prior = sv_winsor$df.prior
  ),
  "R_squeezevar_winsor.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# treat edge cases
# -----------------------------------------------------------------------------
cat("  treat (edge cases)...\n")

# treat with trend
treat_trend <- treat(fit1, lfc = 0.5, trend = TRUE)
write.csv(
  data.frame(
    t_2 = treat_trend$t[, 2],
    p_value_2 = treat_trend$p.value[, 2],
    s2_post = treat_trend$s2.post
  ),
  "R_treat_trend_stats.csv",
  row.names = TRUE
)

# treat with robust
treat_robust <- treat(fit1, lfc = 0.5, robust = TRUE)
write.csv(
  data.frame(
    t_2 = treat_robust$t[, 2],
    p_value_2 = treat_robust$p.value[, 2],
    s2_post = treat_robust$s2.post
  ),
  "R_treat_robust_stats.csv",
  row.names = TRUE
)

# -----------------------------------------------------------------------------
# eBayes + topTable workflow with contrasts (full pipeline)
# -----------------------------------------------------------------------------
cat("  Full pipeline with contrasts...\n")

# Three-way comparison pipeline
fit2_eb <- eBayes(fit2_cont)
tt2_BvsA <- topTable(fit2_eb, coef = "BvsA", number = Inf)
tt2_CvsA <- topTable(fit2_eb, coef = "CvsA", number = Inf)
tt2_CvsB <- topTable(fit2_eb, coef = "CvsB", number = Inf)

write.csv(tt2_BvsA, "R_pipeline_toptable_BvsA.csv", row.names = TRUE)
write.csv(tt2_CvsA, "R_pipeline_toptable_CvsA.csv", row.names = TRUE)
write.csv(tt2_CvsB, "R_pipeline_toptable_CvsB.csv", row.names = TRUE)

# F-test across all contrasts
tt2_F <- topTable(fit2_eb, coef = NULL, number = Inf)
write.csv(tt2_F, "R_pipeline_toptable_F.csv", row.names = TRUE)

# -----------------------------------------------------------------------------
# voom and arrayWeights fixtures
# -----------------------------------------------------------------------------
cat("\nGenerating voom and arrayWeights fixtures...\n")

# RNA-seq count data (simulate realistic counts)
set.seed(45)
n_genes_voom <- 100
n_samples_voom <- 8

# Simulate counts with NB distribution
counts <- matrix(rnbinom(n_genes_voom * n_samples_voom, mu = 500, size = 10), nrow = n_genes_voom)
rownames(counts) <- paste0("gene", 1:n_genes_voom)
colnames(counts) <- paste0("sample", 1:n_samples_voom)

# Add differential expression to first 20 genes
counts[1:20, 5:8] <- counts[1:20, 5:8] * 3

group_voom <- factor(rep(c("A", "B"), each = 4))
design_voom <- model.matrix(~ group_voom)
colnames(design_voom) <- c("Intercept", "groupB")

write.csv(counts, "R_voom_counts.csv", row.names = TRUE)
write.csv(design_voom, "R_voom_design.csv", row.names = FALSE)

# Basic voom
cat("  voom (basic)...\n")
v <- voom(counts, design_voom, plot = FALSE)
write.csv(v$E, "R_voom_E.csv", row.names = TRUE)
write.csv(v$weights, "R_voom_weights.csv", row.names = TRUE)
write.csv(data.frame(lib.size = v$targets$lib.size), "R_voom_libsize.csv", row.names = FALSE)

# voom with custom lib.size
cat("  voom (custom lib.size)...\n")
lib_size_custom <- colSums(counts) * runif(n_samples_voom, 0.9, 1.1)
write.csv(data.frame(lib.size = lib_size_custom), "R_voom_customlib_input.csv", row.names = FALSE)
v_libsize <- voom(counts, design_voom, lib.size = lib_size_custom, plot = FALSE)
write.csv(v_libsize$E, "R_voom_customlib_E.csv", row.names = TRUE)
write.csv(v_libsize$weights, "R_voom_customlib_weights.csv", row.names = TRUE)

# voom -> lmFit -> eBayes pipeline
cat("  voom pipeline...\n")
fit_v <- lmFit(v, design_voom)
fit_v <- eBayes(fit_v)
write.csv(
  data.frame(
    coef = fit_v$coefficients[, 2],
    t = fit_v$t[, 2],
    p_value = fit_v$p.value[, 2]
  ),
  "R_voom_pipeline_stats.csv",
  row.names = TRUE
)

# arrayWeights
cat("  arrayWeights...\n")
aw <- arrayWeights(v, design_voom)
write.csv(data.frame(weight = aw), "R_arrayweights_basic.csv", row.names = FALSE)

# arrayWeights with var.group
cat("  arrayWeights (var.group)...\n")
var_group <- factor(rep(c("high", "low"), each = 4))
aw_vargroup <- arrayWeights(v, design_voom, var.group = var_group)
write.csv(data.frame(weight = aw_vargroup), "R_arrayweights_vargroup.csv", row.names = FALSE)

# arrayWeights with method="reml"
# Pass raw log-expression (no voom weights) so REML path is available
cat("  arrayWeights (method=reml)...\n")
expr_reml <- log2(counts + 1)
aw_reml <- arrayWeights(expr_reml, design_voom, method = "reml")
write.csv(data.frame(weight = aw_reml), "R_arrayweights_reml.csv", row.names = FALSE)

# voomWithQualityWeights
cat("  voomWithQualityWeights...\n")
vwq <- voomWithQualityWeights(counts, design_voom, plot = FALSE)
write.csv(vwq$E, "R_voomqw_E.csv", row.names = TRUE)
write.csv(vwq$weights, "R_voomqw_weights.csv", row.names = TRUE)
write.csv(data.frame(sample.weights = vwq$targets$sample.weights), "R_voomqw_sampleweights.csv", row.names = FALSE)

# vooma (for expression data, not counts)
cat("  vooma...\n")
# Use log-transformed data (simulating microarray)
expr_vooma <- log2(counts + 1)
va <- vooma(expr_vooma, design_voom, plot = FALSE)
write.csv(va$E, "R_vooma_E.csv", row.names = TRUE)
write.csv(va$weights, "R_vooma_weights.csv", row.names = TRUE)

# vooma with legacy span
cat("  vooma (legacy span)...\n")
va_legacy <- vooma(expr_vooma, design_voom, legacy.span = TRUE, plot = FALSE)
write.csv(va_legacy$weights, "R_vooma_legacyspan_weights.csv", row.names = TRUE)
write.csv(data.frame(span = va_legacy$span),
          "R_vooma_legacyspan_span.csv", row.names = FALSE)

# vooma with predictor (precision predictor per gene x sample)
cat("  vooma (predictor)...\n")
set.seed(789)
predictor_vooma <- matrix(
  rnorm(n_genes_voom * n_samples_voom, mean = 0, sd = 0.2),
  nrow = n_genes_voom, ncol = n_samples_voom
)
rownames(predictor_vooma) <- rownames(expr_vooma)
colnames(predictor_vooma) <- colnames(expr_vooma)
va_pred <- vooma(expr_vooma, design_voom, predictor = predictor_vooma, plot = FALSE)
write.csv(predictor_vooma, "R_vooma_predictor_input.csv", row.names = TRUE)
write.csv(va_pred$weights, "R_vooma_predictor_weights.csv", row.names = TRUE)
write.csv(data.frame(span = va_pred$span),
          "R_vooma_predictor_span.csv", row.names = FALSE)

# vooma with block correlation
cat("  vooma (with block)...\n")
block_vooma <- factor(rep(1:4, each = 2))
dupcor_vooma <- duplicateCorrelation(expr_vooma, design_voom, block = block_vooma)
va_block <- vooma(expr_vooma, design_voom, block = block_vooma,
                  correlation = dupcor_vooma$consensus.correlation, plot = FALSE)
write.csv(va_block$weights, "R_vooma_block_weights.csv", row.names = TRUE)
write.csv(data.frame(consensus_correlation = dupcor_vooma$consensus.correlation),
          "R_vooma_block_consensus.csv", row.names = FALSE)

# voom with prior weights
cat("  voom (with prior weights)...\n")
set.seed(123)
prior_weights <- matrix(runif(n_genes_voom * n_samples_voom, 0.5, 1.5),
                        nrow = n_genes_voom, ncol = n_samples_voom)
rownames(prior_weights) <- rownames(counts)
colnames(prior_weights) <- colnames(counts)
v_weighted <- voom(counts, design_voom, weights = prior_weights, plot = FALSE)
write.csv(prior_weights, "R_voom_priorweights_input.csv", row.names = TRUE)
write.csv(v_weighted$weights, "R_voom_priorweights_output.csv", row.names = TRUE)

# voom with offset - manual computation since limma 3.66.0 doesn't have offset parameter
# This replicates the logic from limma >3.67 voom() with offset support
cat("  voom (with offset) - manual computation...\n")
set.seed(456)
offset_matrix <- matrix(rnorm(n_genes_voom * n_samples_voom, 0, 0.5),
                        nrow = n_genes_voom, ncol = n_samples_voom)
rownames(offset_matrix) <- rownames(counts)
colnames(offset_matrix) <- colnames(counts)

# Compute offset_prior = offset - rowMeans(offset)
offset_prior_matrix <- offset_matrix - rowMeans(offset_matrix)

# Compute adjusted lib.size.matrix with offset_prior
lib_size <- colSums(counts)
lib_size_matrix <- matrix(lib_size, nrow = n_genes_voom, ncol = n_samples_voom, byrow = TRUE)
lib_size_matrix_adj <- exp(log(lib_size_matrix) + offset_prior_matrix)

# Compute log-CPM with adjusted library sizes
E_offset <- log2((counts + 0.5) / (lib_size_matrix_adj + 1) * 1e6)

# Fit linear model to get residuals for variance estimation
fit_offset <- lmFit(E_offset, design_voom)

# Get span using same adaptive method
span_offset <- chooseLowessSpan(n_genes_voom, small.n = 50, min.span = 0.3, power = 1/3)

# Compute sx, sy for lowess
sx_offset <- fit_offset$Amean + mean(log2(lib_size + 1)) - log2(1e6)
sy_offset <- sqrt(fit_offset$sigma)

# Fit lowess
l_offset <- lowess(sx_offset, sy_offset, f = span_offset)

# Interpolation function
f_offset <- approxfun(l_offset, rule = 2, ties = list("ordered", mean))

# Compute fitted values and weights
if (fit_offset$rank < ncol(design_voom)) {
  j <- fit_offset$pivot[1:fit_offset$rank]
  fitted_values <- fit_offset$coefficients[, j, drop = FALSE] %*% t(fit_offset$design[, j, drop = FALSE])
} else {
  fitted_values <- fit_offset$coefficients %*% t(fit_offset$design)
}
fitted_cpm <- 2^fitted_values
fitted_count <- 1e-6 * fitted_cpm * (lib_size_matrix_adj + 1)
fitted_logcount <- log2(fitted_count)
weights_offset <- 1 / f_offset(fitted_logcount)^4
dim(weights_offset) <- dim(fitted_logcount)
rownames(weights_offset) <- rownames(counts)
colnames(weights_offset) <- colnames(counts)

write.csv(offset_matrix, "R_voom_offset_input.csv", row.names = TRUE)
write.csv(E_offset, "R_voom_offset_E.csv", row.names = TRUE)
write.csv(weights_offset, "R_voom_offset_weights.csv", row.names = TRUE)

# voom with offset.prior directly (same result since offset_prior = offset - rowMeans)
cat("  voom (with offset.prior)...\n")
write.csv(offset_prior_matrix, "R_voom_offsetprior_input.csv", row.names = TRUE)
write.csv(E_offset, "R_voom_offsetprior_E.csv", row.names = TRUE)
write.csv(weights_offset, "R_voom_offsetprior_weights.csv", row.names = TRUE)

# voom with block correlation
# Estimate intra-block correlation on the voom-transformed data, then re-run
# voom with block+correlation so weights reflect the correlated design.
cat("  voom (with block)...\n")
block_voom <- factor(rep(1:4, each = 2))
dupcor_voom <- duplicateCorrelation(v, design_voom, block = block_voom)
v_block <- voom(counts, design_voom, block = block_voom,
                correlation = dupcor_voom$consensus.correlation, plot = FALSE)
write.csv(v_block$E, "R_voom_block_E.csv", row.names = TRUE)
write.csv(v_block$weights, "R_voom_block_weights.csv", row.names = TRUE)
write.csv(data.frame(consensus_correlation = dupcor_voom$consensus.correlation),
          "R_voom_block_consensus.csv", row.names = FALSE)

# voom with no replication (edge case)
cat("  voom (no replication edge case)...\n")
design_norep <- matrix(1, nrow = n_samples_voom, ncol = n_samples_voom)
diag(design_norep) <- 1
colnames(design_norep) <- paste0("sample", 1:n_samples_voom)
v_norep <- suppressWarnings(voom(counts, design_norep, plot = FALSE))
write.csv(v_norep$weights, "R_voom_norep_weights.csv", row.names = TRUE)

# Debug: save intermediate voom values for comparison
cat("  voom (debug intermediates)...\n")
v_debug <- voom(counts, design_voom, plot = FALSE, save.plot = TRUE)
write.csv(data.frame(x = v_debug$voom.xy$x, y = v_debug$voom.xy$y),
          "R_voom_debug_xy.csv", row.names = FALSE)
write.csv(data.frame(x = v_debug$voom.line$x, y = v_debug$voom.line$y),
          "R_voom_debug_line.csv", row.names = FALSE)
write.csv(data.frame(span = v_debug$span), "R_voom_debug_span.csv", row.names = FALSE)

# voomaLmFit tests
cat("  voomaLmFit (basic)...\n")
vlf_basic <- voomaLmFit(expr_vooma, design_voom, plot = FALSE)
write.csv(vlf_basic$coefficients, "R_voomalmfit_basic_coef.csv", row.names = TRUE)
write.csv(vlf_basic$sigma, "R_voomalmfit_basic_sigma.csv", row.names = TRUE)
write.csv(data.frame(span = vlf_basic$span), "R_voomalmfit_basic_span.csv", row.names = FALSE)

# voomaLmFit with sample weights
cat("  voomaLmFit (with sample weights)...\n")
vlf_sw <- voomaLmFit(expr_vooma, design_voom, sample.weights = TRUE, plot = FALSE)
write.csv(vlf_sw$coefficients, "R_voomalmfit_sw_coef.csv", row.names = TRUE)
write.csv(vlf_sw$sigma, "R_voomalmfit_sw_sigma.csv", row.names = TRUE)
write.csv(data.frame(sample.weights = vlf_sw$targets$sample.weight),
          "R_voomalmfit_sw_sampleweights.csv", row.names = FALSE)

# voomaLmFit with block correlation
cat("  voomaLmFit (with block)...\n")
vlf_block <- voomaLmFit(expr_vooma, design_voom, block = block_vooma, plot = FALSE)
write.csv(vlf_block$coefficients, "R_voomalmfit_block_coef.csv", row.names = TRUE)
write.csv(vlf_block$sigma, "R_voomalmfit_block_sigma.csv", row.names = TRUE)

cat("  voom and arrayWeights fixtures complete.\n")

# -----------------------------------------------------------------------------
# voom / arrayWeights branch coverage: forcing fixtures for previously untested branches
# -----------------------------------------------------------------------------
cat("\nGenerating voom / arrayWeights branch-coverage fixtures...\n")

# voom with adaptive.span=FALSE and an explicit span value
cat("  voom (explicit span)...\n")
v_explicitspan <- voom(counts, design_voom, span = 0.4, adaptive.span = FALSE,
                       plot = FALSE)
write.csv(v_explicitspan$weights, "R_voom_explicitspan_weights.csv",
          row.names = TRUE)

# voom with all-zero rows present (forces the allzero filter)
cat("  voom (allzero rows)...\n")
counts_allzero <- counts
counts_allzero[1:5, ] <- 0L
v_allzero <- voom(counts_allzero, design_voom, plot = FALSE)
write.csv(counts_allzero, "R_voom_allzero_counts.csv", row.names = TRUE)
write.csv(v_allzero$weights, "R_voom_allzero_weights.csv", row.names = TRUE)

# voom with rank-deficient design (forces the pivot branch)
cat("  voom (rank-deficient design)...\n")
design_rankdef <- cbind(design_voom, redundant = design_voom[, 1] + design_voom[, 2])
v_rankdef <- voom(counts, design_rankdef, plot = FALSE)
write.csv(design_rankdef, "R_voom_rankdef_design.csv", row.names = FALSE)
write.csv(v_rankdef$weights, "R_voom_rankdef_weights.csv", row.names = TRUE)

# vooma with 1-column predictor matrix (single per-gene predictor, replicated)
cat("  vooma (1-col predictor)...\n")
set.seed(2026)
predictor_1col <- matrix(rnorm(n_genes_voom, mean = 0, sd = 0.2), ncol = 1)
rownames(predictor_1col) <- rownames(expr_vooma)
va_pred1 <- vooma(expr_vooma, design_voom, predictor = predictor_1col,
                  plot = FALSE)
write.csv(predictor_1col, "R_vooma_predictor1col_input.csv", row.names = TRUE)
write.csv(va_pred1$weights, "R_vooma_predictor1col_weights.csv",
          row.names = TRUE)

# voomaLmFit with legacy.span = TRUE
cat("  voomaLmFit (legacy span)...\n")
vlf_legacy <- voomaLmFit(expr_vooma, design_voom, legacy.span = TRUE,
                         plot = FALSE)
write.csv(vlf_legacy$coefficients, "R_voomalmfit_legacyspan_coef.csv",
          row.names = TRUE)
write.csv(vlf_legacy$sigma, "R_voomalmfit_legacyspan_sigma.csv",
          row.names = TRUE)
write.csv(data.frame(span = vlf_legacy$span),
          "R_voomalmfit_legacyspan_span.csv", row.names = FALSE)

# voomaLmFit with both block and sample.weights
cat("  voomaLmFit (block + sample weights)...\n")
vlf_blocksw <- voomaLmFit(expr_vooma, design_voom, block = block_vooma,
                          sample.weights = TRUE, plot = FALSE)
write.csv(vlf_blocksw$coefficients, "R_voomalmfit_blocksw_coef.csv",
          row.names = TRUE)
write.csv(vlf_blocksw$sigma, "R_voomalmfit_blocksw_sigma.csv",
          row.names = TRUE)
write.csv(data.frame(sample.weights = vlf_blocksw$targets$sample.weight),
          "R_voomalmfit_blocksw_sampleweights.csv", row.names = FALSE)

# voomaLmFit with prior weights only (no sample weights, no block)
cat("  voomaLmFit (prior weights)...\n")
set.seed(7321)
priorw_vlf <- matrix(runif(n_genes_voom * n_samples_voom, 0.5, 1.5),
                     nrow = n_genes_voom, ncol = n_samples_voom)
rownames(priorw_vlf) <- rownames(expr_vooma)
colnames(priorw_vlf) <- colnames(expr_vooma)
vlf_priorw <- voomaLmFit(expr_vooma, design_voom, prior.weights = priorw_vlf,
                         plot = FALSE)
write.csv(priorw_vlf, "R_voomalmfit_priorw_input.csv", row.names = TRUE)
write.csv(vlf_priorw$coefficients, "R_voomalmfit_priorw_coef.csv",
          row.names = TRUE)
write.csv(vlf_priorw$sigma, "R_voomalmfit_priorw_sigma.csv",
          row.names = TRUE)

# voomWithQualityWeights with var.design (centered indicator matrix)
cat("  voomWithQualityWeights (var.design)...\n")
vd_vwq <- model.matrix(~ group_voom)[, -1, drop = FALSE]
vwq_vd <- voomWithQualityWeights(counts, design_voom, var.design = vd_vwq,
                                 plot = FALSE)
write.csv(vwq_vd$weights, "R_voomqw_vardesign_weights.csv", row.names = TRUE)
write.csv(data.frame(sample.weights = vwq_vd$targets$sample.weights),
          "R_voomqw_vardesign_sampleweights.csv", row.names = FALSE)

# voomWithQualityWeights with var.group
cat("  voomWithQualityWeights (var.group)...\n")
vwq_vg <- voomWithQualityWeights(counts, design_voom,
                                 var.group = var_group, plot = FALSE)
write.csv(vwq_vg$weights, "R_voomqw_vargroup_weights.csv", row.names = TRUE)
write.csv(data.frame(sample.weights = vwq_vg$targets$sample.weights),
          "R_voomqw_vargroup_sampleweights.csv", row.names = FALSE)

# voomWithQualityWeights with method="reml"
cat("  voomWithQualityWeights (method=reml)...\n")
vwq_reml <- voomWithQualityWeights(counts, design_voom, method = "reml",
                                   plot = FALSE)
write.csv(vwq_reml$weights, "R_voomqw_reml_weights.csv", row.names = TRUE)
write.csv(data.frame(sample.weights = vwq_reml$targets$sample.weights),
          "R_voomqw_reml_sampleweights.csv", row.names = FALSE)

# arrayWeights with rank-deficient design (forces design column drop branch)
cat("  arrayWeights (rank-deficient design)...\n")
aw_rankdef <- arrayWeights(expr_reml, design_rankdef, method = "reml")
write.csv(data.frame(weight = aw_rankdef),
          "R_arrayweights_rankdef.csv", row.names = FALSE)

# arrayWeights with explicit var.design (not via var.group)
cat("  arrayWeights (var.design)...\n")
aw_vd <- arrayWeights(v, design_voom, var.design = vd_vwq)
write.csv(data.frame(weight = aw_vd),
          "R_arrayweights_vardesign.csv", row.names = FALSE)

# arrayWeights with NA values in E (genebygene NA branch)
cat("  arrayWeights (NA in E)...\n")
expr_na <- log2(counts + 1)
expr_na[1, 3] <- NA
expr_na[5, c(2, 7)] <- NA
expr_na[10, 1] <- NA
write.csv(expr_na, "R_arrayweights_na_input.csv", row.names = TRUE)
aw_na <- arrayWeights(expr_na, design_voom)
write.csv(data.frame(weight = aw_na),
          "R_arrayweights_na.csv", row.names = FALSE)

# arrayWeights method="auto" with raw expression (forces auto -> reml dispatch)
cat("  arrayWeights (method=auto -> reml)...\n")
aw_autoreml <- arrayWeights(expr_reml, design_voom, method = "auto")
write.csv(data.frame(weight = aw_autoreml),
          "R_arrayweights_autoreml.csv", row.names = FALSE)

# arrayWeights method="reml" with NA row removal
cat("  arrayWeights (method=reml + NA)...\n")
suppressMessages(
  aw_remlna <- arrayWeights(expr_na, design_voom, method = "reml")
)
write.csv(data.frame(weight = aw_remlna),
          "R_arrayweights_remlna.csv", row.names = FALSE)

# chooseLowessSpan direct evaluation across an n grid (default + legacy params)
cat("  chooseLowessSpan (default + legacy grids)...\n")
n_grid <- c(20, 50, 100, 200, 500, 1000, 5000, 20000)
span_default <- sapply(n_grid, function(n)
  chooseLowessSpan(n, small.n = 50, min.span = 0.3, power = 1/3))
span_legacy <- sapply(n_grid, function(n)
  chooseLowessSpan(n, small.n = 10, min.span = 0.3, power = 0.5))
write.csv(data.frame(n = n_grid, span = span_default),
          "R_chooselowess_default.csv", row.names = FALSE)
write.csv(data.frame(n = n_grid, span = span_legacy),
          "R_chooselowess_legacy.csv", row.names = FALSE)

cat("  voom / arrayWeights branch-coverage fixtures complete.\n")

# -----------------------------------------------------------------------------
# arrayWeights / voom bug-fix forcing fixtures
# -----------------------------------------------------------------------------
cat("\nGenerating arrayWeights / voom bug-fix fixtures...\n")

# arrayWeights(method="reml", weights=W) -> .arrayWeightsPrWtsREML
cat("  arrayWeights (method=reml + prior weights)...\n")
aw_remlweights <- arrayWeights(v, design_voom, method = "reml")
write.csv(data.frame(weight = aw_remlweights),
          "R_arrayweights_remlweights.csv", row.names = FALSE)

# arrayWeightsQuick(y, fit) -- approximate array quality weights
cat("  arrayWeightsQuick...\n")
fit_for_quick <- lmFit(expr_vooma, design_voom)
aw_quick <- arrayWeightsQuick(expr_vooma, fit_for_quick)
write.csv(data.frame(weight = aw_quick),
          "R_arrayweightsquick.csv", row.names = FALSE)

# voomaLmFit with predictor (first-iteration only -- no block, no sample weights)
cat("  voomaLmFit (predictor)...\n")
vlf_pred <- voomaLmFit(expr_vooma, design_voom, predictor = predictor_vooma,
                       plot = FALSE)
write.csv(vlf_pred$coefficients, "R_voomalmfit_predictor_coef.csv",
          row.names = TRUE)
write.csv(vlf_pred$sigma, "R_voomalmfit_predictor_sigma.csv",
          row.names = TRUE)
write.csv(data.frame(span = vlf_pred$span),
          "R_voomalmfit_predictor_span.csv", row.names = FALSE)

# voomaLmFit with predictor + block (drives the second-iteration predictor branch)
cat("  voomaLmFit (predictor + block)...\n")
vlf_pred_block <- voomaLmFit(expr_vooma, design_voom,
                             predictor = predictor_vooma,
                             block = block_vooma, plot = FALSE)
write.csv(vlf_pred_block$coefficients,
          "R_voomalmfit_predictor_block_coef.csv", row.names = TRUE)
write.csv(vlf_pred_block$sigma,
          "R_voomalmfit_predictor_block_sigma.csv", row.names = TRUE)

cat("  arrayWeights / voom bug-fix fixtures complete.\n")

# -----------------------------------------------------------------------------
# normalizeBetweenArrays direct + via voom
# -----------------------------------------------------------------------------
cat("\nGenerating normalizeBetweenArrays fixtures...\n")

# Build a deterministic log-expression matrix with realistic per-column shifts
set.seed(202604)
n_norm_genes <- 200
n_norm_samples <- 8
norm_input <- matrix(rnorm(n_norm_genes * n_norm_samples, mean = 7, sd = 1.5),
                     nrow = n_norm_genes, ncol = n_norm_samples)
# Inject inter-sample shifts so the methods actually do something meaningful
norm_input <- t(t(norm_input) + c(-0.4, -0.2, 0, 0.1, 0.3, 0.5, -0.1, 0.2))
# Inject per-sample scale variation
norm_input <- t(t(norm_input) * c(1.0, 1.05, 0.95, 1.02, 0.98, 1.08, 0.92, 1.0))
rownames(norm_input) <- paste0("g", 1:n_norm_genes)
colnames(norm_input) <- paste0("s", 1:n_norm_samples)
write.csv(norm_input, "R_norm_input.csv", row.names = TRUE)

# Direct normalizeBetweenArrays per method
cat("  normalizeBetweenArrays (none/scale/quantile/cyclicloess)...\n")
norm_none        <- normalizeBetweenArrays(norm_input, method = "none")
norm_scale       <- normalizeBetweenArrays(norm_input, method = "scale")
norm_quantile    <- normalizeBetweenArrays(norm_input, method = "quantile")
norm_cyclicloess <- normalizeBetweenArrays(norm_input, method = "cyclicloess")
write.csv(norm_none,        "R_norm_none.csv",        row.names = TRUE)
write.csv(norm_scale,       "R_norm_scale.csv",       row.names = TRUE)
write.csv(norm_quantile,    "R_norm_quantile.csv",    row.names = TRUE)
write.csv(norm_cyclicloess, "R_norm_cyclicloess.csv", row.names = TRUE)

# voom() with each non-default normalize.method (re-uses the voom RNA-seq data)
cat("  voom with normalize.method = scale/quantile/cyclicloess...\n")
v_normscale       <- voom(counts, design_voom, normalize.method = "scale",
                          plot = FALSE)
v_normquantile    <- voom(counts, design_voom, normalize.method = "quantile",
                          plot = FALSE)
v_normcyclicloess <- voom(counts, design_voom,
                          normalize.method = "cyclicloess", plot = FALSE)
write.csv(v_normscale$E,          "R_voom_normscale_E.csv",       row.names = TRUE)
write.csv(v_normscale$weights,    "R_voom_normscale_weights.csv", row.names = TRUE)
write.csv(v_normquantile$E,       "R_voom_normquantile_E.csv",       row.names = TRUE)
write.csv(v_normquantile$weights, "R_voom_normquantile_weights.csv", row.names = TRUE)
write.csv(v_normcyclicloess$E,       "R_voom_normcyclicloess_E.csv",       row.names = TRUE)
write.csv(v_normcyclicloess$weights, "R_voom_normcyclicloess_weights.csv", row.names = TRUE)

cat("  normalizeBetweenArrays fixtures complete.\n")

# -----------------------------------------------------------------------------
# model.matrix fixtures
# -----------------------------------------------------------------------------
cat("\nGenerating model.matrix fixtures...\n")

set.seed(456)
mm_data <- data.frame(
  group = factor(rep(c("A", "B", "C"), each = 4)),
  batch = factor(rep(c("X", "Y"), 6)),
  age = rnorm(12, mean = 50, sd = 10)
)
write.csv(mm_data, "R_modelmatrix_data.csv", row.names = FALSE)

# Intercept model (reference coding)
cat("  model.matrix (intercept)...\n")
mm_intercept <- model.matrix(~ group, data = mm_data)
write.csv(mm_intercept, "R_modelmatrix_intercept.csv", row.names = TRUE)

# Cell-means (no intercept)
cat("  model.matrix (cell-means)...\n")
mm_cellmeans <- model.matrix(~ 0 + group, data = mm_data)
write.csv(mm_cellmeans, "R_modelmatrix_cellmeans.csv", row.names = TRUE)

# Multi-factor
cat("  model.matrix (multi-factor)...\n")
mm_multifactor <- model.matrix(~ group + batch, data = mm_data)
write.csv(mm_multifactor, "R_modelmatrix_multifactor.csv", row.names = TRUE)

# With numeric covariate
cat("  model.matrix (numeric covariate)...\n")
mm_numeric <- model.matrix(~ group + age, data = mm_data)
write.csv(mm_numeric, "R_modelmatrix_numeric.csv", row.names = TRUE)

cat("  model.matrix fixtures complete.\n")

# -----------------------------------------------------------------------------
# Branch coverage fixtures: untested sort_by / adjust_method,
# decide_tests method="separate", asymmetric winsor.tail.p, nonEstimable,
# p.adjust hochberg
# -----------------------------------------------------------------------------
cat("\n--- Branch coverage fixtures (topTable, decideTests, eBayes, p.adjust) ---\n")

# topTable alternative sort_by values on eb1 (data1)
cat("  topTable sort_by=t / logFC / AveExpr / none ...\n")
tt_sort_t    <- topTable(eb1, coef = 2, number = Inf, sort.by = "t")
tt_sort_lfc  <- topTable(eb1, coef = 2, number = Inf, sort.by = "logFC")
tt_sort_A    <- topTable(eb1, coef = 2, number = Inf, sort.by = "AveExpr")
tt_sort_none <- topTable(eb1, coef = 2, number = Inf, sort.by = "none")
write.csv(tt_sort_t,    "R_toptable_sort_t.csv",       row.names = TRUE)
write.csv(tt_sort_lfc,  "R_toptable_sort_logfc.csv",   row.names = TRUE)
write.csv(tt_sort_A,    "R_toptable_sort_aveexpr.csv", row.names = TRUE)
write.csv(tt_sort_none, "R_toptable_sort_none.csv",    row.names = TRUE)

# topTable alternative adjust_method values on eb1
cat("  topTable adjust=BY / holm / hommel / hochberg ...\n")
tt_adj_by   <- topTable(eb1, coef = 2, number = Inf, sort.by = "P", adjust.method = "BY")
tt_adj_holm <- topTable(eb1, coef = 2, number = Inf, sort.by = "P", adjust.method = "holm")
tt_adj_homm <- topTable(eb1, coef = 2, number = Inf, sort.by = "P", adjust.method = "hommel")
tt_adj_hoch <- topTable(eb1, coef = 2, number = Inf, sort.by = "P", adjust.method = "hochberg")
write.csv(tt_adj_by,   "R_toptable_adjust_by.csv",       row.names = TRUE)
write.csv(tt_adj_holm, "R_toptable_adjust_holm.csv",     row.names = TRUE)
write.csv(tt_adj_homm, "R_toptable_adjust_hommel.csv",   row.names = TRUE)
write.csv(tt_adj_hoch, "R_toptable_adjust_hochberg.csv", row.names = TRUE)

# decideTests method = "separate" (the default)
cat("  decideTests method=separate ...\n")
dt_sep <- decideTests(eb1, method = "separate", adjust.method = "BH", p.value = 0.05)
write.csv(as.matrix(dt_sep), "R_decidetests_separate_data1.csv", row.names = TRUE)

# fitFDistRobustly with asymmetric winsor.tail.p
cat("  squeezeVar robust with asymmetric winsor.tail.p ...\n")
sv_asym <- squeezeVar(fit1$sigma^2, df = fit1$df.residual,
                      robust = TRUE, winsor.tail.p = c(0.01, 0.10))
write.csv(
  data.frame(
    var_post = sv_asym$var.post,
    df_prior = sv_asym$df.prior
  ),
  "R_squeezevar_asym_winsor.csv", row.names = FALSE
)
write.csv(
  data.frame(var_prior = sv_asym$var.prior),
  "R_squeezevar_asym_winsor_global.csv", row.names = FALSE
)

# nonEstimable on rank-deficient design
cat("  nonEstimable (rank-deficient) ...\n")
# Intercept + two columns that sum to 1: redundant, one column dropped.
ne_design <- cbind(
  intercept = rep(1, 6),
  a         = c(0, 0, 1, 1, 0, 0),
  b         = c(1, 1, 0, 0, 1, 1)
)
ne_result <- nonEstimable(ne_design)
ne_df <- data.frame(
  nonestimable = if (is.null(ne_result)) NA_character_ else ne_result
)
write.csv(ne_df, "R_nonestimable_rankdef.csv", row.names = FALSE)
write.csv(ne_design, "R_nonestimable_design.csv", row.names = FALSE)

# p.adjust vectors across all R methods on a shared input
cat("  p.adjust vectors (all methods) ...\n")
p_input <- c(0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
             0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9)
p_adj_all <- data.frame(
  p          = p_input,
  BH         = p.adjust(p_input, "BH"),
  BY         = p.adjust(p_input, "BY"),
  holm       = p.adjust(p_input, "holm"),
  hochberg   = p.adjust(p_input, "hochberg"),
  hommel     = p.adjust(p_input, "hommel"),
  bonferroni = p.adjust(p_input, "bonferroni")
)
write.csv(p_adj_all, "R_padjust_all_methods.csv", row.names = FALSE)

cat("  Branch coverage fixtures complete.\n")

# -----------------------------------------------------------------------------
# Higher-risk branch coverage: slow fitting paths and eBayes interactions
#   - lm.series slow path (NAs + probe-specific weights)
#   - gls.series slow path (NAs + probe weights + block)
#   - e_bayes trend=TRUE AND robust=TRUE (interaction)
#   - mrlm bisquare via MASS::rlm
# -----------------------------------------------------------------------------
cat("\n--- Higher-risk branch coverage fixtures (lm.series, gls.series, eBayes, mrlm) ---\n")

# lm.series slow path: NAs in expression AND probe-specific weights
cat("  lm.series slow path (NAs + probe weights) ...\n")
set.seed(101)
nb_genes <- 30
nb_samples <- 8
expr_slow <- matrix(rnorm(nb_genes * nb_samples), nrow = nb_genes)
rownames(expr_slow) <- paste0("gene", 1:nb_genes)
# Introduce scattered NAs
expr_slow[cbind(c(1, 3, 7, 12, 20), c(2, 5, 1, 4, 8))] <- NA
# Add group effect so fits are non-trivial
expr_slow[1:10, 5:8] <- expr_slow[1:10, 5:8] + 2

# Probe-specific (per-gene) weights
weights_slow <- matrix(runif(nb_genes * nb_samples, min = 0.5, max = 2.0),
                       nrow = nb_genes)

group_slow <- factor(rep(c("A", "B"), each = nb_samples / 2))
design_slow <- model.matrix(~ group_slow)

fit_slow <- lmFit(expr_slow, design = design_slow, weights = weights_slow)

write.csv(expr_slow, "R_lmseries_slow_expr.csv", row.names = TRUE)
write.csv(weights_slow, "R_lmseries_slow_weights.csv", row.names = TRUE)
write.csv(design_slow, "R_lmseries_slow_design.csv", row.names = FALSE)
write.csv(fit_slow$coefficients, "R_lmseries_slow_coef.csv", row.names = TRUE)
write.csv(fit_slow$stdev.unscaled, "R_lmseries_slow_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(
    sigma = fit_slow$sigma,
    df_residual = fit_slow$df.residual
  ),
  "R_lmseries_slow_stats.csv", row.names = TRUE
)

# gls.series slow path: NAs + probe weights + block correlation
cat("  gls.series slow path (NAs + probe weights + block) ...\n")
set.seed(202)
nb_genes2 <- 20
nb_samples2 <- 12
expr_gls <- matrix(rnorm(nb_genes2 * nb_samples2), nrow = nb_genes2)
rownames(expr_gls) <- paste0("gene", 1:nb_genes2)
expr_gls[cbind(c(2, 5, 8, 11, 15), c(1, 4, 6, 9, 12))] <- NA
expr_gls[1:5, 7:12] <- expr_gls[1:5, 7:12] + 1.5

weights_gls <- matrix(runif(nb_genes2 * nb_samples2, min = 0.5, max = 2.0),
                      nrow = nb_genes2)

group_gls <- factor(rep(c("A", "B"), each = nb_samples2 / 2))
design_gls <- model.matrix(~ group_gls)
block_gls  <- rep(1:6, each = 2)  # 6 blocks of 2 samples
corr_gls   <- 0.35

fit_gls_slow <- lmFit(expr_gls, design = design_gls,
                      block = block_gls, correlation = corr_gls,
                      weights = weights_gls)

write.csv(expr_gls,    "R_gls_slow_expr.csv",    row.names = TRUE)
write.csv(weights_gls, "R_gls_slow_weights.csv", row.names = TRUE)
write.csv(design_gls,  "R_gls_slow_design.csv",  row.names = FALSE)
write.csv(data.frame(block = block_gls), "R_gls_slow_block.csv", row.names = FALSE)
write.csv(fit_gls_slow$coefficients, "R_gls_slow_coef.csv",   row.names = TRUE)
write.csv(fit_gls_slow$stdev.unscaled, "R_gls_slow_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(
    sigma = fit_gls_slow$sigma,
    df_residual = fit_gls_slow$df.residual
  ),
  "R_gls_slow_stats.csv", row.names = TRUE
)

# e_bayes trend=TRUE AND robust=TRUE interaction on data2 (100 genes -> robust has teeth)
cat("  eBayes trend=TRUE + robust=TRUE (simultaneous) ...\n")
eb_tr <- eBayes(fit2, trend = TRUE, robust = TRUE)
write.csv(
  data.frame(
    t = eb_tr$t[, 2],
    p_value = eb_tr$p.value[, 2],
    lods = eb_tr$lods[, 2],
    s2_post = eb_tr$s2.post,
    df_total = eb_tr$df.total
  ),
  "R_ebayes_trend_robust_stats.csv", row.names = TRUE
)
write.csv(
  data.frame(
    s2_prior = eb_tr$s2.prior,
    df_prior = eb_tr$df.prior
  ),
  "R_ebayes_trend_robust_prior.csv", row.names = TRUE
)

# mrlm bisquare: direct MASS::rlm reference
cat("  mrlm bisquare via MASS::rlm ...\n")
suppressMessages(library(MASS))
set.seed(303)
nb_genes3 <- 20
nb_samples3 <- 8
expr_bi <- matrix(rnorm(nb_genes3 * nb_samples3), nrow = nb_genes3)
rownames(expr_bi) <- paste0("gene", 1:nb_genes3)
# Inject outliers in first 5 genes
expr_bi[1:5, 1] <- expr_bi[1:5, 1] + 8
expr_bi[1:10, 5:8] <- expr_bi[1:10, 5:8] + 2

group_bi <- factor(rep(c("A", "B"), each = 4))
design_bi <- model.matrix(~ group_bi)

bi_coef   <- matrix(NA, nb_genes3, ncol(design_bi))
bi_scale  <- numeric(nb_genes3)
bi_stdev  <- matrix(NA, nb_genes3, ncol(design_bi))
for (i in seq_len(nb_genes3)) {
  y <- as.vector(expr_bi[i, ])
  out <- MASS::rlm(x = design_bi, y = y, psi = psi.bisquare, maxit = 50)
  bi_coef[i, ] <- coef(out)
  bi_scale[i]  <- out$s
  bi_stdev[i, ] <- sqrt(diag(chol2inv(out$qr$qr)))
}

write.csv(expr_bi,   "R_mrlm_bisquare_expr.csv",   row.names = TRUE)
write.csv(design_bi, "R_mrlm_bisquare_design.csv", row.names = FALSE)
write.csv(bi_coef,   "R_mrlm_bisquare_coef.csv",   row.names = FALSE)
write.csv(data.frame(scale = bi_scale), "R_mrlm_bisquare_scale.csv", row.names = FALSE)
write.csv(bi_stdev,  "R_mrlm_bisquare_stdev.csv",  row.names = FALSE)

cat("  Higher-risk branch coverage fixtures complete.\n")

# -----------------------------------------------------------------------------
# Non-default arguments (interface completeness)
#   - eBayes with non-default stdev.coef.lim
#   - eBayes robust with non-default winsor.tail.p
#   - contrasts.fit after weighted lmFit (carries cov.coefficients)
#   - qqt theoretical quantiles
# -----------------------------------------------------------------------------
cat("\n--- Non-default argument fixtures (eBayes, contrasts.fit, qqt) ---\n")

# eBayes with non-default stdev.coef.lim
cat("  eBayes stdev.coef.lim=(0.05, 10) ...\n")
eb_stdevlim <- eBayes(fit1, stdev.coef.lim = c(0.05, 10))
write.csv(
  data.frame(
    lods = eb_stdevlim$lods[, 2],
    var_prior = rep(eb_stdevlim$var.prior[2], nrow(fit1$coefficients))
  ),
  "R_ebayes_stdev_coef_lim.csv", row.names = TRUE
)

# eBayes robust with non-default winsor.tail.p on data2
cat("  eBayes robust winsor.tail.p=(0.02, 0.15) ...\n")
eb_winsor <- eBayes(fit2, robust = TRUE, winsor.tail.p = c(0.02, 0.15))
write.csv(
  data.frame(
    t = eb_winsor$t[, 2],
    p_value = eb_winsor$p.value[, 2],
    s2_post = eb_winsor$s2.post,
    df_total = eb_winsor$df.total
  ),
  "R_ebayes_robust_winsor_stats.csv", row.names = TRUE
)
write.csv(
  data.frame(
    s2_prior = if (length(eb_winsor$s2.prior) > 1) eb_winsor$s2.prior
               else rep(eb_winsor$s2.prior, nrow(fit2$coefficients)),
    df_prior = if (length(eb_winsor$df.prior) > 1) eb_winsor$df.prior
               else rep(eb_winsor$df.prior, nrow(fit2$coefficients))
  ),
  "R_ebayes_robust_winsor_prior.csv", row.names = TRUE
)

# contrasts.fit after weighted lmFit - verify cov.coefficients carry-through
cat("  contrasts.fit after weighted lmFit ...\n")
set.seed(404)
nc_genes <- 25
nc_samples <- 12
expr_cw <- matrix(rnorm(nc_genes * nc_samples), nrow = nc_genes)
rownames(expr_cw) <- paste0("gene", 1:nc_genes)
# Three-group design (no intercept / cell-means)
group_cw <- factor(rep(c("A", "B", "C"), each = 4), levels = c("A", "B", "C"))
design_cw <- model.matrix(~ 0 + group_cw)
colnames(design_cw) <- c("A", "B", "C")

# Array weights (one per sample) - must be positive
array_wts <- c(1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 1.0, 1.0, 0.9, 1.1)

fit_cw <- lmFit(expr_cw, design_cw, weights = array_wts)

contr_cw <- makeContrasts(
  BvsA = B - A,
  CvsA = C - A,
  CvsB = C - B,
  levels = design_cw
)
cfit_cw <- contrasts.fit(fit_cw, contr_cw)

write.csv(expr_cw,    "R_contrasts_weighted_expr.csv",   row.names = TRUE)
write.csv(design_cw,  "R_contrasts_weighted_design.csv", row.names = FALSE)
write.csv(data.frame(weights = array_wts),
          "R_contrasts_weighted_arraywts.csv", row.names = FALSE)
write.csv(contr_cw,   "R_contrasts_weighted_contrmat.csv", row.names = TRUE)
write.csv(cfit_cw$coefficients, "R_contrasts_weighted_coef.csv",  row.names = TRUE)
write.csv(cfit_cw$stdev.unscaled, "R_contrasts_weighted_stdev.csv", row.names = TRUE)
write.csv(cfit_cw$cov.coefficients, "R_contrasts_weighted_cov.csv", row.names = TRUE)

# qqt theoretical quantiles - a deterministic check of pylimma's qqt
cat("  qqt theoretical quantiles ...\n")
set.seed(505)
n_qqt <- 100
y_qqt <- rt(n_qqt, df = 10)
# limma::qqt with plot.it=FALSE returns list(x=theoretical, y=sorted.y)
qqt_out <- qqt(y_qqt, df = 10, plot.it = FALSE)
write.csv(
  data.frame(x = qqt_out$x, y = qqt_out$y, y_input = y_qqt),
  "R_qqt_output.csv", row.names = FALSE
)

cat("  Non-default argument fixtures complete.\n")

# -----------------------------------------------------------------------------
# Niche branches
#   - gls.series intercept-only (all(X==0)) branch
#   - eBayes mixed Infdf branch (some genes have df.prior=Inf, others finite)
#   - makeContrasts with level names containing spaces
#   - lmFit(method="robust", ndups=2)
# -----------------------------------------------------------------------------
cat("\n--- Niche branch fixtures (gls.series, eBayes, makeContrasts, robust lmFit) ---\n")

# gls.series all(X==0) branch: per-gene slow path where observed rows of the
# design are all zero. Use no-intercept single-column design; put NAs in the
# samples that have nonzero design entries for a few genes.
cat("  gls.series all(X==0) branch ...\n")
set.seed(606)
nd_genes <- 15
nd_samples <- 8
expr_xz <- matrix(rnorm(nd_genes * nd_samples), nrow = nd_genes)
rownames(expr_xz) <- paste0("gene", 1:nd_genes)
# Single-column design, 0/1 indicator, no intercept
design_xz <- matrix(c(0,0,0,0,1,1,1,1), ncol = 1)
colnames(design_xz) <- "treatment"
# NAs on samples 5-8 for first 3 genes -> those genes only see samples with X=0
expr_xz[1:3, 5:8] <- NA
# Block structure to force gls.series slow path (need block + missing to route
# there)
block_xz <- rep(1:4, each = 2)

fit_xz <- lmFit(expr_xz, design = design_xz,
                block = block_xz, correlation = 0.2)

write.csv(expr_xz,   "R_gls_xzero_expr.csv",   row.names = TRUE)
write.csv(design_xz, "R_gls_xzero_design.csv", row.names = FALSE)
write.csv(data.frame(block = block_xz), "R_gls_xzero_block.csv", row.names = FALSE)
write.csv(fit_xz$coefficients, "R_gls_xzero_coef.csv", row.names = TRUE)
write.csv(
  data.frame(
    sigma = fit_xz$sigma,
    df_residual = fit_xz$df.residual
  ),
  "R_gls_xzero_stats.csv", row.names = TRUE
)

# eBayes mixed Infdf: the lods computation in ebayes.R:78-86 has a branch
# that only fires when `Infdf <- df.prior > 1e6` is mixed (some TRUE, some
# FALSE). Normal data rarely triggers it (robust winsorisation tends to give
# uniform df.prior). Synthesise a mixed-df fit by hand and run R's lods
# formula directly to produce the reference values.
cat("  eBayes mixed Infdf (synthetic fit) ...\n")
set.seed(707)
nm_genes <- 20
ncoef <- 2
# Fake t-statistics and stdev.unscaled
t_mi <- matrix(rnorm(nm_genes * ncoef, sd = 2), nrow = nm_genes)
stdev_mi <- matrix(runif(nm_genes * ncoef, min = 0.3, max = 0.7), nrow = nm_genes)

# Half the genes have Inf df.prior, half finite
df_prior_mi <- c(rep(Inf, nm_genes / 2), rep(15, nm_genes / 2))
df_residual <- rep(5, nm_genes)
df_total_mi <- pmin(df_prior_mi + df_residual, sum(df_residual))

# var.prior (prior variance of true effects) - derived via R's tmixture approach.
# For the mixed-branch test, we just fix it to a plausible value
# (the audit here verifies the kernel / lods formula given df.prior, not
# the upstream estimate).
var_prior_mi <- 1.5

# Apply R's exact kernel formula from ebayes.R:78-88 to produce reference lods
r_mi <- matrix(NA, nm_genes, ncoef)
for (j in seq_len(ncoef)) {
  rmat <- stdev_mi[, j]^2 + var_prior_mi
  r_mi[, j] <- rmat / (stdev_mi[, j]^2)
}

t2_mi <- t_mi^2
Infdf <- df_prior_mi > 1e6
kernel <- matrix(NA, nm_genes, ncoef)
kernel[Infdf, ]  <- t2_mi[Infdf, ] * (1 - 1 / r_mi[Infdf, ]) / 2
dft_f <- df_total_mi[!Infdf]
kernel[!Infdf, ] <- (1 + dft_f) / 2 *
  log((t2_mi[!Infdf, ] + dft_f) / (t2_mi[!Infdf, ] / r_mi[!Infdf, ] + dft_f))

proportion_mi <- 0.01
lods_mi <- log(proportion_mi / (1 - proportion_mi)) - log(r_mi) / 2 + kernel

# Save: inputs and expected output
write.csv(t_mi,       "R_ebayes_mixed_infdf_t.csv",     row.names = FALSE)
write.csv(stdev_mi,   "R_ebayes_mixed_infdf_stdev.csv", row.names = FALSE)
write.csv(
  data.frame(df_prior = df_prior_mi, df_residual = df_residual,
             df_total = df_total_mi),
  "R_ebayes_mixed_infdf_df.csv", row.names = FALSE
)
write.csv(
  data.frame(var_prior = var_prior_mi, proportion = proportion_mi),
  "R_ebayes_mixed_infdf_scalars.csv", row.names = FALSE
)
write.csv(lods_mi, "R_ebayes_mixed_infdf_lods.csv", row.names = FALSE)

# lmFit(method="robust", ndups=2)
cat("  lmFit method=robust + ndups=2 ...\n")
set.seed(808)
nd2_genes <- 10   # 20 spots (2 dups each)
nd2_samples <- 6
# The expression matrix has 2*nd2_genes rows (spots); after unwrapdups it will
# have nd2_genes rows and 2*nd2_samples columns
expr_rd <- matrix(rnorm(2 * nd2_genes * nd2_samples), nrow = 2 * nd2_genes)
rownames(expr_rd) <- paste0("spot", 1:(2 * nd2_genes))
# Inject outliers on a few genes
expr_rd[c(3, 7, 12), 1] <- expr_rd[c(3, 7, 12), 1] + 8

group_rd <- factor(rep(c("A", "B"), each = 3))
design_rd <- model.matrix(~ group_rd)

# R issues a warning but still calls mrlm with ndups=2
fit_rd <- suppressWarnings(
  lmFit(expr_rd, design_rd, method = "robust", ndups = 2, spacing = 1)
)

write.csv(expr_rd,   "R_lmfit_robust_ndups_expr.csv",   row.names = TRUE)
write.csv(design_rd, "R_lmfit_robust_ndups_design.csv", row.names = FALSE)
write.csv(fit_rd$coefficients, "R_lmfit_robust_ndups_coef.csv", row.names = TRUE)
write.csv(fit_rd$stdev.unscaled, "R_lmfit_robust_ndups_stdev.csv", row.names = TRUE)
write.csv(
  data.frame(
    sigma = fit_rd$sigma,
    df_residual = fit_rd$df.residual
  ),
  "R_lmfit_robust_ndups_stats.csv", row.names = TRUE
)

cat("  Niche branch fixtures complete.\n")

# -----------------------------------------------------------------------------
# Normalisation and batch-correction fixtures
# -----------------------------------------------------------------------------
cat("\nGenerating normalisation and batch-correction fixtures...\n")
cat("  R version:", R.version.string, "\n")
cat("  limma version:", as.character(packageVersion("limma")), "\n")

set.seed(2026)

# Foreground / background matrices for normexp + backgroundCorrect.
# Noise floor (normal) plus signal (exponential) mirrors the mixture the
# normexp model is defined against.
n_probes <- 2000
n_arrays <- 4
noise  <- matrix(rnorm(n_probes * n_arrays, mean = 50, sd = 10), n_probes, n_arrays)
signal <- matrix(rexp(n_probes * n_arrays, rate = 1 / 200),      n_probes, n_arrays)
E_fg <- noise + signal
E_bg <- matrix(rnorm(n_probes * n_arrays, mean = 40, sd = 8),    n_probes, n_arrays)
colnames(E_fg) <- paste0("array", 1:n_arrays)
colnames(E_bg) <- paste0("array", 1:n_arrays)
write.csv(E_fg, "R_intensities_foreground.csv", row.names = FALSE)
write.csv(E_bg, "R_intensities_background.csv", row.names = FALSE)

# normexp.fit: one column at a time, each method the port supports.
# Method "rma" requires the affy package and is out of scope for the port.
cat("  normexp.fit (saddle)...\n")
fit_saddle <- normexp.fit(E_fg[, 1], method = "saddle")
write.csv(data.frame(par = fit_saddle$par), "R_normexp_fit_saddle.csv", row.names = FALSE)

cat("  normexp.fit (mle)...\n")
fit_mle <- normexp.fit(E_fg[, 1], method = "mle")
write.csv(data.frame(par = fit_mle$par), "R_normexp_fit_mle.csv", row.names = FALSE)

cat("  normexp.fit (rma75)...\n")
fit_rma75 <- normexp.fit(E_fg[, 1], method = "rma75")
write.csv(data.frame(par = fit_rma75$par), "R_normexp_fit_rma75.csv", row.names = FALSE)

# normexp.fit (rma) delegates to affy::bg.parameters. affy is not installed
# so we source bg.parameters from the vendored sibling file
# tests/fixtures/affy_bg_parameters.R (LGPL-2+, see that file's header for
# attribution) and replicate normexp.fit's rma branch inline.
cat("  normexp.fit (rma via vendored affy)...\n")
source("affy_bg_parameters.R", local = TRUE)
rma_out <- bg.parameters(E_fg[, 1])
fit_rma_par <- c(rma_out$mu, log(rma_out$sigma), -log(rma_out$alpha))
write.csv(data.frame(par = fit_rma_par), "R_normexp_fit_rma.csv", row.names = FALSE)

# normexp.signal: expected signal given fitted parameters
cat("  normexp.signal...\n")
sig_saddle <- normexp.signal(fit_saddle$par, E_fg[, 1])
write.csv(data.frame(signal = sig_saddle), "R_normexp_signal_saddle.csv", row.names = FALSE)

# backgroundCorrect: every in-scope method. subtract/half/minimum need Eb
# (R's top-level dispatch on a bare matrix rejects them otherwise, so pass
# Eb unconditionally so the branch is actually exercised).
cat("  backgroundCorrect methods...\n")
for (m in c("none", "subtract", "half", "minimum", "normexp")) {
  bc <- backgroundCorrect.matrix(E_fg, Eb = E_bg, method = m, verbose = FALSE,
                                 normexp.method = "saddle")
  write.csv(bc, paste0("R_background_correct_", m, ".csv"), row.names = FALSE)
}

# backgroundCorrect with an offset (normexp path)
bc_off <- backgroundCorrect.matrix(E_fg, method = "normexp", offset = 50,
                                   normexp.method = "saddle", verbose = FALSE)
write.csv(bc_off, "R_background_correct_normexp_offset.csv", row.names = FALSE)

# backgroundCorrect with an explicit Eb (subtract path, no offset)
bc_bg <- backgroundCorrect.matrix(E_fg, Eb = E_bg, method = "subtract")
write.csv(bc_bg, "R_background_correct_subtract_eb.csv", row.names = FALSE)

# avearrays: .default (matrix) and .EList
cat("  avearrays.default...\n")
ids     <- c("a", "a", "b", "b")
weights_mat <- matrix(runif(n_probes * n_arrays, 0.5, 1.5), n_probes, n_arrays)
av_basic    <- avearrays(E_fg, ID = ids)
av_weighted <- avearrays(E_fg, ID = ids, weights = weights_mat)
write.csv(av_basic,    "R_avearrays_basic.csv",    row.names = FALSE)
write.csv(av_weighted, "R_avearrays_weighted.csv", row.names = FALSE)
write.csv(weights_mat, "R_avearrays_weights_input.csv", row.names = FALSE)

cat("  avearrays.EList...\n")
el <- new("EList", list(E = E_fg, weights = weights_mat,
                        targets = data.frame(id = ids)))
av_el <- avearrays(el, ID = ids)
write.csv(av_el$E,       "R_avearrays_elist_E.csv",       row.names = FALSE)
write.csv(av_el$weights, "R_avearrays_elist_weights.csv", row.names = FALSE)

# removeBatchEffect: three representative branches. Build a 12-sample data
# matrix so the combined design (2 design + batch + batch2 + covariates) is
# full-rank - otherwise lmFit's column pivoting would leak into rbe output
# as a confound with which column gets marked non-estimable, which is an
# lmFit QR-pivoting concern rather than removeBatchEffect behaviour.
set.seed(2026 + 1)
n_rbe_samples <- 12
noise_rbe  <- matrix(rnorm(n_probes * n_rbe_samples, mean = 50, sd = 10),
                     n_probes, n_rbe_samples)
signal_rbe <- matrix(rexp(n_probes * n_rbe_samples, rate = 1 / 200),
                     n_probes, n_rbe_samples)
E_rbe <- noise_rbe + signal_rbe
colnames(E_rbe) <- paste0("array", 1:n_rbe_samples)
write.csv(E_rbe, "R_rbe_E_input.csv", row.names = FALSE)

cat("  removeBatchEffect (batch only)...\n")
batch  <- factor(rep(c("B1", "B2", "B3"), each = 4))
group  <- factor(c("ctrl", "trt", "trt", "ctrl",
                   "trt", "ctrl", "ctrl", "trt",
                   "ctrl", "trt", "trt", "ctrl"))
design <- model.matrix(~ group)
rbe1 <- removeBatchEffect(E_rbe, batch = batch, design = design)
write.csv(rbe1, "R_rbe_batch.csv", row.names = FALSE)

cat("  removeBatchEffect (batch + covariates)...\n")
cov_mat <- matrix(rnorm(n_rbe_samples * 2), n_rbe_samples, 2,
                  dimnames = list(NULL, c("cv1", "cv2")))
rbe2 <- removeBatchEffect(E_rbe, batch = batch, covariates = cov_mat,
                          design = design)
write.csv(rbe2, "R_rbe_batch_covariates.csv", row.names = FALSE)
write.csv(cov_mat, "R_rbe_covariates_input.csv", row.names = FALSE)

cat("  removeBatchEffect (batch + batch2)...\n")
batch2 <- factor(c("X", "X", "Y", "Y", "X", "Y", "X", "Y", "Y", "X", "Y", "X"))
rbe3 <- removeBatchEffect(E_rbe, batch = batch, batch2 = batch2,
                          design = design)
write.csv(rbe3, "R_rbe_batch_batch2.csv", row.names = FALSE)

cat("  normalisation and batch-correction fixtures complete.\n")

# -----------------------------------------------------------------------------
# Normalisation and batch-correction branch-forcing fixtures
# -----------------------------------------------------------------------------
# Force R branches that the normalisation and batch-correction fixtures above
# do not exercise. Do not restructure the blocks above when editing these;
# append new branches here.

cat("Normalisation and batch-correction branch-forcing fixtures...\n")

cat("  normexp.fit(n.pts=200) downsample path...\n")
set.seed(2026)
x_npts <- rnorm(2000, mean = 50, sd = 10) + rexp(2000, 1/200)
nf_npts <- normexp.fit(x_npts, method = "saddle", n.pts = 200)
write.csv(data.frame(par = nf_npts$par), "R_normexp_fit_saddle_npts.csv",
          row.names = FALSE)
write.csv(data.frame(x = x_npts), "R_normexp_fit_npts_input.csv",
          row.names = FALSE)

cat("  normexp.fit(all-equal) degenerate q[1]==q[4] branch...\n")
x_const <- rep(50.0, 100)
nf_const <- normexp.fit(x_const, method = "saddle")
# Returns par=c(q[1], -Inf, -Inf), m2loglik=NA, convergence=0
write.csv(data.frame(par = nf_const$par), "R_normexp_fit_degenerate.csv",
          row.names = FALSE)

cat("  normexp.fit mu-fallback branch (q[3]>q[1] but q[2]==q[1])...\n")
# Construct x such that 5th percentile == minimum (so q[2]==q[1]) but 10th
# percentile > minimum (so q[3]>q[1]). Achieved by tying >=5% of mass at min.
set.seed(7)
x_mu_fb <- c(rep(0.0, 150), rnorm(1850, mean = 50, sd = 10) + rexp(1850, 1/100))
nf_mu_fb <- normexp.fit(x_mu_fb, method = "saddle")
write.csv(data.frame(par = nf_mu_fb$par), "R_normexp_fit_mu_fallback.csv",
          row.names = FALSE)
write.csv(data.frame(x = x_mu_fb), "R_normexp_fit_mu_fallback_input.csv",
          row.names = FALSE)

cat("  backgroundCorrect.matrix silent-downgrade branch (Eb=NULL + method=subtract)...\n")
set.seed(11)
E_down <- matrix(runif(40, min = 100, max = 200), nrow = 10, ncol = 4)
# method="subtract" with Eb NULL should silently downgrade to "none", i.e.
# return E unchanged.
bc_down <- backgroundCorrect.matrix(E_down, method = "subtract")
write.csv(bc_down, "R_background_correct_downgrade.csv", row.names = FALSE)
write.csv(E_down, "R_background_correct_downgrade_input.csv",
          row.names = FALSE)

cat("  avearrays character-mode short-circuit...\n")
char_mat <- matrix(paste0("x", 1:12), nrow = 3, ncol = 4)
aa_char <- avearrays(char_mat, ID = c("a", "a", "b", "b"))
write.csv(aa_char, "R_avearrays_character.csv", row.names = FALSE)

cat("  removeBatchEffect(group=) path...\n")
set.seed(13)
E_grp <- matrix(rnorm(100 * 12), nrow = 100, ncol = 12)
batch_grp <- factor(rep(c("B1", "B2", "B3"), each = 4))
group_grp <- factor(rep(c("ctrl", "trt"), 6))
rbe_grp <- removeBatchEffect(E_grp, batch = batch_grp, group = group_grp)
write.csv(rbe_grp, "R_rbe_group.csv", row.names = FALSE)
write.csv(E_grp, "R_rbe_group_E_input.csv", row.names = FALSE)

cat("  removeBatchEffect rank-deficient (batch confounded with group)...\n")
set.seed(17)
E_rd <- matrix(rnorm(100 * 8), nrow = 100, ncol = 8)
batch_rd <- factor(c("B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2"))
group_rd <- factor(c("A", "A", "A", "A", "B", "B", "B", "B"))
des_rd <- model.matrix(~ group_rd)
# batch is perfectly confounded with group; lmFit drops the redundant batch
# column. The remaining beta for batch is NA and should be replaced with 0 by
# removeBatchEffect's beta[is.na(beta)] <- 0 line, giving output == input.
rbe_rd <- removeBatchEffect(E_rd, batch = batch_rd, design = des_rd)
write.csv(rbe_rd, "R_rbe_rank_deficient.csv", row.names = FALSE)
write.csv(E_rd, "R_rbe_rank_deficient_E_input.csv", row.names = FALSE)

cat("  _sum_to_zero_design: integer batch vs character batch...\n")
# Integer batch labels: R's as.factor(numeric) sorts numerically, so levels
# become c(1, 2, 10) - not c("1","10","2") as alphabetical string sort would
# produce.
set.seed(19)
E_int_batch <- matrix(rnorm(50 * 9), nrow = 50, ncol = 9)
batch_int <- c(1, 1, 1, 10, 10, 10, 2, 2, 2)  # deliberately mis-ordered ints
design_int <- matrix(1, 9, 1)
rbe_int <- removeBatchEffect(E_int_batch, batch = batch_int, design = design_int)
write.csv(rbe_int, "R_rbe_integer_batch.csv", row.names = FALSE)
write.csv(E_int_batch, "R_rbe_integer_batch_E_input.csv", row.names = FALSE)

cat("  Normalisation and batch-correction branch-forcing fixtures complete.\n")

# -----------------------------------------------------------------------------
# Gene-set testing fixtures
# -----------------------------------------------------------------------------
cat("\nGenerating gene-set testing fixtures...\n")
cat("  R version:", R.version.string, "\n")
cat("  limma version:", as.character(packageVersion("limma")), "\n")

set.seed(4)

# Shared simulated dataset: 400 genes x 8 samples, 2-group design, three
# overlapping gene sets, plus a singleton set. Modest differential signal
# in set A so roast/camera/fry return non-trivial p-values.
n_genes_geneset  <- 400L
n_arrays_geneset <- 8L
y_geneset        <- matrix(rnorm(n_genes_geneset * n_arrays_geneset), n_genes_geneset, n_arrays_geneset)
rownames(y_geneset) <- paste0("g", seq_len(n_genes_geneset))
group_geneset    <- factor(rep(c("A", "B"), each = n_arrays_geneset / 2L))
design_geneset   <- model.matrix(~group_geneset)
colnames(design_geneset) <- c("(Intercept)", "groupB")
contrast_geneset <- 2L  # groupB vs groupA

# Inject signal into the first 20 genes for group B
y_geneset[1:20, 5:8] <- y_geneset[1:20, 5:8] + 1.0

# Gene sets: named list of integer indices (how R limma stores them after
# ids2indices). Keep some overlap to exercise camera's VIF path.
gene.sets <- list(
  setA = 1:20,            # the upregulated set
  setB = 15:40,           # overlaps setA
  setC = sample(50:200, 30L),
  setD = c(300L)          # singleton (roast/fry must still run)
)
write.csv(y_geneset,      "R_geneset_y.csv",      row.names = TRUE)
write.csv(design_geneset, "R_geneset_design.csv", row.names = FALSE)
# gene sets -> long-form CSV (set_name, gene_index) so the Python side
# can reconstruct the list without eval()
sets.long <- do.call(rbind, lapply(names(gene.sets), function(nm)
  data.frame(set = nm, index = gene.sets[[nm]])))
write.csv(sets.long, "R_geneset_gene_sets.csv", row.names = FALSE)

# ids2indices: round-trip via identifiers vector
ids <- ids2indices(
  gene.sets = list(setA = paste0("g", 1:20), setB = paste0("g", 15:40)),
  identifiers = rownames(y_geneset)
)
write.csv(data.frame(
  set   = rep(names(ids), lengths(ids)),
  index = unlist(ids, use.names = FALSE)
), "R_ids2indices_basic.csv", row.names = FALSE)

# zscoreT: one fixture per branch. R's match.arg restricts `method` to
# {"bailey","hill","wallace"}; the exact quantile path is reached via
# approx=FALSE (method is ignored on that path).
t.vec  <- qt(ppoints(50L), df = 7)
df.vec <- rep(c(3, 7, 15, 30, 100), length.out = 50L)
for (m in c("bailey", "hill", "wallace")) {
  z <- zscoreT(t.vec, df = df.vec, approx = TRUE, method = m)
  write.csv(data.frame(t = t.vec, df = df.vec, z = z),
            paste0("R_zscoret_", m, ".csv"), row.names = FALSE)
}
# Exact quantile path
z_quantile <- zscoreT(t.vec, df = df.vec, approx = FALSE)
write.csv(data.frame(t = t.vec, df = df.vec, z = z_quantile),
          "R_zscoret_quantile.csv", row.names = FALSE)

# tricubeMovingAverage: deterministic input, default span + a wider one
set.seed(42)
tma.x <- rnorm(50L)
write.csv(data.frame(x = tma.x,
                     y_default = tricubeMovingAverage(tma.x),
                     y_wide    = tricubeMovingAverage(tma.x, span = 0.8)),
          "R_tricube_moving_average.csv", row.names = FALSE)

# roast (single set) with FROZEN seed - roast uses sample.int internally
cat("  roast (single set, seed 4)...\n")
set.seed(4)
r_single <- roast(y_geneset, index = gene.sets$setA, design = design_geneset,
                  contrast = contrast_geneset, nrot = 999)
write.csv(as.data.frame(r_single$p.value),       "R_roast_pvalues.csv", row.names = TRUE)
write.csv(data.frame(ngenes = r_single$ngenes),  "R_roast_ngenes.csv",  row.names = FALSE)

# mroast: every summary statistic path
cat("  mroast (mean, floormean, median, msq)...\n")
for (ss in c("mean", "floormean", "mean50", "msq")) {
  set.seed(4)
  mr <- mroast(y_geneset, index = gene.sets, design = design_geneset, contrast = contrast_geneset,
               set.statistic = ss, nrot = 999)
  write.csv(mr, paste0("R_mroast_", ss, ".csv"), row.names = TRUE)
}

# fry (closed-form, no RNG)
cat("  fry (single + multi)...\n")
fr_single <- fry(y_geneset, index = gene.sets$setA, design = design_geneset, contrast = contrast_geneset)
write.csv(as.data.frame(fr_single), "R_fry_single.csv", row.names = TRUE)
fr_multi  <- fry(y_geneset, index = gene.sets,      design = design_geneset, contrast = contrast_geneset)
write.csv(fr_multi, "R_fry_multi.csv", row.names = TRUE)

# camera: ranks-based and parametric paths, with and without VIF estimation
cat("  camera (default + use.ranks + inter.gene.cor)...\n")
cam_default <- camera(y_geneset, index = gene.sets, design = design_geneset, contrast = contrast_geneset)
write.csv(cam_default, "R_camera_default.csv", row.names = TRUE)
cam_ranks   <- camera(y_geneset, index = gene.sets, design = design_geneset, contrast = contrast_geneset,
                      use.ranks = TRUE)
write.csv(cam_ranks,   "R_camera_ranks.csv",   row.names = TRUE)
cam_cor     <- camera(y_geneset, index = gene.sets, design = design_geneset, contrast = contrast_geneset,
                      inter.gene.cor = 0.05)
write.csv(cam_cor,     "R_camera_intergene.csv", row.names = TRUE)

# cameraPR: preranked statistic input
stat <- rnorm(n_genes_geneset)
names(stat) <- rownames(y_geneset)
stat[1:20] <- stat[1:20] + 1.5
cpr <- cameraPR(stat, index = gene.sets)
write.csv(cpr, "R_camera_pr.csv", row.names = TRUE)
write.csv(data.frame(statistic = stat), "R_camera_pr_input.csv", row.names = TRUE)

# romer: three set.statistic branches, FROZEN seed
cat("  romer (mean, floormean, mean50)...\n")
for (ss in c("mean", "floormean", "mean50")) {
  set.seed(4)
  ro <- romer(y_geneset, index = gene.sets, design = design_geneset, contrast = contrast_geneset,
              set.statistic = ss, nrot = 999)
  write.csv(ro, paste0("R_romer_", ss, ".csv"), row.names = TRUE)
}

# geneSetTest + rankSumTestWithCorrelation
cat("  geneSetTest (alternatives, ranks.only)...\n")
stat_vec <- rnorm(n_genes_geneset); stat_vec[1:20] <- stat_vec[1:20] + 1.5
write.csv(data.frame(statistic = stat_vec), "R_genesettest_input.csv", row.names = FALSE)
if (file.exists("R_geneSetTest.csv")) file.remove("R_geneSetTest.csv")
for (alt in c("mixed", "up", "down", "either")) {
  for (ranks in c(TRUE, FALSE)) {
    set.seed(4)
    p <- geneSetTest(index = 1:20, statistics = stat_vec,
                     alternative = alt, ranks.only = ranks, nsim = 999)
    write(paste(alt, ranks, p, sep = ","),
          file = "R_geneSetTest.csv", append = TRUE)
  }
}
# rankSumTestWithCorrelation
cat("  rankSumTestWithCorrelation...\n")
rstc <- rankSumTestWithCorrelation(index = 1:20, statistics = stat_vec,
                                   correlation = 0.05, df = Inf)
write.csv(data.frame(less = rstc["less"], greater = rstc["greater"]),
          "R_rank_sum_test_with_correlation.csv", row.names = FALSE)

# geneSetTest simulation path with a MODERATE signal so the Monte-Carlo
# p-value is not pinned to 1/(nsim+1). Runs nsim=9999 at several seeds so
# the Python parity test has something non-trivial to compare against.
cat("  geneSetTest simulation (moderate signal)...\n")
set.seed(7)
stat_mod <- rnorm(n_genes_geneset)
stat_mod[1:40] <- stat_mod[1:40] + 0.35   # modest uplift
write.csv(data.frame(statistic = stat_mod),
          "R_genesettest_sim_input.csv", row.names = FALSE)
if (file.exists("R_geneSetTest_sim.csv")) file.remove("R_geneSetTest_sim.csv")
for (seed in c(1L, 11L, 101L)) {
  for (alt in c("mixed", "up", "either")) {
    set.seed(seed)
    p <- geneSetTest(index = 1:40, statistics = stat_mod,
                     alternative = alt, ranks.only = FALSE, nsim = 9999)
    write(paste(seed, alt, p, sep = ","),
          file = "R_geneSetTest_sim.csv", append = TRUE)
  }
}

# convest + propTrueNull: pi0 estimation on a p-value vector with known
# pi0 ~ 0.7 (700 nulls + 300 alternatives)
cat("  convest + propTrueNull...\n")
set.seed(4)
p_null    <- runif(700L)
p_de      <- rbeta(300L, 1, 20)
p_vec     <- c(p_null, p_de)
write.csv(data.frame(p = p_vec), "R_propTrueNull_input.csv", row.names = FALSE)
if (file.exists("R_propTrueNull.csv")) file.remove("R_propTrueNull.csv")
for (m in c("lfdr", "mean", "hist", "convest")) {
  pi0 <- propTrueNull(p_vec, method = m)
  write(paste(m, pi0, sep = ","),
        file = "R_propTrueNull.csv", append = TRUE)
}
c1 <- convest(p_vec)
c2 <- convest(p_vec, niter = 200)
write.csv(data.frame(pi0_default = c1, pi0_200iter = c2),
          "R_convest.csv", row.names = FALSE)

# detectionPValues.default (matrix path): status = "negative" for a
# control subset, "regular" for the rest. Uses an exponential intensity
# distribution so negative controls and regular probes differ.
cat("  detectionPValues (matrix path)...\n")
set.seed(4)
pr_mat <- matrix(rexp(200L * 4L, rate = 1/50), 200L, 4L)
status_geneset <- c(rep("negative", 30L), rep("regular", 170L))
dp <- detectionPValues(pr_mat, status = status_geneset, negctrl = "negative")
write.csv(pr_mat, "R_detectionPValues_input.csv", row.names = FALSE)
write.csv(data.frame(status = status_geneset), "R_detectionPValues_status.csv", row.names = FALSE)
write.csv(dp, "R_detectionPValues.csv", row.names = FALSE)

cat("  gene-set testing fixtures complete.\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("\n=== Fixture Generation Complete ===\n")
cat(sprintf("Generated %d CSV files\n", length(list.files(pattern = "^R_.*\\.csv$"))))


# -----------------------------------------------------------------------------
# Visualisation, wsva and differential-splicing fixtures
# -----------------------------------------------------------------------------
cat("\nGenerating visualisation, wsva and splicing fixtures...\n")
cat("  R version:", R.version.string, "\n")
cat("  limma version:", as.character(packageVersion("limma")), "\n")

library(limma)
set.seed(5)

# Shared simulated dataset: 500 genes x 6 samples, 2 groups, for MA/MD/volcano.
n_genes  <- 500L
n_arrays <- 6L
E_twogroup     <- matrix(rnorm(n_genes * n_arrays, mean = 8, sd = 1), n_genes, n_arrays)
rownames(E_twogroup) <- paste0("g", seq_len(n_genes))
colnames(E_twogroup) <- paste0("s", seq_len(n_arrays))
group_twogroup     <- factor(rep(c("A", "B"), each = n_arrays / 2L))
design_twogroup    <- model.matrix(~group_twogroup)
colnames(design_twogroup) <- c("(Intercept)", "groupB")
# Inject signal
E_twogroup[1:50, 4:6] <- E_twogroup[1:50, 4:6] + 1.5
fit_twogroup  <- lmFit(E_twogroup, design_twogroup)
fit_twogroup_c <- eBayes(contrasts.fit(fit_twogroup, coefficients = 2))
write.csv(E_twogroup,      "R_twogroup_E.csv",      row.names = TRUE)
write.csv(design_twogroup, "R_twogroup_design.csv", row.names = FALSE)

# plotMA numeric substrate for an MArrayLM: Amean vs coef[, 1].
x_ma <- fit_twogroup_c$Amean
y_ma <- fit_twogroup_c$coefficients[, 1]
write.csv(data.frame(A = x_ma, M = y_ma),
          "R_plot_ma_data.csv", row.names = TRUE)

# plotMD on a matrix: column 1 vs midpoint of (column 1, rowMeans of others)
#   This mirrors R plotMD.default lines 99-101 exactly.
ave_twogroup <- rowMeans(E_twogroup[, -1, drop = FALSE])
md_x <- (E_twogroup[, 1] + ave_twogroup) / 2
md_y <- E_twogroup[, 1] - ave_twogroup
write.csv(data.frame(A = md_x, M = md_y),
          "R_plot_md_matrix.csv", row.names = TRUE)

# volcano: log2FC vs -log10(p-value) and B-statistic styles
write.csv(data.frame(
  log_fc = fit_twogroup_c$coefficients[, 1],
  neg_log10_p = -log10(fit_twogroup_c$p.value[, 1]),
  b = fit_twogroup_c$lods[, 1]
), "R_volcano_data.csv", row.names = TRUE)

# plotSA: sqrt(sigma) on y-axis; trend overlay sqrt(sqrt(s2.prior[order(x)]))
# in sorted-x order. Two fixtures: trend=TRUE+robust=TRUE gives per-gene
# s2.prior AND per-gene df.prior (triggers outlier detection); default
# eBayes gives scalar s2.prior + scalar df.prior (flat line, no outliers).
fit_trend <- eBayes(contrasts.fit(lmFit(E_twogroup, design_twogroup), coefficients = 2),
                    trend = TRUE, robust = TRUE)
write.csv(data.frame(
  Amean       = fit_trend$Amean,
  sqrt_sigma  = sqrt(fit_trend$sigma),
  s2_prior    = fit_trend$s2.prior,
  df_prior    = fit_trend$df.prior
), "R_plot_sa_trend.csv", row.names = TRUE)
o_trend <- order(fit_trend$Amean)
write.csv(data.frame(
  x_sorted    = fit_trend$Amean[o_trend],
  trend_y     = sqrt(sqrt(fit_trend$s2.prior[o_trend]))
), "R_plot_sa_trend_line.csv", row.names = FALSE)

fit_flat <- eBayes(contrasts.fit(lmFit(E_twogroup, design_twogroup), coefficients = 2))
write.csv(data.frame(
  Amean       = fit_flat$Amean,
  sqrt_sigma  = sqrt(fit_flat$sigma),
  s2_prior    = rep(fit_flat$s2.prior, length(fit_flat$Amean)),
  df_prior    = rep(fit_flat$df.prior, length(fit_flat$Amean))
), "R_plot_sa_flat.csv", row.names = TRUE)
write.csv(data.frame(
  flat_y = sqrt(sqrt(fit_flat$s2.prior))
), "R_plot_sa_flat_line.csv", row.names = FALSE)

# plotDensities: compute the density curves R would draw (default kernel = gaussian)
pd_densities <- apply(E_twogroup, 2, function(col) {
  d <- density(col, n = 512)
  data.frame(x = d$x, y = d$y)
})
pd_densities_long <- do.call(rbind, lapply(seq_along(pd_densities), function(i)
  data.frame(sample = colnames(E_twogroup)[i], pd_densities[[i]])))
write.csv(pd_densities_long, "R_plot_densities.csv", row.names = FALSE)

# plotMDS: extract coordinates and variance-explained for both gene.selection
# branches, with two different top values. Eigenvector signs are left to
# LAPACK by R (and differ between R installations), so each dimension is
# written with a canonical sign: its largest-magnitude entry positive
# (first maximum on ties). The Python test applies the same rule.
.canonical_sign <- function(v) {
  s <- sign(v[which.max(abs(v))])
  if (s == 0) v else v * s
}
for (sel in c("pairwise", "common")) {
  for (top_k in c(100, 500)) {
    mds <- plotMDS(E_twogroup, plot = FALSE, top = top_k, gene.selection = sel)
    write.csv(data.frame(
      sample = colnames(E_twogroup),
      dim1   = .canonical_sign(mds$x),
      dim2   = .canonical_sign(mds$y),
      var_explained_1 = mds$var.explained[1],
      var_explained_2 = mds$var.explained[2]
    ), sprintf("R_plot_mds_%s_top%d.csv", sel, top_k), row.names = FALSE)
  }
}

# vennCounts: need multi-contrast fit for a meaningful Venn (3 contrasts).
set.seed(5)
grp3 <- factor(rep(c("A","B","C"), each = 4L))
design3 <- model.matrix(~0 + grp3)
colnames(design3) <- c("A","B","C")
n_g3 <- 500L
E3 <- matrix(rnorm(n_g3 * 12L, mean = 8, sd = 1), n_g3, 12L)
rownames(E3) <- paste0("g", seq_len(n_g3))
colnames(E3) <- paste0("s", seq_len(12L))
E3[1:50,  5:8]  <- E3[1:50,  5:8]  + 1.5  # B-A signal
E3[51:100, 9:12] <- E3[51:100, 9:12] + 1.5  # C-A signal
fit3 <- lmFit(E3, design3)
cm <- makeContrasts(BvsA = B - A, CvsA = C - A, CvsB = C - B, levels = design3)
fit3c <- eBayes(contrasts.fit(fit3, cm))
dec <- decideTests(fit3c)
write.csv(as.data.frame(unclass(dec)), "R_decideTests_input.csv", row.names = FALSE)

vc  <- vennCounts(dec)
write.csv(as.data.frame(unclass(vc)), "R_venn_counts.csv", row.names = FALSE)
vc_up   <- vennCounts(dec, include = "up")
write.csv(as.data.frame(unclass(vc_up)),   "R_venn_counts_up.csv",   row.names = FALSE)
vc_down <- vennCounts(dec, include = "down")
write.csv(as.data.frame(unclass(vc_down)), "R_venn_counts_down.csv", row.names = FALSE)

# coolmap: dump the scaled matrix and both dendrogram orders for each
# cluster.by branch. Uses the first 50 genes of E_twogroup for compactness.
for (cb in c("de pattern", "expression level")) {
  E_cm <- E_twogroup[1:50, ]
  if (cb == "de pattern") {
    M <- rowMeans(E_cm, na.rm = TRUE)
    DF <- ncol(E_cm) - 1L
    z <- E_cm - M
    V <- rowSums(z^2, na.rm = TRUE) / DF
    z <- z / sqrt(V + 0.01)
  } else {
    z <- E_cm
  }
  hr <- hclust(dist(z,    method = "euclidean"), method = "complete")
  hc <- hclust(dist(t(z), method = "euclidean"), method = "complete")
  write.csv(z, sprintf("R_coolmap_z_%s.csv",
                       gsub(" ", "_", cb)), row.names = TRUE)
  write.csv(data.frame(order = hr$order),
            sprintf("R_coolmap_row_order_%s.csv", gsub(" ", "_", cb)),
            row.names = FALSE)
  write.csv(data.frame(order = hc$order),
            sprintf("R_coolmap_col_order_%s.csv", gsub(" ", "_", cb)),
            row.names = FALSE)
}

# plotWithHighlights input fixture: 200 points with 3 status categories.
set.seed(5)
x_pwh <- rnorm(200)
y_pwh <- rnorm(200)
status_pwh <- rep("background", 200)
status_pwh[1:10]   <- "up"
status_pwh[11:20]  <- "down"
write.csv(data.frame(i = seq_along(x_pwh), x = x_pwh, y = y_pwh,
                     status = status_pwh),
          "R_plot_with_highlights_input.csv", row.names = FALSE)

# barcodeplot substrate: sorted positions + tricube worm.
set.seed(5)
stat_bp  <- rnorm(1000); stat_bp[1:50] <- stat_bp[1:50] + 1
index_bp <- 1:50
# barcodeplot sorts DECREASING internally only for index lookup; for the
# numeric substrate we reproduce: sort ascending (R default), then the worm
# is tricubeMovingAverage of the membership indicator over that order.
ostat <- order(stat_bp, na.last = TRUE, decreasing = FALSE)
stat_sorted <- stat_bp[ostat]
idx <- rep_len(FALSE, length(stat_bp))
idx[index_bp] <- TRUE
idx_sorted <- idx[ostat]
ave_enrich <- sum(idx_sorted) / length(idx_sorted)
worm <- tricubeMovingAverage(idx_sorted, span = 0.45) / ave_enrich
write.csv(data.frame(
  rank   = seq_along(stat_bp),
  stat   = stat_sorted,
  member = idx_sorted,
  worm   = worm
), "R_barcodeplot_substrate.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# wsva (weighted surrogate variable analysis).
# -----------------------------------------------------------------------------
cat("  wsva (unweighted + weighted-by-sd)...\n")
set.seed(5)
E_wsva <- E_twogroup
batch_latent <- rep(c(-0.5, 0.5), each = n_arrays / 2L)
E_wsva[1:250, ] <- E_wsva[1:250, ] + rep(batch_latent, each = 250L)
sv_unweighted <- wsva(E_wsva, design_twogroup, n.sv = 2L, weight.by.sd = FALSE)
sv_weighted   <- wsva(E_wsva, design_twogroup, n.sv = 2L, weight.by.sd = TRUE)
write.csv(E_wsva, "R_wsva_input.csv", row.names = FALSE)
write.csv(sv_unweighted, "R_wsva_unweighted.csv", row.names = FALSE)
write.csv(sv_weighted,   "R_wsva_weighted.csv",   row.names = FALSE)

# -----------------------------------------------------------------------------
# diffSplice / topSplice / plotSplice.
# -----------------------------------------------------------------------------
cat("  diffSplice + topSplice substrate...\n")
set.seed(5)
n_exons  <- 100L
n_arr_ds <- 6L
geneid_ds <- rep(paste0("gene", 1:20), each = 5L)
exonid_ds <- paste0("exon", seq_len(n_exons))
y_ds     <- matrix(rnorm(n_exons * n_arr_ds), n_exons, n_arr_ds)
rownames(y_ds) <- exonid_ds
colnames(y_ds) <- paste0("s", seq_len(n_arr_ds))
group_ds <- factor(rep(c("A", "B"), each = n_arr_ds / 2L))
design_ds <- model.matrix(~group_ds)
y_ds[1:2, 4:6] <- y_ds[1:2, 4:6] + 2.0
fit_ds <- lmFit(y_ds, design_ds)
fit_ds$genes <- data.frame(GeneID = geneid_ds, ExonID = exonid_ds)
ds <- diffSplice(fit_ds, geneid = "GeneID", exonid = "ExonID", verbose = FALSE)

write.csv(y_ds,                   "R_diffSplice_input_y.csv",        row.names = TRUE)
write.csv(design_ds,              "R_diffSplice_input_design.csv",   row.names = FALSE)
write.csv(ds$coefficients,        "R_diffSplice_coefficients.csv",   row.names = TRUE)
write.csv(ds$t,                   "R_diffSplice_t.csv",              row.names = TRUE)
write.csv(ds$p.value,             "R_diffSplice_p.csv",              row.names = TRUE)
write.csv(ds$gene.F,              "R_diffSplice_gene_F.csv",         row.names = TRUE)
write.csv(ds$gene.F.p.value,      "R_diffSplice_gene_F_p.csv",       row.names = TRUE)
write.csv(ds$gene.simes.p.value,  "R_diffSplice_gene_simes_p.csv",   row.names = TRUE)
write.csv(ds$genes,               "R_diffSplice_genes.csv",          row.names = TRUE)
write.csv(ds$gene.genes,          "R_diffSplice_gene_genes.csv",     row.names = TRUE)
write.csv(data.frame(first = ds$gene.firstexon, last = ds$gene.lastexon),
          "R_diffSplice_gene_firstlast.csv",  row.names = FALSE)

# legacy=TRUE: switches squeezeVar's hyperparameter estimation to the
# pre-2024 fit_f_dist path; Simes logic is unchanged by the flag.
ds_legacy <- diffSplice(fit_ds, geneid = "GeneID", exonid = "ExonID",
                        legacy = TRUE, verbose = FALSE)
write.csv(ds_legacy$coefficients,       "R_diffSplice_legacy_coefficients.csv", row.names = TRUE)
write.csv(ds_legacy$t,                  "R_diffSplice_legacy_t.csv",            row.names = TRUE)
write.csv(ds_legacy$p.value,            "R_diffSplice_legacy_p.csv",            row.names = TRUE)
write.csv(ds_legacy$gene.F,             "R_diffSplice_legacy_gene_F.csv",       row.names = TRUE)
write.csv(ds_legacy$gene.F.p.value,     "R_diffSplice_legacy_gene_F_p.csv",     row.names = TRUE)
write.csv(ds_legacy$gene.simes.p.value, "R_diffSplice_legacy_gene_simes_p.csv", row.names = TRUE)

# topSplice: every legal test x sort combination
cat("  topSplice (legal combinations)...\n")
topsplice_combinations <- list(
  c("simes", "p"), c("simes", "none"), c("simes", "NExons"),
  c("F",     "p"), c("F",     "none"), c("F",     "NExons"),
  c("t",     "p"), c("t",     "none"), c("t",     "logFC")
)
for (combo in topsplice_combinations) {
  tst    <- combo[1]
  sortby <- combo[2]
  ts <- topSplice(ds, coef = 2, test = tst, sort.by = sortby, number = Inf)
  write.csv(ts, sprintf("R_topSplice_%s_%s.csv", tst, sortby),
            row.names = FALSE)
}

# plotSplice substrate: top gene's exons.
cat("  plotSplice substrate...\n")
top_gene_idx <- which.min(ds$gene.F.p.value[, 2])
top_gene_id  <- ds$gene.genes[top_gene_idx, "GeneID"]
exon_mask    <- ds$genes[, "GeneID"] == top_gene_id
write.csv(data.frame(
  exon   = ds$genes[exon_mask, "ExonID"],
  log_fc = ds$coefficients[exon_mask, 2],
  t      = ds$t[exon_mask, 2],
  p      = ds$p.value[exon_mask, 2],
  gene   = top_gene_id
), "R_plotSplice_substrate.csv", row.names = FALSE)

cat("  visualisation, wsva and splicing fixtures complete.\n")


# -----------------------------------------------------------------------------
# GO / KEGG enrichment fixtures (goana, kegga, goanaTrend)
# -----------------------------------------------------------------------------
#
# These fixtures validate pylimma's enrichment.py port. They are deliberately
# self-contained: no GO.db / org.*.eg.db / KEGG REST dependency. The
# gene-pathway tables are hand-crafted and goana's hypergeometric counts /
# p-values are computed by feeding the goana data into kegga.default (whose
# algorithm is byte-identical to goana.default's modulo column names: both
# build the same incidence matrix and call phyper(...,lower.tail=FALSE) the
# same way). The Term column - which real goana fills from GO.db - is
# supplied via a synthetic pathway.names table; pylimma's port reads it as
# the optional 4th column of gene.pathway.
cat("\nGenerating enrichment fixtures (goana / kegga / goanaTrend)...\n")
library(limma)

# Universe of 200 genes
goana_universe <- sprintf("gene%03d", 1:200)
de_up   <- goana_universe[1:10]
de_down <- goana_universe[11:20]
goana_de_list <- list(Up = de_up, Down = de_down)

# GO gene-pathway table (10 GO terms; deliberately a mix of overlap with
# Up, Down, both, and neither so the fixture exercises non-trivial
# hypergeometric tails).
.make_go_rows <- function(go_id, ontology, term, idx) {
  data.frame(
    gene_id  = sprintf("gene%03d", idx),
    go_id    = go_id,
    ontology = ontology,
    term     = term,
    stringsAsFactors = FALSE
  )
}
goana_gp <- rbind(
  .make_go_rows("GO:0000001", "BP", "regulation of biological process", c(1,2,3,15,16)),
  .make_go_rows("GO:0000002", "BP", "metabolic process",                c(1,2,30,31,32,33)),
  .make_go_rows("GO:0000003", "BP", "signal transduction",              c(11,12,13,40,41)),
  .make_go_rows("GO:0000004", "BP", "transport",                        c(50,51,52,53,54,55,56)),
  .make_go_rows("GO:0000005", "CC", "membrane",                         c(4,5,60,61)),
  .make_go_rows("GO:0000006", "CC", "nucleus",                          c(70,80,90,100,110,120)),
  .make_go_rows("GO:0000007", "CC", "extracellular matrix",             c(14,15,16,17,100,101)),
  .make_go_rows("GO:0000008", "MF", "binding",                          c(1,2,3,4,5,6,7)),
  .make_go_rows("GO:0000009", "MF", "catalytic activity",               c(11,12,13,14,15)),
  .make_go_rows("GO:0000010", "MF", "transferase activity",             c(130,140,150,160,170))
)

write.csv(goana_gp,
          "R_goana_genepathway.csv", row.names = FALSE)
write.csv(data.frame(gene_id = goana_universe),
          "R_goana_universe.csv",    row.names = FALSE)
write.csv(data.frame(
            gene_id  = goana_universe,
            is_de_up   = goana_universe %in% de_up,
            is_de_down = goana_universe %in% de_down,
            stringsAsFactors = FALSE),
          "R_goana_de.csv", row.names = FALSE)

# KEGG gene-pathway table + names
.make_kegg <- function(path_id, idx) {
  data.frame(gene_id = sprintf("gene%03d", idx), path_id = path_id,
             stringsAsFactors = FALSE)
}
kegga_gp <- rbind(
  .make_kegg("path:hsa00010", c(1,2,11,12,50,51,52)),
  .make_kegg("path:hsa00020", c(3,4,5,60,61,62)),
  .make_kegg("path:hsa00030", c(13,14,70,71)),
  .make_kegg("path:hsa00040", c(80,81,82,83,84,85)),
  .make_kegg("path:hsa00050", c(1,2,3,15,16,17,18)),
  .make_kegg("path:hsa00060", c(11,12,13,90,91,92,93)),
  .make_kegg("path:hsa00070", c(100,101,102,103,104)),
  .make_kegg("path:hsa00080", c(6,7,8,110,111))
)
kegga_pn <- data.frame(
  path_id = sprintf("path:hsa%05d", c(10,20,30,40,50,60,70,80)),
  description = c("Glycolysis","Citrate cycle","Pentose phosphate","Fructose",
                  "Galactose","Starch","TCA","Oxidative phosphorylation"),
  stringsAsFactors = FALSE
)
write.csv(kegga_gp, "R_kegga_genepathway.csv", row.names = FALSE)
write.csv(kegga_pn, "R_kegga_pathnames.csv",   row.names = FALSE)

# goana algorithm via kegga.default + Term/Ont reshape
.term_lookup <- setNames(goana_gp$term, goana_gp$go_id)
.term_lookup <- .term_lookup[!duplicated(names(.term_lookup))]
.ont_lookup  <- setNames(goana_gp$ontology, goana_gp$go_id)
.ont_lookup  <- .ont_lookup[!duplicated(names(.ont_lookup))]
.goana_termnames <- data.frame(
  path_id     = names(.term_lookup),
  description = unname(.term_lookup),
  stringsAsFactors = FALSE
)
res_goana_alg <- kegga(de = goana_de_list, universe = goana_universe,
                       gene.pathway  = goana_gp[, c("gene_id","go_id")],
                       pathway.names = .goana_termnames,
                       trend = FALSE)
goana_result <- data.frame(
  Term   = res_goana_alg$Pathway,
  Ont    = unname(.ont_lookup[rownames(res_goana_alg)]),
  N      = res_goana_alg$N,
  Up     = res_goana_alg$Up,
  Down   = res_goana_alg$Down,
  P.Up   = res_goana_alg$P.Up,
  P.Down = res_goana_alg$P.Down,
  row.names = rownames(res_goana_alg),
  stringsAsFactors = FALSE
)
write.csv(goana_result, "R_goana_default.csv", row.names = TRUE)

# topGO with default sort and number=20
write.csv(topGO(goana_result, number = 20L),
          "R_top_go.csv", row.names = TRUE)

# kegga.default and topKEGG
res_kegga <- kegga(de = goana_de_list, universe = goana_universe,
                   gene.pathway  = kegga_gp,
                   pathway.names = kegga_pn,
                   trend = FALSE)
write.csv(res_kegga, "R_kegga_default.csv", row.names = TRUE)
write.csv(topKEGG(res_kegga, number = 20L),
          "R_top_kegg.csv", row.names = TRUE)

# goanaTrend stand-alone fixture
set.seed(2026)
.gt_n <- 200
.gt_cov <- sort(runif(.gt_n, 0, 1))
.gt_isde <- rep(0L, .gt_n)
# Inject DE genes with mild covariate-trend dependence so the lowess
# smoother produces a non-trivial probability curve.
.gt_isde[seq(1, .gt_n, by = 4)] <- 1L
.gt_isde[150:170] <- as.integer(runif(21) < 0.6)
write.csv(data.frame(is_de = .gt_isde, covariate = .gt_cov),
          "R_goanatrend_input.csv", row.names = FALSE)
.gt_prob <- goanaTrend(index.de = which(as.logical(.gt_isde)),
                       covariate = .gt_cov, plot = FALSE)
write.csv(data.frame(prob = .gt_prob),
          "R_goanatrend.csv", row.names = FALSE)

cat("  enrichment fixtures complete.\n")


# =============================================================================
# Public API: chooseLowessSpan, qqf, zscore family, loessFit, contrastAsCoef
# =============================================================================
cat("\nGenerating public-API fixtures (chooseLowessSpan, qqf, zscore, loessFit, contrastAsCoef)...\n")

set.seed(20260430)

# chooseLowessSpan: deterministic over n
.cls_n <- c(20, 50, 100, 500, 5000)
write.csv(
  data.frame(n = .cls_n, span = sapply(.cls_n, chooseLowessSpan)),
  "R_choose_lowess_span.csv", row.names = FALSE
)

# qqf: 100 random F-deviates, df1=5, df2=20
.qqf_y <- rf(100, df1 = 5, df2 = 20)
.qqf_qq <- qqf(.qqf_y, df1 = 5, df2 = 20, plot.it = FALSE)
write.csv(
  data.frame(x = .qqf_qq$x, y = .qqf_qq$y),
  "R_qqf.csv", row.names = FALSE
)

# zscore family: per-row distribution parameters
.zs_n <- 50
.zs_q_t <- rt(.zs_n, df = 8)
.zs_df_t <- rep(8.0, .zs_n)
.zs_q_gamma <- rgamma(.zs_n, shape = 3, rate = 0.5)
.zs_shape <- rep(3.0, .zs_n)
.zs_rate <- rep(0.5, .zs_n)
.zs_q_hyper <- rhyper(.zs_n, m = 30, n = 70, k = 20)
.zs_m <- rep(30L, .zs_n)
.zs_n_h <- rep(70L, .zs_n)
.zs_k <- rep(20L, .zs_n)

write.csv(
  data.frame(
    q_t = .zs_q_t, df = .zs_df_t,
    q_gamma = .zs_q_gamma, shape = .zs_shape, rate = .zs_rate,
    q_hyper = .zs_q_hyper, m = .zs_m, n = .zs_n_h, k = .zs_k
  ),
  "R_zscore_input.csv", row.names = FALSE
)

write.csv(
  data.frame(
    z_t = zscore(.zs_q_t, "t", df = .zs_df_t[1]),
    z_gamma = zscoreGamma(.zs_q_gamma, shape = .zs_shape, rate = .zs_rate),
    z_hyper = zscoreHyper(.zs_q_hyper, m = .zs_m, n = .zs_n_h, k = .zs_k)
  ),
  "R_zscore.csv", row.names = FALSE
)

# loessFit: a 200-point noisy sine
.lf_x <- seq(0, 2 * pi, length.out = 200)
.lf_y <- sin(.lf_x) + rnorm(200, sd = 0.2)
.lf_w <- runif(200, 0.5, 1.5)
.lf_fit <- loessFit(.lf_y, .lf_x, weights = .lf_w, span = 0.3)
write.csv(
  data.frame(
    x = .lf_x, y = .lf_y, w = .lf_w,
    fitted = .lf_fit$fitted, residuals = .lf_fit$residuals
  ),
  "R_loess_fit.csv", row.names = FALSE
)

# contrastAsCoef: a 2-condition design with one contrast
.cac_design <- model.matrix(~ 0 + factor(c("A", "A", "A", "B", "B", "B")))
colnames(.cac_design) <- c("A", "B")
.cac_contr <- makeContrasts(BvsA = B - A, levels = .cac_design)
.cac_res <- contrastAsCoef(.cac_design, .cac_contr)
write.csv(.cac_res$design, "R_contrast_as_coef_design.csv", row.names = FALSE)
write.csv(
  data.frame(qr_diag = diag(qr.R(qr(.cac_res$design)))),
  "R_contrast_as_coef_qr.csv", row.names = FALSE
)

cat("  Public-API fixtures complete.\n")

# =============================================================================
# weightedLowess (weighted_lowess.c) and loessFit: branch-forcing fixtures
# =============================================================================
cat("\nGenerating weightedLowess / loessFit branch fixtures...\n")

set.seed(20260930)

.wl_write <- function(case, x, y, weights, fit_loess, fit_lowess, delta_arg = NA_real_) {
  write.csv(
    data.frame(
      x = x, y = y, weights = if (is.null(weights)) NA_real_ else weights,
      delta_arg = delta_arg,
      fitted = fit_loess$fitted, residuals = fit_loess$residuals,
      robustness = fit_loess$weights, delta = fit_loess$delta,
      lowess_x = fit_lowess$x, lowess_y = fit_lowess$y
    ),
    sprintf("R_wl_%s.csv", case), row.names = FALSE
  )
}
.wl_both <- function(case, x, y, weights = NULL, delta = NULL, span = 0.3) {
  fl <- weightedLowess(x, y, weights = weights, delta = delta, span = span)
  fw <- weightedLowess(x, y, weights = weights, delta = delta, span = span, output.style = "lowess")
  .wl_write(case, x, y, weights, fl, fw, delta_arg = if (is.null(delta)) NA_real_ else delta)
  invisible(fl)
}

# More points than npts: delta > 0, anchors + linear interpolation
.wl_x <- round(runif(1000, 0, 10), 2)
.wl_y <- sin(.wl_x) + rnorm(1000, sd = 0.3)
.wl_w <- runif(1000, 0.2, 3)
.f <- .wl_both("interp", .wl_x, .wl_y, .wl_w)
stopifnot(.f$delta > 0, anyDuplicated(.wl_x) > 0)

# User-supplied delta
.f <- .wl_both("delta_given", .wl_x, .wl_y, .wl_w, delta = 0.05)

# Fewer points than npts: delta = 0, every point is an anchor
.f <- .wl_both("small_n", .wl_x[1:150], .wl_y[1:150], .wl_w[1:150])
stopifnot(.f$delta == 0)

# Unit weights, even n: the weighted median hits half the total weight exactly
.f <- .wl_both("unweighted", .wl_x[1:500], .wl_y[1:500])

# A large block of tied x: windows inside the block have zero width
# (dist < THRESHOLD) and the final anchor ties its predecessor (averaged
# interpolation)
.wl_x_tied <- c(rep(0, 400), sort(runif(596, 1, 10)), rep(10, 4))
.wl_y_tied <- rnorm(1000)
.f <- .wl_both("tied_block", .wl_x_tied, .wl_y_tied, runif(1000, 0.5, 1.5), span = 0.2)

# Constant response with one outlier: MAD is zero so robustness iterations stop
.wl_y_const <- rep(1, 300)
.wl_y_const[10] <- 5
.f <- .wl_both("mad_zero", .wl_x[1:300], .wl_y_const, .wl_w[1:300])
stopifnot(all(.f$weights == 1))

# loessFit branches -------------------------------------------------------
.lf_write <- function(case, y, x, weights, fit, span, min_weight = 1e-5, max_weight = 1e5) {
  write.csv(
    data.frame(
      y = y, x = x, weights = if (is.null(weights)) NA_real_ else weights,
      weights_null = is.null(weights), span = span,
      min_weight = min_weight, max_weight = max_weight,
      fitted = fit$fitted, residuals = fit$residuals
    ),
    sprintf("R_lf_%s.csv", case), row.names = FALSE
  )
}
.lf_case <- function(case, y, x, weights = NULL, span = 0.3, min_weight = 1e-5, max_weight = 1e5) {
  fit <- loessFit(y, x, weights = weights, span = span, min.weight = min_weight, max.weight = max_weight)
  .lf_write(case, y, x, weights, fit, span, min_weight, max_weight)
  invisible(fit)
}

.lf_y <- .wl_y[1:600]
.lf_x <- .wl_x[1:600]
.lf_w <- .wl_w[1:600]
.lf_y[c(3, 30)] <- NA
.lf_x[c(50)] <- NA

# Weighted, with non-finite y/x dropped
.lf_case("weighted_na", .lf_y, .lf_x, .lf_w)
# No weights -> base lowess()
.lf_case("unweighted", .lf_y, .lf_x)
# Equal weights treated as NULL -> base lowess()
.lf_case("equal_weights", .lf_y, .lf_x, rep(2, 600))
# Zero weights clamped to min.weight/max.weight as fitFDistUnequalDF1 does
.lf_w0 <- .lf_w
.lf_w0[seq(1, 600, by = 7)] <- 0
.lf_case("clamped_weights", .lf_y, .lf_x, .lf_w0, min_weight = 1e-8, max_weight = 1e2)
# min.weight = 0: too few positive weights -> weighted linear regression
.lf_case("few_positive", .lf_y[4:13], .lf_x[4:13], c(1, 0, 2, 0, 1, 0, 3, 0, 1, 1), min_weight = 0)
# min.weight = 0: a single positive weight
.lf_case("one_positive", .lf_y[4:13], .lf_x[4:13], c(0, 0, 2, 0, 0, 0, 0, 0, 0, 0), min_weight = 0)
# Positively weighted x all equal: slope aliased in lm.wfit
.lf_case("aliased_slope", .lf_y[4:13], c(rep(1, 5), 2:6), c(rep(1, 5), rep(0, 5)), min_weight = 0)
# span below 1/nobs -> fitted = y
.lf_case("span_tiny", .lf_y[4:13], .lf_x[4:13], .lf_w[4:13], span = 0.05)
# No finite observations
.lf_case("all_na", rep(NA_real_, 5), 1:5, rep(1, 5))

cat("  weightedLowess / loessFit branch fixtures complete.\n")

# =============================================================================
# qr() / dqrdc2 column pivoting and lm.series genewise branches
# =============================================================================
cat("\nGenerating dqrdc2 / lm.series branch fixtures...\n")

set.seed(20261001)

# qr(): pivot, rank and nonEstimable for matrices that force each dqrdc2 path
.qr_base <- cbind(1, rep(0:1, each = 4), rnorm(8))
.qr_near <- function(eps) cbind(.qr_base[, 1:2], .qr_base[, 2] + eps * rnorm(8), rnorm(8))
.qr_cases <- list(
  # Large-scale covariate: tol is relative to each column's own norm
  scaled_covariate = cbind(.qr_base[, 1:2], runif(8, 1.5e8, 2.5e8)),
  # Fewer rows than columns
  wide = matrix(rnorm(6), 2, 3),
  # A zero column ahead of estimable columns is cycled to the end
  zero_column_first = cbind(0, .qr_base),
  # An earlier column collinear with an even earlier one
  collinear_middle = cbind(.qr_base[, 1], 2 * .qr_base[, 1], .qr_base[, 2:3]),
  # Near-collinear columns either side of tol = 1e-7 (also forces the
  # reduced-norm recomputation branch)
  near_tol_dropped = .qr_near(1e-9),
  near_tol_kept = .qr_near(1e-5),
  all_zero = matrix(0, 5, 3),
  single_row = matrix(c(2, 0, 3), 1, 3)
)
.qr_long <- do.call(rbind, lapply(names(.qr_cases), function(nm) {
  m <- .qr_cases[[nm]]
  data.frame(case = nm, row = as.vector(row(m)), col = as.vector(col(m)), value = as.vector(m))
}))
write.csv(.qr_long, "R_qr_inputs.csv", row.names = FALSE)
.qr_out <- do.call(rbind, lapply(names(.qr_cases), function(nm) {
  q <- qr(.qr_cases[[nm]])
  ne <- nonEstimable(.qr_cases[[nm]])
  data.frame(case = nm, rank = q$rank, pivot = paste(q$pivot, collapse = ";"),
             non_estimable = if (is.null(ne)) "" else paste(ne, collapse = ";"))
}))
stopifnot(.qr_out$rank[.qr_out$case == "near_tol_dropped"] == 3,
          .qr_out$rank[.qr_out$case == "near_tol_kept"] == 4,
          .qr_out$rank[.qr_out$case == "all_zero"] == 0)
write.csv(.qr_out, "R_qr_outputs.csv", row.names = FALSE)

# lmFit: genewise fits that exercise the same pivoting inside lm.fit/lm.wfit
.lsq_write <- function(case, expr, design, fit, weights = NULL) {
  write.csv(expr, sprintf("R_lsq_%s_expr.csv", case), row.names = FALSE)
  write.csv(design, sprintf("R_lsq_%s_design.csv", case), row.names = FALSE)
  if (!is.null(weights)) write.csv(weights, sprintf("R_lsq_%s_weights.csv", case), row.names = FALSE)
  nb <- ncol(design)
  out <- data.frame(sigma = fit$sigma, df_residual = fit$df.residual)
  for (j in seq_len(nb)) {
    out[[paste0("coef_", j)]] <- fit$coefficients[, j]
    out[[paste0("stdev_", j)]] <- fit$stdev.unscaled[, j]
  }
  write.csv(out, sprintf("R_lsq_%s.csv", case), row.names = FALSE)
}

# Fast path: large-scale covariate
.lsq_design_scaled <- unname(.qr_cases$scaled_covariate)
.lsq_expr <- matrix(rnorm(20 * 8), 20, 8)
.f <- lmFit(.lsq_expr, .lsq_design_scaled)
stopifnot(!anyNA(.f$coefficients))
.lsq_write("scaled_covariate", .lsq_expr, .lsq_design_scaled, .f)

# Fast path: as many coefficients as samples -> df.residual = 0, sigma NA
.lsq_design_sat <- cbind(1, c(0, 1, 0), c(0, 0, 1))
.lsq_expr_sat <- matrix(rnorm(10 * 3), 10, 3)
.f <- lmFit(.lsq_expr_sat, .lsq_design_sat)
stopifnot(all(.f$df.residual == 0), all(is.na(.f$sigma)))
.lsq_write("saturated", .lsq_expr_sat, .lsq_design_sat, .f)

# Slow path: missing values giving n < p, a missing group and all-missing
.lsq_design3 <- cbind(1, rep(c(0, 1, 0), each = 3), rep(c(0, 0, 1), each = 3))
.lsq_expr3 <- matrix(rnorm(30 * 9), 30, 9)
.lsq_expr3[runif(length(.lsq_expr3)) < 0.1] <- NA
.lsq_expr3[1, ] <- c(1.2, rep(NA, 8))             # one observation
.lsq_expr3[2, ] <- c(NA, NA, NA, 0.4, NA, NA, 1.1, NA, NA)  # two obs, one per non-baseline group
.lsq_expr3[3, 4:6] <- NA                          # group 2 absent: zero column in subset
.lsq_expr3[4, ] <- NA                             # nothing observed
.f <- lmFit(.lsq_expr3, .lsq_design3)
stopifnot(.f$df.residual[1] == 0, .f$df.residual[4] == 0, is.na(.f$coefficients[3, 2]))
.lsq_write("missing", .lsq_expr3, .lsq_design3, .f)

# Slow path with probe weights: zero/negative weights become missing
.lsq_w <- matrix(runif(30 * 9, 0.2, 2), 30, 9)
.lsq_w[1:3, 1:2] <- 0
.lsq_w[5, 7:9] <- -1
.f <- lmFit(.lsq_expr3, .lsq_design3, weights = .lsq_w)
.lsq_write("weighted", .lsq_expr3, .lsq_design3, .f, weights = .lsq_w)

# Slow path, rank 0: the only observed rows of the design are zero.
# R's lm.series stops (chol2inv with size = 0); record the error.
.lsq_design_r0 <- cbind(c(0, 0, 0, 0, 1, 2, 3, 4))
.lsq_expr_r0 <- matrix(rnorm(5 * 8), 5, 8)
.lsq_expr_r0[1, 5:8] <- NA
write.csv(.lsq_expr_r0, "R_lsq_rank_zero_expr.csv", row.names = FALSE)
write.csv(.lsq_design_r0, "R_lsq_rank_zero_design.csv", row.names = FALSE)
.lsq_r0_error <- tryCatch({ lmFit(.lsq_expr_r0, .lsq_design_r0); NA_character_ },
                          error = function(e) conditionMessage(e))
stopifnot(!is.na(.lsq_r0_error))
write.csv(data.frame(r_error = .lsq_r0_error), "R_lsq_rank_zero_error.csv", row.names = FALSE)

cat("  dqrdc2 / lm.series branch fixtures complete.\n")

# =============================================================================
# fitFDistUnequalDF1: branch-forcing fixtures
# Each case forces a distinct R branch; the stored outputs let the Python test
# assert that the branch was actually taken (e.g. df2.shrunk present).
# =============================================================================
cat("\nGenerating fitFDistUnequalDF1 branch fixtures...\n")

set.seed(20260929)

.ffdu_write <- function(case, x, df1, fit, covariate = NULL, prior_weights = NULL,
                        robust = FALSE, span = NA_real_) {
  n <- length(x)
  out <- data.frame(
    x = x,
    df1 = rep_len(df1, n),
    covariate = if (is.null(covariate)) NA_real_ else covariate,
    prior_weights = if (is.null(prior_weights)) NA_real_ else prior_weights,
    df1_is_unit = length(df1) == 1L,
    robust = robust,
    span = span,
    scale = rep_len(fit$scale, n),
    df2 = fit$df2,
    df2_outlier = if (is.null(fit$df2.outlier)) NA_real_ else fit$df2.outlier,
    df2_shrunk = if (is.null(fit$df2.shrunk)) NA_real_ else fit$df2.shrunk
  )
  write.csv(out, sprintf("R_ffdu_%s.csv", case), row.names = FALSE)
}

# Variances and residual df from a genewise fit with missing values, as in a
# proteomics experiment: unequal df1 and a handful of high-variance genes.
.ffdu_n <- 400
.ffdu_expr <- matrix(rnorm(.ffdu_n * 8), .ffdu_n, 8)
.ffdu_expr[1:12, ] <- .ffdu_expr[1:12, ] * 3
.ffdu_expr[runif(length(.ffdu_expr)) < 0.1] <- NA
.ffdu_fit <- lmFit(.ffdu_expr, cbind(1, rep(0:1, each = 4)))
.ffdu_x <- .ffdu_fit$sigma^2
.ffdu_df1 <- .ffdu_fit$df.residual
.ffdu_amean <- .ffdu_fit$Amean
stopifnot(all(.ffdu_df1 > 0), length(unique(.ffdu_df1)) > 1)

# Right outliers -> refit, per-gene df2.shrunk, monotonic step
.f <- fitFDistUnequalDF1(.ffdu_x, .ffdu_df1, robust = TRUE)
stopifnot(!is.null(.f$df2.shrunk), length(unique(.f$df2.shrunk)) > 1)
.ffdu_write("shrink", .ffdu_x, .ffdu_df1, .f, robust = TRUE)

# Tied F-statistics inside the right tail (rank ties)
.ffdu_x_ties <- .ffdu_x
.ffdu_df1_ties <- .ffdu_df1
.top <- order(.ffdu_x / .f$scale, decreasing = TRUE)[1:6]
.ffdu_x_ties[.top[2:3]] <- .ffdu_x_ties[.top[4]]
.ffdu_df1_ties[.top[2:3]] <- .ffdu_df1_ties[.top[4]]
.f <- fitFDistUnequalDF1(.ffdu_x_ties, .ffdu_df1_ties, robust = TRUE)
stopifnot(!is.null(.f$df2.shrunk), .f$df2.shrunk[.top[4]] < .f$df2)
.ffdu_write("shrink_ties", .ffdu_x_ties, .ffdu_df1_ties, .f, robust = TRUE)

# Covariate trend + NA variances (zero prior weight) + robust
.ffdu_x_na <- .ffdu_x
.ffdu_x_na[c(20, 40, 60)] <- NA
.f <- fitFDistUnequalDF1(.ffdu_x_na, .ffdu_df1, covariate = .ffdu_amean, robust = TRUE)
stopifnot(!is.null(.f$df2.shrunk))
.ffdu_write("trend_robust", .ffdu_x_na, .ffdu_df1, .f, covariate = .ffdu_amean, robust = TRUE)

# As above with an explicit span; R's robust refit does not forward span
.f_span <- fitFDistUnequalDF1(.ffdu_x_na, .ffdu_df1, covariate = .ffdu_amean, span = 0.5, robust = TRUE)
stopifnot(!isTRUE(all.equal(.f_span$scale, .f$scale)))
.ffdu_write("trend_robust_span", .ffdu_x_na, .ffdu_df1, .f_span, covariate = .ffdu_amean, robust = TRUE, span = 0.5)

# Covariate trend + NA variances, non-robust (zero-weight loess floor)
.f <- fitFDistUnequalDF1(.ffdu_x_na, .ffdu_df1, covariate = .ffdu_amean)
.ffdu_write("trend_na", .ffdu_x_na, .ffdu_df1, .f, covariate = .ffdu_amean)

# Right-tail probability underflows to zero -> df2.outlier = 0.
# Needs a large, well-behaved background so the initial df2 stays high.
.ffdu_x_uf <- rchisq(5000, df = 50) / 50
.ffdu_x_uf[1] <- 1e20
.f <- fitFDistUnequalDF1(.ffdu_x_uf, 50, robust = TRUE)
stopifnot(identical(.f$df2.outlier, 0))
.ffdu_write("underflow", .ffdu_x_uf, 50, .f, robust = TRUE)

# Left outliers only -> refit returned without df2.shrunk.
# Random data essentially never satisfies RightP >= UniformP at every rank;
# a tied background (one shared average rank) with two small values does.
.ffdu_x_left <- rep(1, 200)
.ffdu_x_left[1:2] <- 0.01
.f <- fitFDistUnequalDF1(.ffdu_x_left, 6, robust = TRUE)
.f_nonrobust <- fitFDistUnequalDF1(.ffdu_x_left, 6)
stopifnot(is.null(.f$df2.shrunk), !isTRUE(all.equal(.f$scale, .f_nonrobust$scale)))
.ffdu_write("left_only", .ffdu_x_left, 6, .f, robust = TRUE)

# df1 supplied as a unit vector, robust
.ffdu_x_unit <- rchisq(.ffdu_n, df = 6) / 6
.ffdu_x_unit[11:16] <- .ffdu_x_unit[11:16] * 30
.f <- fitFDistUnequalDF1(.ffdu_x_unit, 6, robust = TRUE)
stopifnot(!is.null(.f$df2.shrunk))
.ffdu_write("unit_df1", .ffdu_x_unit, 6, .f, robust = TRUE)

# User prior weights, combined with NA x and df1 < 0.01
.ffdu_pw <- runif(.ffdu_n, 0.2, 2)
.ffdu_df1_small <- .ffdu_df1
.ffdu_df1_small[c(7, 8)] <- 0.001
.f <- fitFDistUnequalDF1(.ffdu_x_na, .ffdu_df1_small, prior.weights = .ffdu_pw)
.ffdu_write("prior_weights", .ffdu_x_na, .ffdu_df1_small, .f, prior_weights = .ffdu_pw)

# Exactly two informative values: covariate and robust are dropped
.ffdu_x_two <- c(0.5, 1.5, rep(0, 8))
.ffdu_cov_two <- seq(1, 10)
.f <- fitFDistUnequalDF1(.ffdu_x_two, 4, covariate = .ffdu_cov_two, robust = TRUE)
.ffdu_write("two_informative", .ffdu_x_two, 4, .f, covariate = .ffdu_cov_two, robust = TRUE)

# Fewer than two informative values -> NA hyperparameters
.ffdu_x_one <- c(0.5, rep(0, 9))
.f <- fitFDistUnequalDF1(.ffdu_x_one, 4)
stopifnot(is.na(.f$scale), is.na(.f$df2))
.ffdu_write("one_informative", .ffdu_x_one, 4, .f)

# Input checks: record R's error for each stop()
.ffdu_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.ffdu_errors <- data.frame(
  case = c("df1_length", "df1_na", "covariate_length", "covariate_na",
           "prior_weights_length", "prior_weights_na", "prior_weights_negative"),
  r_error = c(
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), c(4, 4))),
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), c(4, NA, 4))),
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), 4, covariate = c(1, 2))),
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), 4, covariate = c(1, NA, 3))),
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), 4, prior.weights = c(1, 1))),
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), 4, prior.weights = c(1, NA, 1))),
    .ffdu_err(fitFDistUnequalDF1(c(1, 2, 3), 4, prior.weights = c(1, -1, 1)))
  )
)
stopifnot(!anyNA(.ffdu_errors$r_error))
write.csv(.ffdu_errors, "R_ffdu_errors.csv", row.names = FALSE)

# End to end: eBayes robust (and trend + robust) on the unequal-df fit.
# A single-observation gene gives df = 0, which squeezeVar passes to
# fitFDistUnequalDF1 as an uninformative (zero-weight) value.
.ffdu_expr_e2e <- .ffdu_expr
.ffdu_expr_e2e[400, ] <- c(1.3, rep(NA, 7))
.ffdu_design <- cbind(1, rep(0:1, each = 4))
.ffdu_fit_e2e <- lmFit(.ffdu_expr_e2e, .ffdu_design)
stopifnot(.ffdu_fit_e2e$df.residual[400] == 0)
write.csv(.ffdu_expr_e2e, "R_ffdu_ebayes_expr.csv", row.names = FALSE)
for (.trend in c(FALSE, TRUE)) {
  .eb <- eBayes(.ffdu_fit_e2e, robust = TRUE, trend = .trend)
  stopifnot(length(unique(.eb$df.prior)) > 1)
  write.csv(
    data.frame(
      t = .eb$t[, 2], p_value = .eb$p.value[, 2], lods = .eb$lods[, 2],
      s2_post = .eb$s2.post, df_prior = .eb$df.prior,
      s2_prior = rep_len(.eb$s2.prior, .ffdu_n), df_total = .eb$df.total
    ),
    sprintf("R_ffdu_ebayes_robust%s.csv", if (.trend) "_trend" else ""),
    row.names = FALSE
  )
}

cat("  fitFDistUnequalDF1 branch fixtures complete.\n")

# =============================================================================
# fry / .fryEffects: branch-forcing fixtures
# =============================================================================
cat("\nGenerating fry branch fixtures...\n")

set.seed(20261002)

.fry_n <- 200
.fry_ids <- sprintf("gene%03d", seq_len(.fry_n))
.fry_expr <- matrix(rnorm(.fry_n * 8), .fry_n, 8, dimnames = list(.fry_ids, NULL))
.fry_expr[1:15, 5:8] <- .fry_expr[1:15, 5:8] + 1
.fry_expr[30:45, 5:8] <- .fry_expr[30:45, 5:8] - 0.7
.fry_expr[100:105, ] <- .fry_expr[100:105, ] * 4          # high-variance genes
.fry_design <- cbind(1, rep(0:1, each = 4))
.fry_symbols <- sprintf("SYM%03d", seq_len(.fry_n))
write.csv(data.frame(id = .fry_ids, symbol = .fry_symbols, .fry_expr, check.names = FALSE),
          "R_fry_expr.csv", row.names = FALSE)

# Equal residual variances for every gene (sign flips of one vector with
# values permuted within groups), so fitFDist gives df2 = Inf
.fry_base <- rnorm(8)
.fry_expr_null <- t(sapply(seq_len(.fry_n), function(g)
  sample(c(-1, 1), 1) * c(sample(.fry_base[1:4]), sample(.fry_base[5:8]))))
dimnames(.fry_expr_null) <- list(.fry_ids, NULL)
write.csv(data.frame(id = .fry_ids, .fry_expr_null, check.names = FALSE),
          "R_fry_expr_null.csv", row.names = FALSE)

.fry_sets <- list(up = 1:15, down = 30:45, mixed = c(1:5, 30:35), noise = 150:170)

# Gene sets in long form: case, set, member (1-based index or gene id), weight
.fry_set_rows <- list()
.fry_add_sets <- function(case, index) {
  rows <- lapply(seq_along(index), function(i) {
    s <- index[[i]]
    nm <- if (is.null(names(index))) NA_character_ else names(index)[i]
    if (is.data.frame(s)) {
      data.frame(case = case, set_number = i, set = nm, member = as.character(s[, 1]), weight = s[, 2])
    } else {
      data.frame(case = case, set_number = i, set = nm, member = as.character(s), weight = NA_real_)
    }
  })
  .fry_set_rows[[case]] <<- do.call(rbind, rows)
}
.fry_tab_write <- function(case, tab) {
  write.csv(data.frame(set = rownames(tab), tab, check.names = FALSE),
            sprintf("R_fry_%s.csv", case), row.names = FALSE)
}
.fry_case <- function(case, index, y = .fry_expr, ...) {
  if (!is.null(index)) .fry_add_sets(case, index)
  tab <- fry(y, index = index, design = .fry_design, ...)
  .fry_tab_write(case, tab)
  invisible(tab)
}

# standardize options
.fry_case("std_none", .fry_sets, standardize = "none", sort = "none")
.fry_case("std_residual", .fry_sets, standardize = "residual.sd", sort = "none")
.fry_case("std_p2", .fry_sets, standardize = "p2", sort = "none")
.fry_case("std_posterior", .fry_sets, standardize = "posterior.sd", sort = "none")

# posterior.sd when the prior df is infinite
.fry_eff_null <- limma:::.lmEffects(.fry_expr_null, .fry_design)
stopifnot(is.infinite(fitFDist(rowMeans(.fry_eff_null[, -1]^2), df1 = ncol(.fry_eff_null) - 1)$df2))
.fry_case("posterior_inf", .fry_sets, y = .fry_expr_null, sort = "none")

# robust hyperparameters: df2.shrunk is genewise
.fry_eff <- limma:::.lmEffects(.fry_expr, .fry_design)
.fry_rob <- fitFDistRobustly(rowMeans(.fry_eff[, -1]^2), df1 = ncol(.fry_eff) - 1)
stopifnot(length(unique(.fry_rob$df2.shrunk)) > 1)
.fry_case("posterior_robust", .fry_sets, robust = TRUE, sort = "none")
.fry_case("p2_robust", .fry_sets, standardize = "p2", robust = TRUE, sort = "none")

# trend.var: covariate = rowMeans(y)
.fry_case("trend", .fry_sets, trend.var = TRUE, sort = "none")

# NGenes == 1 set, and a single set (no FDR columns)
.fry_case("singleton", list(single = 7L, up = 1:15), sort = "none")
.fry_case("one_set", list(up = 1:15), sort = "none")

# index = NULL, and many unnamed sets (zero-padded default names)
.fry_case("index_null", NULL, sort = "none")
.fry_unnamed <- lapply(0:11, function(k) (k * 15 + 1):(k * 15 + 12))
.fry_case("unnamed_many", .fry_unnamed, sort = "none")

# Character ids, weighted data.frame sets (integer and character), gene.weights
.fry_case("character_ids", list(up = .fry_ids[1:15], down = .fry_ids[30:45]), sort = "none")
.fry_case("df_weights_int", list(w = data.frame(i = 1:15, w = seq(0.5, 2, length.out = 15)),
                                  plain = 30:45), sort = "none")
.fry_case("df_weights_chr", list(w = data.frame(i = .fry_ids[c(3, 1, 2, 30:35)], w = c(2, 1, 1, rep(-1, 6)))),
          sort = "none")
.fry_gene_w <- seq(0.5, 1.5, length.out = .fry_n)
write.csv(data.frame(gene_weights = .fry_gene_w), "R_fry_input_gene_weights.csv", row.names = FALSE)
.fry_case("gene_weights", .fry_sets, gene.weights = .fry_gene_w, sort = "none")

# geneid as an annotation column and as a vector
.fry_el <- new("EList", list(E = .fry_expr, genes = data.frame(Symbol = .fry_symbols, row.names = .fry_ids)))
.fry_case("geneid_column", list(up = .fry_symbols[1:15], down = .fry_symbols[30:45]),
          y = .fry_el, geneid = "Symbol", sort = "none")
.fry_case("geneid_vector", list(up = .fry_symbols[1:15], down = .fry_symbols[30:45]),
          geneid = .fry_symbols, sort = "none")

# sort options
.fry_case("sort_directional", .fry_sets, sort = "directional")
.fry_case("sort_mixed", .fry_sets, sort = "mixed")
.fry_case("sort_true", .fry_sets, sort = TRUE)
.fry_case("sort_false", .fry_sets, sort = FALSE)

# ... arguments: array weights, observation weights, block, contrast vector
.fry_aw <- c(1, 0.5, 2, 1, 1.5, 1, 0.8, 1.2)
.fry_obs_w <- matrix(runif(.fry_n * 8, 0.3, 2), .fry_n, 8)
write.csv(data.frame(array_weights = .fry_aw), "R_fry_input_array_weights.csv", row.names = FALSE)
write.csv(.fry_obs_w, "R_fry_input_obs_weights.csv", row.names = FALSE)
.fry_case("array_weights", .fry_sets, array.weights = .fry_aw, sort = "none")
.fry_case("obs_weights", .fry_sets, weights = .fry_obs_w, sort = "none")
.fry_case("block", .fry_sets, block = rep(1:4, 2), correlation = 0.3, sort = "none")
.fry_case("contrast_vector", .fry_sets, contrast = c(-1, 1), sort = "none")

write.csv(do.call(rbind, .fry_set_rows), "R_fry_sets.csv", row.names = FALSE)

# Errors raised by fry.default / .fryEffects
.fry_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.fry_errors <- data.frame(
  case = c("gene_weights_length", "block_without_correlation", "index_empty",
           "duplicate_set_names", "duplicate_ids_in_df_set", "df_set_without_weights",
           "geneid_length"),
  r_error = c(
    .fry_err(fry(.fry_expr, .fry_sets, .fry_design, gene.weights = 1:3)),
    .fry_err(fry(.fry_expr, .fry_sets, .fry_design, block = rep(1:4, 2))),
    .fry_err(fry(.fry_expr, list(), .fry_design)),
    .fry_err(fry(.fry_expr, list(a = 1:5, a = 6:10), .fry_design)),
    .fry_err(fry(.fry_expr, list(w = data.frame(i = .fry_ids[c(1, 1)], w = c(1, 2))), .fry_design)),
    .fry_err(fry(.fry_expr, list(w = data.frame(i = 1:5)), .fry_design)),
    .fry_err(fry(.fry_expr, .fry_sets, .fry_design, geneid = .fry_symbols[1:10]))
  )
)
stopifnot(!anyNA(.fry_errors$r_error))
write.csv(.fry_errors, "R_fry_errors.csv", row.names = FALSE)

cat("  fry branch fixtures complete.\n")

# =============================================================================
# decideTests (MArrayLM and default methods): branch-forcing fixtures
# =============================================================================
cat("\nGenerating decideTests branch fixtures...\n")

set.seed(20261003)

.dt_design <- cbind(1, rep(c(0, 1, 0), each = 4), rep(c(0, 0, 1), each = 4))
.dt_n <- 300
.dt_expr <- matrix(rnorm(.dt_n * 12), .dt_n, 12)
.dt_expr[1:30, 5:8] <- .dt_expr[1:30, 5:8] + 1.5
.dt_expr[20:50, 9:12] <- .dt_expr[20:50, 9:12] - 1.2
# Genes whose group-2 and group-3 samples are identical: tied |t| for coefs 2, 3
.dt_expr[51:55, 9:12] <- .dt_expr[51:55, 5:8] <- .dt_expr[51:55, 5:8] + 3
.dt_expr_na <- .dt_expr
.dt_expr_na[runif(length(.dt_expr_na)) < 0.1] <- NA
.dt_expr_na[56, 5:8] <- NA                          # coefficient 2 not estimable
write.csv(.dt_expr, "R_dt_expr.csv", row.names = FALSE)
write.csv(.dt_expr_na, "R_dt_expr_na.csv", row.names = FALSE)

# Equal residual variances -> df.prior = Inf, so eBayes caps df.total while
# .classifyTestsP / classifyTestsF use df.residual + df.prior uncapped
.dt_base <- rnorm(12)
.dt_expr_inf <- t(sapply(seq_len(.dt_n), function(g)
  c(sample(.dt_base[1:4]), sample(.dt_base[5:8]), sample(.dt_base[9:12]))))
.dt_expr_inf[1:20, 5:8] <- .dt_expr_inf[1:20, 5:8] + 2
write.csv(.dt_expr_inf, "R_dt_expr_inf.csv", row.names = FALSE)

.dt_fit <- eBayes(lmFit(.dt_expr, .dt_design))
.dt_fit_na <- eBayes(lmFit(.dt_expr_na, .dt_design))
.dt_fit_inf <- eBayes(lmFit(.dt_expr_inf, .dt_design))
stopifnot(anyNA(.dt_fit_na$p.value), anyNA(.dt_fit_na$F.p.value),
          is.infinite(.dt_fit_inf$df.prior),
          any(.dt_fit$t[51:55, 2] == .dt_fit$t[51:55, 3]))

.dt_rows <- list()
.dt_record <- function(case, data, method, adjust, p_value = 0.05, lfc = 0, fn = decideTests, ...) {
  res <- unclass(fn(get(sprintf(".dt_%s", data)), method = method, adjust.method = adjust,
                    p.value = p_value, lfc = lfc, ...))
  write.csv(res, sprintf("R_dt_%s.csv", case), row.names = FALSE)
  .dt_rows[[case]] <<- data.frame(case = case, data = data, method = method, adjust = adjust,
                                  p_value = p_value, lfc = lfc)
}

# MArrayLM: separate / global with every adjust method, NA p-values, lfc
for (.adj in c("BH", "fdr", "none", "bonferroni", "holm", "BY")) {
  .dt_record(sprintf("fit_separate_%s", .adj), "fit_na", "separate", .adj)
}
.dt_record("fit_global_BH", "fit_na", "global", "BH")
.dt_record("fit_global_holm", "fit_na", "global", "holm")
.dt_record("fit_separate_lfc", "fit_na", "separate", "BH", lfc = 0.5)
# Partial matching of method / adjust.method
.dt_record("fit_partial_args", "fit_na", "sep", "bonf")

# MArrayLM: hierarchical and nestedF with every adjust method
for (.adj in c("BH", "none", "bonferroni", "holm", "BY")) {
  .dt_record(sprintf("fit_hierarchical_%s", .adj), "fit", "hierarchical", .adj)
  .dt_record(sprintf("fit_nestedF_%s", .adj), "fit", "nestedF", .adj)
}
.dt_record("fit_hierarchical_lfc", "fit", "hierarchical", "BH", lfc = 0.5)
# Uncapped df in .classifyTestsP / classifyTestsF
.dt_record("fit_inf_hierarchical", "fit_inf", "hierarchical", "BH")
.dt_record("fit_inf_nestedF", "fit_inf", "nestedF", "BH")

# Few genes, so the pooled df is small: capping df.total at it changes the
# hierarchical decision, which R makes with the uncapped df
set.seed(7)
.dt_base_small <- rnorm(12)
.dt_expr_inf_small <- t(sapply(1:4, function(g)
  c(sample(.dt_base_small[1:4]), sample(.dt_base_small[5:8]), sample(.dt_base_small[9:12]))))
.dt_expr_inf_small[1, 5:8] <- .dt_expr_inf_small[1, 5:8] + 2.9
write.csv(.dt_expr_inf_small, "R_dt_expr_inf_small.csv", row.names = FALSE)
.dt_fit_inf_small <- eBayes(lmFit(.dt_expr_inf_small, .dt_design))
.dt_capped <- .dt_fit_inf_small
.dt_capped$df.prior <- .dt_fit_inf_small$df.total - .dt_fit_inf_small$df.residual
stopifnot(is.infinite(.dt_fit_inf_small$df.prior),
          !identical(unclass(decideTests(.dt_fit_inf_small, method = "hierarchical"))[],
                     unclass(decideTests(.dt_capped, method = "hierarchical"))[]))
.dt_record("fit_inf_small_hierarchical", "fit_inf_small", "hierarchical", "BH")

# Automatic eBayes when p.value is absent
.dt_lmfit <- lmFit(.dt_expr, .dt_design)
.dt_record("fit_auto_ebayes", "lmfit", "separate", "BH")

# Default method: a matrix of p-values, with coefficients / tstat / none.
# (decideTests.default stops on NA p-values; see the error cases.)
.dt_p <- .dt_fit$p.value
.dt_coef <- .dt_fit$coefficients
.dt_tstat <- .dt_fit$t
.dt_gw <- .dt_fit$F.p.value
write.csv(.dt_p, "R_dt_input_p.csv", row.names = FALSE)
write.csv(.dt_coef, "R_dt_input_coef.csv", row.names = FALSE)
write.csv(.dt_tstat, "R_dt_input_tstat.csv", row.names = FALSE)
write.csv(data.frame(genewise = .dt_gw), "R_dt_input_genewise.csv", row.names = FALSE)

.dt_default <- function(case, method, adjust, ...) {
  res <- unclass(decideTests(.dt_p, method = method, adjust.method = adjust, ...))
  attributes(res) <- list(dim = dim(res))
  write.csv(res, sprintf("R_dt_%s.csv", case), row.names = FALSE)
}
.dt_default("p_separate_nocoef", "separate", "BH")
.dt_default("p_separate_coef_lfc", "separate", "BH", coefficients = .dt_coef, lfc = 0.5)
.dt_default("p_global_tstat", "global", "holm", tstat = .dt_tstat)
for (.adj in c("BH", "none", "bonferroni", "holm", "BY")) {
  .dt_default(sprintf("p_hierarchical_%s", .adj), "hierarchical", .adj, coefficients = .dt_coef)
}
.dt_default("p_hierarchical_nocoef", "hierarchical", "BH")
# Supplied genewise.p.value: R only defines ngenes when it computes Simes
# p-values itself, so every adjust.method except "none" errors (see errors).
# pylimma follows the intended behaviour, ngenes = nrow(p); the reference
# comes from decideTests.default with that single line added before switch().
.dt_default("p_hierarchical_genewise_none", "hierarchical", "none", genewise.p.value = .dt_gw)
.dt_decide_intended <- limma:::decideTests.default
.dt_body <- as.list(body(.dt_decide_intended))
.dt_switch <- which(vapply(.dt_body, function(e) is.call(e) && identical(e[[1]], as.name("switch")), TRUE))
body(.dt_decide_intended) <- as.call(append(.dt_body, list(quote(ngenes <- nrow(p))), after = .dt_switch - 1))
environment(.dt_decide_intended) <- asNamespace("limma")
# The patch changes nothing where R itself works
stopifnot(identical(
  unclass(.dt_decide_intended(.dt_p, method = "hierarchical", coefficients = .dt_coef))[],
  unclass(decideTests(.dt_p, method = "hierarchical", coefficients = .dt_coef))[]
))
for (.adj in c("BH", "bonferroni", "holm", "BY")) {
  .res <- unclass(.dt_decide_intended(.dt_p, method = "hierarchical", adjust.method = .adj,
                                      genewise.p.value = .dt_gw))
  attributes(.res) <- list(dim = dim(.res))
  write.csv(.res, sprintf("R_dt_p_hierarchical_genewise_%s_intended.csv", .adj), row.names = FALSE)
}

write.csv(do.call(rbind, .dt_rows), "R_dt_cases.csv", row.names = FALSE)

# Errors
.dt_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.dt_errors <- data.frame(
  case = c("fit_hierarchical_na", "fit_nestedF_na", "fit_bad_method", "p_nestedF", "p_out_of_range",
           "p_coef_dims", "p_with_na", "p_hierarchical_genewise_bh"),
  r_error = c(
    .dt_err(decideTests(.dt_fit_na, method = "hierarchical")),
    .dt_err(decideTests(.dt_fit_na, method = "nestedF")),
    .dt_err(decideTests(.dt_fit, method = "bogus")),
    .dt_err(decideTests(.dt_p, method = "nestedF")),
    .dt_err(decideTests(.dt_p * 2, method = "separate")),
    .dt_err(decideTests(.dt_p, coefficients = .dt_coef[, 1:2])),
    .dt_err(decideTests(.dt_fit_na$p.value)),
    .dt_err(decideTests(.dt_p, method = "hierarchical", genewise.p.value = .dt_gw))
  )
)
stopifnot(!anyNA(.dt_errors$r_error))
write.csv(.dt_errors, "R_dt_errors.csv", row.names = FALSE)

cat("  decideTests branch fixtures complete.\n")

# =============================================================================
# topTable: coefficient selection by name and sort.by / resort.by matching
# =============================================================================
cat("\nGenerating topTable argument-matching fixtures...\n")

set.seed(20261004)

.tt_design <- cbind(Intercept = 1, B = rep(c(0, 1, 0), each = 4), C = rep(c(0, 0, 1), each = 4))
.tt_expr <- matrix(rnorm(100 * 12), 100, 12, dimnames = list(sprintf("g%03d", 1:100), NULL))
.tt_expr[1:10, 5:8] <- .tt_expr[1:10, 5:8] + 2
.tt_expr[5:20, 9:12] <- .tt_expr[5:20, 9:12] - 1.5
write.csv(data.frame(id = rownames(.tt_expr), .tt_expr, check.names = FALSE), "R_tt_expr.csv", row.names = FALSE)
.tt_fit <- eBayes(lmFit(.tt_expr, .tt_design))
.tt_cfit <- eBayes(contrasts.fit(lmFit(.tt_expr, .tt_design),
                                 makeContrasts(CvsB = C - B, levels = .tt_design)))

.tt_write <- function(case, tab) {
  write.csv(data.frame(id = rownames(tab), tab, check.names = FALSE),
            sprintf("R_tt_%s.csv", case), row.names = FALSE)
}
.tt_write("coef_name", topTable(.tt_fit, coef = "B", number = Inf, sort.by = "none"))
.tt_write("coef_names_f", topTable(.tt_fit, coef = c("B", "C"), number = 15))
.tt_write("contrast_name", topTable(.tt_cfit, coef = "CvsB", number = 15))
# Default coef with an "(Intercept)" column: R drops it and subsets the fit,
# which regenerates F from the remaining coefficients
.tt_design_int <- .tt_design
colnames(.tt_design_int)[1] <- "(Intercept)"
.tt_fit_int <- eBayes(lmFit(.tt_expr, .tt_design_int))
stopifnot(!isTRUE(all.equal(topTable(.tt_fit_int, number = Inf, sort.by = "none")$F, .tt_fit_int$F)))
.tt_write("default_drops_intercept", topTable(.tt_fit_int, number = 15))
.tt_write("sort_partial", topTable(.tt_fit, coef = "B", number = 15, sort.by = "Ave"))
.tt_write("sort_alias_M", topTable(.tt_fit, coef = "B", number = 15, sort.by = "M"))
.tt_write("resort_logfc", topTable(.tt_fit, coef = "B", number = 15, resort.by = "logFC"))
.tt_write("resort_partial", topTable(.tt_fit, coef = "B", number = 15, resort.by = "Am"))

.tt_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.tt_errors <- data.frame(
  case = c("sort_by_invalid", "resort_by_none", "resort_by_invalid"),
  r_error = c(
    .tt_err(topTable(.tt_fit, coef = "B", sort.by = "pvalue")),
    .tt_err(topTable(.tt_fit, coef = "B", resort.by = "none")),
    .tt_err(topTable(.tt_fit, coef = "B", resort.by = "adj"))
  )
)
stopifnot(!anyNA(.tt_errors$r_error))
write.csv(.tt_errors, "R_tt_errors.csv", row.names = FALSE)

cat("  topTable argument-matching fixtures complete.\n")

# =============================================================================
# roast / mroast with user-supplied var.prior and df.prior
# Deterministic outputs only (active proportions); p-values depend on the RNG.
# =============================================================================
cat("\nGenerating roast user-prior fixtures...\n")

set.seed(20261005)

.ro_n <- 120
.ro_expr <- matrix(rnorm(.ro_n * 8), .ro_n, 8)
.ro_expr[1:12, 5:8] <- .ro_expr[1:12, 5:8] + 1.2
.ro_expr[30:40, 5:8] <- .ro_expr[30:40, 5:8] - 1
.ro_design <- cbind(1, rep(0:1, each = 4))
write.csv(.ro_expr, "R_ro_expr.csv", row.names = FALSE)

.ro_var_prior <- runif(.ro_n, 0.5, 1.5)
.ro_df_prior <- ifelse(seq_len(.ro_n) %% 3 == 0, Inf, 4)
write.csv(data.frame(var_prior = .ro_var_prior, df_prior = .ro_df_prior), "R_ro_priors.csv", row.names = FALSE)

.ro_single <- function(case, var_prior, df_prior) {
  r <- roast(.ro_expr, index = 1:20, design = .ro_design, var.prior = var_prior, df.prior = df_prior, nrot = 99)
  write.csv(data.frame(direction = rownames(r$p.value), active_prop = r$p.value$Active.Prop),
            sprintf("R_ro_%s.csv", case), row.names = FALSE)
  invisible(r)
}
.r <- .ro_single("roast_inf", 1, Inf)
stopifnot(any(.r$p.value$Active.Prop > 0))
.ro_single("roast_finite", 1, 4)

.ro_sets <- list(a = 1:20, b = 25:45, c = 60:90)
.ro_m <- mroast(.ro_expr, .ro_sets, .ro_design, var.prior = .ro_var_prior, df.prior = .ro_df_prior,
                nrot = 99, sort = "none")
stopifnot(any(.ro_m$PropUp > 0), any(.ro_m$PropDown > 0))
write.csv(data.frame(set = rownames(.ro_m), NGenes = .ro_m$NGenes, PropDown = .ro_m$PropDown, PropUp = .ro_m$PropUp),
          "R_ro_mroast_vector_priors.csv", row.names = FALSE)

cat("  roast user-prior fixtures complete.\n")

# =============================================================================
# Input dispatch: one R pipeline that every supported Python input container
# (ndarray, DataFrame incl. nullable dtypes, AnnData dense/sparse/h5ad, EList)
# must reproduce
# =============================================================================
cat("\nGenerating input-dispatch fixtures...\n")

set.seed(20261006)

.io_n <- 80
.io_ids <- sprintf("P%03d", seq_len(.io_n))
.io_expr <- matrix(rnorm(.io_n * 8, 20, 1.5), .io_n, 8, dimnames = list(.io_ids, sprintf("s%d", 1:8)))
.io_expr[1:10, 5:8] <- .io_expr[1:10, 5:8] + 2
.io_expr[runif(length(.io_expr)) < 0.08] <- NA
.io_group <- factor(rep(c("A", "B"), each = 4))
.io_design <- model.matrix(~ .io_group)
colnames(.io_design) <- c("(Intercept)", "groupB")
write.csv(data.frame(id = .io_ids, symbol = sprintf("GENE%03d", seq_len(.io_n)), .io_expr, check.names = FALSE),
          "R_io_expr.csv", row.names = FALSE)

.io_write <- function(case, tab) {
  write.csv(data.frame(id = rownames(tab), tab, check.names = FALSE), sprintf("R_io_%s.csv", case), row.names = FALSE)
}
.io_fit <- lmFit(.io_expr, .io_design)
.io_write("pipeline", topTable(eBayes(.io_fit), coef = "groupB", number = Inf, sort.by = "none"))
.io_write("contrasts_by_name",
          topTable(eBayes(contrasts.fit(.io_fit, coefficients = "groupB")), number = Inf, sort.by = "none"))

# data.frame whose first column is non-numeric: treated as gene IDs
.io_df <- data.frame(symbol = sprintf("GENE%03d", seq_len(.io_n)), .io_expr, check.names = FALSE)
.io_write("dataframe_id_column", topTable(eBayes(lmFit(.io_df, .io_design)), coef = "groupB",
                                          number = Inf, sort.by = "none"))

.io_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.io_errors <- data.frame(
  case = c("id_column_not_first", "two_nonnumeric_columns"),
  r_error = c(
    .io_err(lmFit(data.frame(.io_expr, symbol = .io_ids, check.names = FALSE), .io_design)),
    .io_err(lmFit(data.frame(a = .io_ids, b = .io_ids, .io_expr, check.names = FALSE), .io_design))
  )
)
stopifnot(!anyNA(.io_errors$r_error))
write.csv(.io_errors, "R_io_errors.csv", row.names = FALSE)

cat("  input-dispatch fixtures complete.\n")

# =============================================================================
# Interface, fit-slot and edge-case fixtures (full-output references for the
# tests in "Interface, fit-slot and edge-case parity")
# =============================================================================
cat("\nGenerating interface, fit-slot and edge-case fixtures...\n")

set.seed(20261007)

.ie_csv <- function(name, x, ids = NULL) {
  out <- if (is.null(ids)) as.data.frame(x) else data.frame(id = ids, x, check.names = FALSE)
  write.csv(out, sprintf("R_ie_%s.csv", name), row.names = FALSE)
}
.ie_table <- function(name, tab) .ie_csv(name, tab, ids = rownames(tab))
.ie_canonical <- function(m) apply(as.matrix(m), 2, function(v) {
  s <- sign(v[which.max(abs(v))])
  if (s == 0) v else v * s
})

# Shared two-group data (30 genes x 8 samples) and designs
.ie_X <- matrix(rnorm(30 * 8), 30, 8, dimnames = list(sprintf("g%02d", 1:30), NULL))
.ie_X[1:5, 5:8] <- .ie_X[1:5, 5:8] + 2
.ie_csv("X", .ie_X, ids = rownames(.ie_X))
.ie_design_int <- cbind("(Intercept)" = 1, groupB = rep(0:1, each = 4))
.ie_design_cm <- cbind(groupA = rep(1:0, each = 4), groupB = rep(0:1, each = 4))

# topTable(coef=NULL) drops the "(Intercept)" column
.ie_table("intercept_strip", topTable(eBayes(lmFit(.ie_X, .ie_design_int)), number = Inf, sort.by = "none"))

# contrasts.fit selecting a coefficient by name, and by 1-based index
.ie_fit_cm <- lmFit(.ie_X, .ie_design_cm)
.ie_cf_name <- contrasts.fit(.ie_fit_cm, coefficients = "groupB")
.ie_cf_int <- contrasts.fit(.ie_fit_cm, coefficients = 2)
stopifnot(identical(.ie_cf_name$coefficients, .ie_cf_int$coefficients))
for (.nm in c("coefficients", "stdev.unscaled", "cov.coefficients")) {
  .ie_csv(sprintf("contrasts_by_name_%s", gsub(".", "_", .nm, fixed = TRUE)), .ie_cf_name[[.nm]])
}

# Hand-built fit with |coef| exactly at the lfc boundary for topTable / decideTests
.ie_hb <- new("MArrayLM", list(
  coefficients = cbind(0, c(2, 1, 0.5, 1.5)), stdev.unscaled = matrix(0.1, 4, 2),
  sigma = rep(0.1, 4), df.residual = rep(10, 4), cov.coefficients = diag(2) * 0.01, Amean = rep(0, 4)))
.ie_hb <- eBayes(.ie_hb)
.ie_hb_tt <- topTable(.ie_hb, coef = 2, lfc = 1, number = Inf, sort.by = "none")
stopifnot(1 %in% .ie_hb_tt$logFC)
.ie_table("lfc_boundary_toptable", .ie_hb_tt)
.ie_hb_dt <- unclass(decideTests(.ie_hb, lfc = 1))
stopifnot(.ie_hb_dt[2, 2] == 0)
attributes(.ie_hb_dt) <- list(dim = dim(.ie_hb_dt))
.ie_csv("lfc_boundary_decidetests", .ie_hb_dt)

# EList input to voom (R drops the EList slots; weights passed explicitly)
# and to vooma (R uses y$design, ignores y$weights)
.ie_counts <- matrix(rpois(50 * 6, 20) + 1, 50, 6)
.ie_w <- matrix(runif(50 * 6, 0.5, 2), 50, 6)
.ie_d6 <- cbind(1, c(0, 0, 0, 1, 1, 1))
.ie_csv("counts", .ie_counts)
.ie_csv("elist_weights", .ie_w)
.ie_voom_w <- voom(.ie_counts, .ie_d6, weights = .ie_w)
.ie_csv("voom_explicit_weights_E", .ie_voom_w$E)
.ie_csv("voom_explicit_weights_weights", .ie_voom_w$weights)
.ie_logc <- log2(.ie_counts + 1)
.ie_vooma <- vooma(new("EList", list(E = .ie_logc, weights = .ie_w, design = .ie_d6)))
stopifnot(isTRUE(all.equal(.ie_vooma$weights, vooma(.ie_logc, .ie_d6)$weights)))
.ie_csv("vooma_elist_weights", .ie_vooma$weights)

# lmFit on an EList carries genes through to topTable
.ie_genes <- data.frame(ID = sprintf("ENSG%05d", 1:30), symbol = sprintf("SYM%d", 1:30))
.ie_el <- new("EList", list(E = unname(.ie_X), genes = .ie_genes))
.ie_table("elist_genes_toptable",
          topTable(eBayes(lmFit(.ie_el, .ie_design_int)), coef = 2, number = Inf, sort.by = "none"))

# plotRLDF: training scores (canonical sign per dimension) and singular values
.ie_y3 <- matrix(rnorm(200 * 12), 200, 12)
.ie_d3 <- cbind(1, rep(c(0, 1, 0), each = 4), rep(c(0, 0, 1), each = 4))
.ie_csv("rldf_y", .ie_y3)
.ie_rldf <- plotRLDF(.ie_y3, .ie_d3, plot = FALSE)
.ie_csv("rldf_training", .ie_canonical(.ie_rldf$training))
.ie_csv("rldf_singular_values", data.frame(d = .ie_rldf$singular.values))

# Fit slots set by lmFit / eBayes
.ie_csv("fit_slots", data.frame(
  method_ls = lmFit(.ie_X, .ie_design_int, method = "ls")$method,
  method_robust = suppressWarnings(lmFit(.ie_X, .ie_design_int, method = "robust"))$method,
  proportion = eBayes(lmFit(.ie_X, .ie_design_int), proportion = 0.05)$proportion))

# contrasts.fit with an empty contrast matrix: every slot's dimensions
.ie_d3b <- cbind(1, rep(0:1, each = 4), rep(0:1, 4))
.ie_fit3 <- lmFit(.ie_X, .ie_d3b)
.ie_cf_empty <- contrasts.fit(.ie_fit3, contrasts = matrix(0, 3, 0))
.ie_csv("contrasts_empty_dims", do.call(rbind, lapply(
  c("coefficients", "stdev.unscaled", "cov.coefficients", "contrasts"),
  function(nm) data.frame(slot = nm, nrow = NROW(.ie_cf_empty[[nm]]), ncol = NCOL(.ie_cf_empty[[nm]])))))

# contrasts.fit with coefficients that are zero in every contrast column
.ie_X10 <- matrix(rnorm(40 * 10), 40, 10)
.ie_d10 <- cbind(1, rep(0:1, each = 5), rep(0:1, 5))
.ie_csv("X10", .ie_X10)
.ie_cf_zero <- contrasts.fit(eBayes(lmFit(.ie_X10, .ie_d10)), contrasts = matrix(c(0, 0, 1), ncol = 1))
for (.nm in c("coefficients", "stdev.unscaled", "cov.coefficients")) {
  .ie_csv(sprintf("contrasts_all_zero_%s", gsub(".", "_", .nm, fixed = TRUE)), .ie_cf_zero[[.nm]])
}

# lmFit(method="robust") forwards ... to MASS::rlm: maxit = 2 changes the fit
.ie_Xr <- matrix(rnorm(20 * 6), 20, 6)
.ie_Xr[cbind(1:20, sample(1:6, 20, replace = TRUE))] <- .ie_Xr[cbind(1:20, sample(1:6, 20, replace = TRUE))] + 8
.ie_d6r <- cbind(1, rep(0:1, each = 3))
.ie_csv("robust_X", .ie_Xr)
.ie_rob2 <- suppressWarnings(lmFit(.ie_Xr, .ie_d6r, method = "robust", maxit = 2))
.ie_rob_default <- suppressWarnings(lmFit(.ie_Xr, .ie_d6r, method = "robust"))
stopifnot(!isTRUE(all.equal(.ie_rob2$coefficients, .ie_rob_default$coefficients)))
.ie_csv("robust_maxit2_coefficients", .ie_rob2$coefficients)
.ie_csv("robust_maxit2_stdev_unscaled", .ie_rob2$stdev.unscaled)
.ie_csv("robust_maxit2_sigma", data.frame(sigma = .ie_rob2$sigma))

# classifyTestsF on a rank-deficient fit (duplicated design column)
.ie_Xd <- matrix(rnorm(30 * 6), 30, 6)
.ie_Xd[1:6, 4:6] <- .ie_Xd[1:6, 4:6] + 3
.ie_dd <- cbind(1, c(0, 0, 0, 1, 1, 1), c(0, 0, 0, 1, 1, 1))
.ie_csv("rankdef_X", .ie_Xd)
.ie_rd <- suppressMessages(eBayes(lmFit(.ie_Xd, .ie_dd)))
.ie_ctf <- unclass(classifyTestsF(.ie_rd, p.value = 0.05))
attributes(.ie_ctf) <- list(dim = dim(.ie_ctf))
.ie_csv("rankdef_classifytestsf", .ie_ctf)

# eBayes t-mixture with tied moderated t but different stdev.unscaled: the
# tie order changes var.prior (R's order() is stable)
.ie_ng <- 400
.ie_se <- rep(0.5, .ie_ng)
.ie_sig <- sqrt(rchisq(.ie_ng, 6) / 6)
.ie_coef <- rnorm(.ie_ng) * .ie_se * .ie_sig
.ie_tie <- 1:3
.ie_se[.ie_tie] <- c(0.3, 1, 2)
.ie_sig[.ie_tie] <- 1
.ie_coef[.ie_tie] <- 6 * .ie_se[.ie_tie]
.ie_hb2 <- function(ord) {
  idx <- c(ord, setdiff(seq_len(.ie_ng), ord))
  new("MArrayLM", list(coefficients = cbind(.ie_coef[idx]), stdev.unscaled = cbind(.ie_se[idx]),
                       sigma = .ie_sig[idx], df.residual = rep(6, .ie_ng), Amean = rep(0, .ie_ng)))
}
.ie_eb_tie <- eBayes(.ie_hb2(.ie_tie), proportion = 0.01)
stopifnot(length(unique(round(.ie_eb_tie$t[.ie_tie, 1], 10))) == 1,
          !isTRUE(all.equal(.ie_eb_tie$var.prior, eBayes(.ie_hb2(rev(.ie_tie)), proportion = 0.01)$var.prior)))
.ie_csv("tmixture_ties_input", data.frame(coefficient = .ie_coef, stdev_unscaled = .ie_se, sigma = .ie_sig))
.ie_csv("tmixture_ties_ebayes", data.frame(var_prior = .ie_eb_tie$var.prior, t = .ie_eb_tie$t[, 1],
                                           p_value = .ie_eb_tie$p.value[, 1], lods = .ie_eb_tie$lods[, 1]))

# eBayes(trend=TRUE) where the t-mixture estimate is NA (a non-estimable
# coefficient has all-NA t), so var.prior falls back to 1/s2.prior. Under
# trend s2.prior is genewise and R recycles its first element into the NA slot.
.ie_eb_fb <- suppressWarnings(eBayes(suppressMessages(lmFit(.ie_Xd, .ie_dd)), trend = TRUE))
stopifnot(all(is.na(.ie_eb_fb$t[, 3])), isTRUE(all.equal(.ie_eb_fb$var.prior[3], 1 / .ie_eb_fb$s2.prior[1])))
.ie_csv("var_prior_fallback_var_prior", data.frame(var_prior = .ie_eb_fb$var.prior))
.ie_csv("var_prior_fallback", data.frame(t = .ie_eb_fb$t[, 2], p_value = .ie_eb_fb$p.value[, 2],
                                         lods = .ie_eb_fb$lods[, 2], s2_prior = .ie_eb_fb$s2.prior))

# avereps on an EList: E and weights averaged by genes$ID, genes de-duplicated
.ie_el4 <- new("EList", list(E = matrix(c(1, 3, 5, 7, 2, 4, 6, 8), 4, 2),
                             weights = matrix(c(1, 0.5, 1, 1, 1, 2, 1, 1), 4, 2),
                             genes = data.frame(ID = c("A", "A", "B", "B"))))
.ie_av <- avereps(.ie_el4)
.ie_csv("avereps_elist_E", .ie_av$E, ids = rownames(.ie_av$E))
.ie_csv("avereps_elist_weights", .ie_av$weights, ids = rownames(.ie_av$weights))

cat("  interface, fit-slot and edge-case fixtures complete.\n")

# =============================================================================
# selectModel: branch-forcing fixtures
# =============================================================================
cat("\nGenerating selectModel fixtures...\n")

set.seed(20261008)

.sm_n <- 60
.sm_group <- rep(0:1, each = 6)
.sm_cov <- rnorm(12)
.sm_y <- matrix(rnorm(.sm_n * 12), .sm_n, 12)
.sm_y[1:15, 7:12] <- .sm_y[1:15, 7:12] + 1.5               # group effect
.sm_y[16:30, ] <- .sm_y[16:30, ] + outer(rep(1, 15), 1.2 * .sm_cov)  # covariate effect
write.csv(.sm_y, "R_sm_y.csv", row.names = FALSE)
write.csv(data.frame(group = .sm_group, cov = .sm_cov), "R_sm_covariates.csv", row.names = FALSE)
.sm_designs <- list(null = cbind(1), group = cbind(1, .sm_group), full = cbind(1, .sm_group, .sm_cov))
.sm_designs$null <- matrix(1, 12, 1)
.sm_s2_true <- runif(.sm_n, 0.8, 1.2)
write.csv(data.frame(s2_true = .sm_s2_true), "R_sm_s2_true.csv", row.names = FALSE)

.sm_write <- function(case, sm) {
  write.csv(data.frame(unname(sm$IC), pref = as.character(sm$pref), criterion = sm$criterion),
            sprintf("R_sm_%s.csv", case), row.names = FALSE)
}
.sm_write("aic", selectModel(.sm_y, .sm_designs, criterion = "aic"))
.sm_write("bic", selectModel(.sm_y, .sm_designs, criterion = "bic"))
.sm_write("mallowscp", selectModel(.sm_y, .sm_designs, criterion = "mallowscp", s2.true = .sm_s2_true))
.sm_write("mallowscp_scalar", selectModel(.sm_y, .sm_designs, criterion = "mallowscp", s2.true = 1))
.sm_write("prior", selectModel(.sm_y, .sm_designs, criterion = "aic", df.prior = 4, s2.prior = 0.9))
.sm_write("unnamed", selectModel(.sm_y, unname(.sm_designs), criterion = "bic"))
.sm_aic <- selectModel(.sm_y, .sm_designs, criterion = "aic")
stopifnot(length(unique(as.character(.sm_aic$pref))) == 3)   # every model is preferred somewhere

.sm_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.sm_y_na <- .sm_y
.sm_y_na[1, 1] <- NA
.sm_errors <- data.frame(
  case = c("na_input", "prior_without_s2", "mallowscp_without_s2_true", "s2_true_wrong_length", "bad_criterion"),
  r_error = c(
    .sm_err(selectModel(.sm_y_na, .sm_designs)),
    .sm_err(selectModel(.sm_y, .sm_designs, df.prior = 4)),
    .sm_err(selectModel(.sm_y, .sm_designs, criterion = "mallowscp")),
    .sm_err(selectModel(.sm_y, .sm_designs, criterion = "mallowscp", s2.true = c(1, 2))),
    .sm_err(selectModel(.sm_y, .sm_designs, criterion = "xv"))
  )
)
stopifnot(!anyNA(.sm_errors$r_error))
write.csv(.sm_errors, "R_sm_errors.csv", row.names = FALSE)

cat("  selectModel fixtures complete.\n")

# =============================================================================
# camera with an estimated inter-gene correlation (inter.gene.cor = NA / NULL)
# and allow.neg.cor
# =============================================================================
cat("\nGenerating camera estimated-correlation fixtures...\n")

set.seed(20261009)

.cam_n <- 200
.cam_design <- cbind(1, rep(0:1, each = 4))
.cam_y <- matrix(rnorm(.cam_n * 8), .cam_n, 8)
# Anti-correlated pairs: genes 11-15 mirror genes 1-5, so the set's
# estimated inter-gene correlation is negative
.cam_y[11:15, ] <- -.cam_y[1:5, ] + matrix(rnorm(5 * 8, sd = 0.1), 5, 8)
# Positively correlated set sharing a latent factor, with a group effect
.cam_latent <- rnorm(8)
.cam_y[21:35, ] <- .cam_y[21:35, ] + outer(rep(1, 15), 1.5 * .cam_latent)
.cam_y[21:35, 5:8] <- .cam_y[21:35, 5:8] + 0.8
write.csv(.cam_y, "R_cam_y.csv", row.names = FALSE)
.cam_sets <- list(anti = c(1:5, 11:15), pos = 21:35, plain = 100:130, single = 150)
write.csv(do.call(rbind, lapply(names(.cam_sets), function(nm) data.frame(set = nm, index = .cam_sets[[nm]]))),
          "R_cam_sets.csv", row.names = FALSE)

.cam_run <- function(...) camera(.cam_y, .cam_sets, .cam_design, sort = FALSE, ...)
.cam_write <- function(case, tab) {
  write.csv(data.frame(set = rownames(tab), tab, check.names = FALSE), sprintf("R_cam_%s.csv", case), row.names = FALSE)
}
.cam_cases <- list(
  est_parametric = list(inter.gene.cor = NA),
  est_parametric_neg = list(inter.gene.cor = NA, allow.neg.cor = TRUE),
  est_ranks = list(inter.gene.cor = NA, use.ranks = TRUE),
  est_ranks_neg = list(inter.gene.cor = NA, use.ranks = TRUE, allow.neg.cor = TRUE),
  est_null = list(inter.gene.cor = NULL),
  est_nondirectional = list(inter.gene.cor = NA, directional = FALSE)
)
.cam_out <- lapply(.cam_cases, function(args) suppressWarnings(do.call(.cam_run, args)))
for (.nm in names(.cam_out)) .cam_write(.nm, .cam_out[[.nm]])
stopifnot(
  .cam_out$est_parametric["anti", "Correlation"] < 0,
  is.na(.cam_out$est_parametric["single", "Correlation"]),
  .cam_out$est_parametric["anti", "PValue"] != .cam_out$est_parametric_neg["anti", "PValue"],
  .cam_out$est_ranks["anti", "PValue"] != .cam_out$est_ranks_neg["anti", "PValue"],
  identical(.cam_out$est_null, .cam_out$est_parametric)
)

cat("  camera estimated-correlation fixtures complete.\n")

# =============================================================================
# gls.series with a correlation: block and duplicate-spot structures, fast
# and slow paths, estimated correlation, and input errors
# =============================================================================
cat("\nGenerating gls.series correlation fixtures...\n")

set.seed(20261010)

.gls_n <- 40
.gls_design <- cbind(1, rep(0:1, each = 4))
.gls_block <- rep(1:4, 2)
.gls_M <- matrix(rnorm(.gls_n * 8), .gls_n, 8) + outer(rep(1, .gls_n), rnorm(4)[.gls_block])
.gls_M[1:8, 5:8] <- .gls_M[1:8, 5:8] + 1.5
.gls_M_na <- .gls_M
.gls_M_na[cbind(c(2, 5, 9, 17, 30), c(1, 6, 3, 8, 2))] <- NA
.gls_w_probe <- matrix(runif(.gls_n * 8, 0.3, 2), .gls_n, 8)
.gls_w_array <- c(1, 0.5, 2, 1, 1.5, 1, 0.8, 1.2)
write.csv(.gls_M, "R_gls_cor_M.csv", row.names = FALSE)
write.csv(.gls_M_na, "R_gls_cor_M_na.csv", row.names = FALSE)
write.csv(.gls_w_probe, "R_gls_cor_probe_weights.csv", row.names = FALSE)
write.csv(data.frame(w = .gls_w_array), "R_gls_cor_array_weights.csv", row.names = FALSE)

.gls_write <- function(case, fit) {
  write.csv(data.frame(fit$coefficients, fit$stdev.unscaled, sigma = fit$sigma, df_residual = fit$df.residual),
            sprintf("R_gls_cor_%s.csv", case), row.names = FALSE)
  write.csv(fit$cov.coefficients, sprintf("R_gls_cor_%s_cov.csv", case), row.names = FALSE)
  write.csv(data.frame(correlation = fit$correlation), sprintf("R_gls_cor_%s_correlation.csv", case), row.names = FALSE)
}
.gls_write("block", gls.series(.gls_M, .gls_design, block = .gls_block, correlation = 0.4))
.gls_write("block_na", gls.series(.gls_M_na, .gls_design, block = .gls_block, correlation = 0.4))
.gls_write("block_probe_weights", gls.series(.gls_M, .gls_design, block = .gls_block, correlation = 0.4,
                                              weights = .gls_w_probe))
.gls_write("block_array_weights", gls.series(.gls_M, .gls_design, block = .gls_block, correlation = 0.4,
                                              weights = .gls_w_array))
.gls_est <- gls.series(.gls_M, .gls_design, block = .gls_block)
stopifnot(.gls_est$correlation != 0.4)
.gls_write("block_estimated", .gls_est)
.gls_write("dups", gls.series(.gls_M, .gls_design, ndups = 2, spacing = 1, correlation = 0.3))
.gls_write("dups_na", gls.series(.gls_M_na, .gls_design, ndups = 2, spacing = 1, correlation = 0.3))
.gls_nodup <- withCallingHandlers(gls.series(.gls_M, .gls_design, ndups = 1, correlation = 0.3),
                                  warning = function(w) invokeRestart("muffleWarning"))
stopifnot(.gls_nodup$correlation == 0)
.gls_write("ndups1", .gls_nodup)

.gls_err <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
.gls_errors <- data.frame(
  case = c("correlation_one", "block_length"),
  r_error = c(.gls_err(gls.series(.gls_M, .gls_design, block = .gls_block, correlation = 1)),
              .gls_err(gls.series(.gls_M, .gls_design, block = 1:3, correlation = 0.4)))
)
stopifnot(!anyNA(.gls_errors$r_error))
write.csv(.gls_errors, "R_gls_cor_errors.csv", row.names = FALSE)

cat("  gls.series correlation fixtures complete.\n")

# =============================================================================
# gls.series forwards ... to duplicateCorrelation (trim) when the correlation
# is estimated
# =============================================================================
cat("\nGenerating gls.series ... forwarding fixtures...\n")

.glt_M <- as.matrix(read.csv("R_gls_cor_M.csv"))
.glt_design <- cbind(1, rep(0:1, each = 4))
.glt_block <- rep(1:4, 2)
.glt_gls <- gls.series(.glt_M, .glt_design, block = .glt_block, trim = 0.3)
stopifnot(!isTRUE(all.equal(.glt_gls$correlation, gls.series(.glt_M, .glt_design, block = .glt_block)$correlation)))
write.csv(data.frame(.glt_gls$coefficients, .glt_gls$stdev.unscaled, sigma = .glt_gls$sigma,
                     correlation = .glt_gls$correlation), "R_gls_trim_gls.csv", row.names = FALSE)

cat("  gls.series ... forwarding fixtures complete.\n")

# =============================================================================
# contrasts.fit on fits with NA coefficients, on singular designs, and on a
# row subset fit[1:3, ]
# =============================================================================
cat("\nGenerating contrasts.fit NA-coefficient and subset fixtures...\n")

set.seed(20261011)

.cn_n <- 30
.cn_grp <- rep(1:3, each = 3)
.cn_y <- matrix(rnorm(.cn_n * 9), .cn_n, 9)
.cn_y[1:6, 7:9] <- .cn_y[1:6, 7:9] + 2
.cn_y[1, 4:6] <- NA        # group 2 missing entirely: NA coefficient
.cn_y[2, c(1, 5, 9)] <- NA # scattered missing values, all coefficients estimable
write.csv(.cn_y, "R_cn_y.csv", row.names = FALSE)

.cn_designs <- list(
  cellmeans = cbind(A = .cn_grp == 1, B = .cn_grp == 2, C = .cn_grp == 3) * 1,
  treatment = cbind(Intercept = 1, B = (.cn_grp == 2) * 1, C = (.cn_grp == 3) * 1)
)
.cn_contrasts <- list(
  cellmeans = cbind(BvsA = c(-1, 1, 0), CvsA = c(-1, 0, 1)),
  treatment = cbind(CvsB = c(0, -1, 1), C = c(0, 0, 1))
)
.cn_write <- function(case, fit) {
  eb <- eBayes(fit)
  for (nm in c("coefficients", "stdev.unscaled", "t", "p.value")) {
    write.csv(unname(as.matrix(eb[[nm]])), sprintf("R_cn_%s_%s.csv", case, gsub(".", "_", nm, fixed = TRUE)),
              row.names = FALSE)
  }
  write.csv(unname(fit$cov.coefficients), sprintf("R_cn_%s_cov_coefficients.csv", case), row.names = FALSE)
}
for (.d in names(.cn_designs)) {
  .fit <- suppressWarnings(lmFit(.cn_y, .cn_designs[[.d]]))
  stopifnot(is.na(.fit$coefficients[1, 2]))
  .cf <- contrasts.fit(.fit, .cn_contrasts[[.d]])
  # NA coefficients are set to 0 with stdev 1e30 before contrasting, then
  # contrasts whose stdev exceeds 1e20 are set back to NA: the contrast using
  # the NA coefficient is NA, the other is finite
  stopifnot(is.na(.cf$coefficients[1, 1]), is.na(.cf$stdev.unscaled[1, 1]),
            is.finite(.cf$coefficients[1, 2]))
  .cn_write(sprintf("na_%s", .d), .cf)
}

# Singular design: the third column duplicates the second
.cn_d_sing <- cbind(1, .cn_grp == 2, .cn_grp == 2, .cn_grp == 3) * 1
.cn_fit_sing <- suppressMessages(lmFit(.cn_y[3:30, ], .cn_d_sing))
.cn_write("singular", contrasts.fit(.cn_fit_sing, cbind(c(0, 0, 0, 1), c(0, 1, 0, -1))))
.cn_sing_err <- tryCatch({ contrasts.fit(.cn_fit_sing, cbind(c(0, 0, 1, 0))); NA_character_ },
                         error = function(e) conditionMessage(e))
stopifnot(!is.na(.cn_sing_err))
write.csv(data.frame(r_error = .cn_sing_err), "R_cn_singular_error.csv", row.names = FALSE)

# Row subset fit[1:3, ] then contrasts.fit, and eBayes on the three genes
.cn_fit_tr <- suppressWarnings(lmFit(.cn_y, .cn_designs$treatment))
.cn_write("subset", contrasts.fit(.cn_fit_tr[1:3, ], .cn_contrasts$treatment))
.cn_write("subset_coefficients", contrasts.fit(.cn_fit_tr[1:3, ], coefficients = c("B", "C")))

cat("  contrasts.fit NA-coefficient and subset fixtures complete.\n")

# =============================================================================
# trigammaInverse: every branch, including the extremes
# =============================================================================
cat("\nGenerating trigammaInverse fixtures...\n")

.ti_x <- c(NA, NaN, -1, 0, 1e-12, 9.99e-7, 1e-6, 1e-3, 0.1, 1, 10, 1e3, 1e6, 1e7, 1.0001e7, 1e12, Inf)
.ti_vector <- suppressWarnings(trigammaInverse(.ti_x))
.ti_scalar <- vapply(.ti_x, function(v) suppressWarnings(trigammaInverse(v)), numeric(1))
stopifnot(identical(.ti_vector[4], Inf), .ti_vector[17] == 0, is.nan(.ti_vector[3]))
.ti_warn <- tryCatch({ trigammaInverse(-1); NA_character_ }, warning = function(w) conditionMessage(w))
stopifnot(identical(.ti_warn, "NaNs produced"), length(trigammaInverse(numeric(0))) == 0)
write.csv(data.frame(x = .ti_x, vector_call = .ti_vector, scalar_call = .ti_scalar,
                     is_nan = is.nan(.ti_vector)),
          "R_trigamma_inverse_extremes.csv", row.names = FALSE)

cat("  trigammaInverse fixtures complete.\n")
