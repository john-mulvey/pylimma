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
  # a matrix of p-values is used as the vector of its values
  stopifnot(all.equal(propTrueNull(matrix(p_vec, ncol = 2), method = m), pi0))
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
# same way). goana.default always restricts an explicit universe to the
# annotated genes ("Make universe and set of all annotated genes agree"),
# which is kegga's restrict.universe = TRUE. The Term column - which real
# goana fills from GO.db - is supplied via a synthetic pathway.names table;
# pylimma's port reads it as the optional 4th column of gene.pathway.
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
                       restrict.universe = TRUE,
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

.ti_x <- c(NA, NaN, -1, 0, 1e-12, 9.99e-7, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100, 1e3, 1e6,
           1e7, 1.0001e7, 1e8, 1e12, Inf)
.ti_vector <- suppressWarnings(trigammaInverse(.ti_x))
.ti_scalar <- vapply(.ti_x, function(v) suppressWarnings(trigammaInverse(v)), numeric(1))
stopifnot(identical(.ti_vector[4], Inf), .ti_vector[length(.ti_x)] == 0, is.nan(.ti_vector[3]))
.ti_warn <- tryCatch({ trigammaInverse(-1); NA_character_ }, warning = function(w) conditionMessage(w))
stopifnot(identical(.ti_warn, "NaNs produced"), length(trigammaInverse(numeric(0))) == 0)
write.csv(data.frame(x = .ti_x, vector_call = .ti_vector, scalar_call = .ti_scalar,
                     is_nan = is.nan(.ti_vector)),
          "R_trigamma_inverse_extremes.csv", row.names = FALSE)

cat("  trigammaInverse fixtures complete.\n")

# =============================================================================
# eBayes(trend = TRUE) on fits with missing values (non-robust)
# =============================================================================
cat("\nGenerating eBayes trend + missing-value fixtures...\n")

set.seed(20261012)

.tn_n <- 300
.tn_design <- cbind(1, rep(0:1, each = 4))
.tn_amean <- runif(.tn_n, 2, 12)
.tn_y <- matrix(rnorm(.tn_n * 8, sd = rep(2 / sqrt(.tn_amean), 8)), .tn_n, 8) + .tn_amean
.tn_y[1:20, 5:8] <- .tn_y[1:20, 5:8] + 1.5
.tn_y[runif(length(.tn_y)) < 0.1] <- NA
.tn_y_df0 <- .tn_y
.tn_y_df0[300, ] <- c(5, rep(NA, 7))
.tn_y_allna <- .tn_y
.tn_y_allna[299, ] <- NA
write.csv(.tn_y, "R_tn_y.csv", row.names = FALSE)
write.csv(.tn_y_df0, "R_tn_y_df0.csv", row.names = FALSE)
write.csv(.tn_y_allna, "R_tn_y_allna.csv", row.names = FALSE)
.tn_cov <- rowMeans(.tn_y, na.rm = TRUE) + rnorm(.tn_n, sd = 0.1)
write.csv(data.frame(covariate = .tn_cov), "R_tn_covariate.csv", row.names = FALSE)

.tn_write <- function(case, eb) {
  write.csv(data.frame(t = eb$t[, 2], p_value = eb$p.value[, 2], lods = eb$lods[, 2], s2_post = eb$s2.post,
                       df_prior = rep_len(eb$df.prior, .tn_n), s2_prior = rep_len(eb$s2.prior, .tn_n),
                       df_total = eb$df.total, F = eb$F, F_p_value = eb$F.p.value),
            sprintf("R_tn_%s.csv", case), row.names = FALSE)
}
.tn_fit <- lmFit(.tn_y, .tn_design)
stopifnot(length(unique(.tn_fit$df.residual)) > 1)
.tn_eb <- eBayes(.tn_fit, trend = TRUE)
stopifnot(length(unique(.tn_eb$s2.prior)) > 1)
.tn_write("trend", .tn_eb)
.tn_fit_df0 <- lmFit(.tn_y_df0, .tn_design)
stopifnot(.tn_fit_df0$df.residual[300] == 0)
.tn_write("trend_df0", eBayes(.tn_fit_df0, trend = TRUE))
.tn_write("trend_numeric", eBayes(.tn_fit, trend = .tn_cov))
.tn_err <- tryCatch({ eBayes(lmFit(.tn_y_allna, .tn_design), trend = TRUE); NA_character_ },
                    error = function(e) conditionMessage(e))
stopifnot(!is.na(.tn_err))
write.csv(data.frame(r_error = .tn_err), "R_tn_allna_error.csv", row.names = FALSE)

cat("  eBayes trend + missing-value fixtures complete.\n")

# =============================================================================
# fitFDist spline trend predicted at excluded points: predict.ns() is linear
# beyond the boundary knots
# =============================================================================
cat("\nGenerating fitFDist out-of-range prediction fixtures...\n")

set.seed(20261013)

.ffo_write <- function(case, x, covariate) {
  ok <- is.finite(x)
  # Precondition: some excluded points lie outside the range of the fitted
  # covariate on each side, so predict.ns extrapolates
  stopifnot(any(covariate[!ok] < min(covariate[ok])), any(covariate[!ok] > max(covariate[ok])))
  fit <- fitFDist(x, df1 = 4, covariate = covariate)
  write.csv(data.frame(x = x, covariate = covariate, scale = fit$scale, df2 = fit$df2, nok = sum(ok)),
            sprintf("R_fit_f_dist_outside_%s.csv", case), row.names = FALSE)
}
.ffo_cov <- sort(runif(60, 2, 10))
# Prior-variance spread between points makes df2 finite
.ffo_x <- exp(0.3 * .ffo_cov + rnorm(60, sd = 0.7)) * rchisq(60, df = 4) / 4
.ffo_x[c(1:3, 30, 45, 58:60)] <- NA
.ffo_write("splinedf4", .ffo_x, .ffo_cov)
stopifnot(is.finite(read.csv("R_fit_f_dist_outside_splinedf4.csv")$df2[1]))
# Five usable points give splinedf = 2, a basis with no interior knots
.ffo_cov2 <- c(1, 3, 4, 5, 6, 7, 9)
.ffo_x2 <- c(NA, rchisq(5, df = 4) / 4, NA)
.ffo_write("splinedf2", .ffo_x2, .ffo_cov2)

cat("  fitFDist out-of-range prediction fixtures complete.\n")

# =============================================================================
# lmFit on a two-group design (lmfit_*.csv)
# =============================================================================
cat("\nGenerating lmfit fixtures...\n")

local({
  set.seed(42)

  # Generate test data
  n_genes <- 50
  n_samples <- 8

  # Expression matrix (genes x samples)
  expr <- matrix(rnorm(n_genes * n_samples), nrow = n_genes)
  rownames(expr) <- paste0("gene", 1:n_genes)
  colnames(expr) <- paste0("sample", 1:n_samples)

  # Two-group design
  group <- factor(rep(c("A", "B"), each = 4))
  design <- model.matrix(~ group)

  # Fit linear model
  fit <- lmFit(expr, design)

  # Save inputs
  write.csv(expr, "lmfit_expr.csv", row.names = TRUE)
  write.csv(design, "lmfit_design.csv", row.names = FALSE)

  # Save outputs
  write.csv(fit$coefficients, "lmfit_coefficients.csv", row.names = TRUE)
  write.csv(fit$stdev.unscaled, "lmfit_stdev_unscaled.csv", row.names = TRUE)
  write.csv(data.frame(sigma = fit$sigma, df_residual = fit$df.residual, Amean = fit$Amean),
            "lmfit_stats.csv", row.names = TRUE)
  write.csv(fit$cov.coefficients, "lmfit_cov_coef.csv", row.names = TRUE)
})

cat("  lmfit fixtures complete.\n")

# =============================================================================
# contrasts.fit on a three-group cell-means design (contrasts_*.csv)
# =============================================================================
cat("\nGenerating contrasts fixtures...\n")

local({
  set.seed(42)

  # Generate test data
  n_genes <- 30
  n_samples <- 12

  # Expression matrix (genes x samples)
  expr <- matrix(rnorm(n_genes * n_samples), nrow = n_genes)
  rownames(expr) <- paste0("gene", 1:n_genes)

  # Three-group design
  group <- factor(rep(c("A", "B", "C"), each = 4))
  design <- model.matrix(~ 0 + group)
  colnames(design) <- levels(group)

  # Fit linear model
  fit <- lmFit(expr, design)

  # Create contrasts
  contrast_matrix <- makeContrasts(
    BvsA = B - A,
    CvsA = C - A,
    CvsB = C - B,
    levels = design
  )

  # Apply contrasts
  fit2 <- contrasts.fit(fit, contrast_matrix)

  # Save inputs
  write.csv(expr, "contrasts_expr.csv", row.names = TRUE)
  write.csv(design, "contrasts_design.csv", row.names = FALSE)
  write.csv(contrast_matrix, "contrast_matrix.csv", row.names = TRUE)

  # Save original fit
  write.csv(fit$coefficients, "contrasts_fit_coef.csv", row.names = TRUE)
  write.csv(fit$stdev.unscaled, "contrasts_fit_stdev.csv", row.names = TRUE)

  # Save contrast fit
  write.csv(fit2$coefficients, "contrasts_fit2_coef.csv", row.names = TRUE)
  write.csv(fit2$stdev.unscaled, "contrasts_fit2_stdev.csv", row.names = TRUE)
  write.csv(fit2$cov.coefficients, "contrasts_fit2_cov.csv", row.names = TRUE)
})

cat("  contrasts fixtures complete.\n")

# =============================================================================
# eBayes on a two-group design (ebayes_*.csv)
# =============================================================================
cat("\nGenerating ebayes fixtures...\n")

local({
  set.seed(42)

  # Generate test data
  n_genes <- 50
  n_samples <- 8

  # Expression matrix with some true differences
  expr <- matrix(rnorm(n_genes * n_samples), nrow = n_genes)
  rownames(expr) <- paste0("gene", 1:n_genes)

  # Add true effects to first 10 genes
  expr[1:10, 5:8] <- expr[1:10, 5:8] + 2

  # Two-group design
  group <- factor(rep(c("A", "B"), each = 4))
  design <- model.matrix(~ group)

  # Fit and run eBayes
  fit <- lmFit(expr, design)
  fit <- eBayes(fit)

  # Save outputs
  write.csv(
    data.frame(
      t = fit$t[, 2],
      p_value = fit$p.value[, 2],
      lods = fit$lods[, 2],
      s2_post = fit$s2.post,
      df_total = fit$df.total
    ),
    "ebayes_stats.csv",
    row.names = TRUE
  )

  write.csv(
    data.frame(
      s2_prior = fit$s2.prior,
      df_prior = fit$df.prior,
      F_stat = fit$F,
      F_p_value = fit$F.p.value
    ),
    "ebayes_global.csv",
    row.names = FALSE
  )

  # Also save the fit coefficients for reference
  write.csv(fit$coefficients, "ebayes_coef.csv", row.names = TRUE)
  write.csv(expr, "ebayes_expr.csv", row.names = TRUE)
  write.csv(design, "ebayes_design.csv", row.names = FALSE)
})

cat("  ebayes fixtures complete.\n")

# =============================================================================
# topTable on the eBayes data (toptable_*.csv)
# =============================================================================
cat("\nGenerating toptable fixtures...\n")

local({
  set.seed(42)

  # Generate test data
  n_genes <- 50
  n_samples <- 8

  # Expression with true effects
  expr <- matrix(rnorm(n_genes * n_samples), nrow = n_genes)
  rownames(expr) <- paste0("gene", 1:n_genes)
  expr[1:10, 5:8] <- expr[1:10, 5:8] + 2  # True DE genes

  # Two-group design
  group <- factor(rep(c("A", "B"), each = 4))
  design <- model.matrix(~ group)

  # Fit
  fit <- lmFit(expr, design)
  fit <- eBayes(fit)

  # Get top table for coefficient 2 (groupB)
  tt <- topTable(fit, coef = 2, number = 20, sort.by = "B")

  # Rename columns to match our snake_case convention
  names(tt) <- c("log_fc", "ave_expr", "t", "p_value", "adj_p_value", "b")
  tt$gene <- rownames(tt)
  rownames(tt) <- NULL
  tt <- tt[, c("gene", "log_fc", "ave_expr", "t", "p_value", "adj_p_value", "b")]

  write.csv(tt, "toptable_output.csv", row.names = FALSE)

  # Also get all genes for verification
  tt_all <- topTable(fit, coef = 2, number = Inf, sort.by = "none")
  names(tt_all) <- c("log_fc", "ave_expr", "t", "p_value", "adj_p_value", "b")
  tt_all$gene <- rownames(tt_all)
  rownames(tt_all) <- NULL
  tt_all <- tt_all[, c("gene", "log_fc", "ave_expr", "t", "p_value", "adj_p_value", "b")]
  write.csv(tt_all, "toptable_all.csv", row.names = FALSE)
})

cat("  toptable fixtures complete.\n")

# =============================================================================
# fitFDist and squeezeVar on F-distributed variances
# =============================================================================
cat("\nGenerating squeeze_var fixtures...\n")

local({
  set.seed(42)

  # Generate test data: sample variances from chi-squared
  n_genes <- 100
  df_residual <- 5
  true_s0 <- 0.5
  true_d0 <- 4

  # Generate sample variances: s^2 ~ s0^2 * F(df, d0)
  # which is equivalent to s^2 ~ s0^2 * (chi2(df)/df) / (chi2(d0)/d0)
  sample_var <- true_s0 * rf(n_genes, df1 = df_residual, df2 = true_d0)

  # Run fitFDist
  fit <- fitFDist(sample_var, df1 = df_residual)

  # Run squeezeVar
  sv <- squeezeVar(sample_var, df = df_residual)

  # Save test inputs and outputs
  write.csv(
    data.frame(sample_var = sample_var),
    "squeeze_var_input.csv",
    row.names = FALSE
  )

  write.csv(
    data.frame(
      fit_scale = fit$scale,
      fit_df2 = fit$df2
    ),
    "fit_f_dist_output.csv",
    row.names = FALSE
  )

  write.csv(
    data.frame(
      var_post = sv$var.post,
      var_prior = sv$var.prior,
      df_prior = sv$df.prior
    ),
    "squeeze_var_output.csv",
    row.names = FALSE
  )
})

cat("  squeeze_var fixtures complete.\n")

# =============================================================================
# EList and MArrayLM [i, j] subsetting (R_elist_*, R_marraylm_*)
# =============================================================================
cat("\nGenerating classes fixtures...\n")

local({
  set.seed(42)

  n_genes <- 30
  n_samples <- 8

  E <- matrix(rnorm(n_genes * n_samples), nrow = n_genes)
  rownames(E) <- paste0("gene", 1:n_genes)
  colnames(E) <- paste0("sample", 1:n_samples)

  weights <- matrix(runif(n_genes * n_samples, 0.5, 1.5), nrow = n_genes)
  rownames(weights) <- rownames(E)
  colnames(weights) <- colnames(E)

  genes <- data.frame(
    ID = rownames(E),
    chromosome = sample(c("chr1", "chr2", "chr3"), n_genes, replace = TRUE),
    row.names = rownames(E),
    stringsAsFactors = FALSE
  )

  group <- factor(rep(c("A", "B"), each = 4))
  targets <- data.frame(
    SampleID = colnames(E),
    Group = group,
    row.names = colnames(E),
    stringsAsFactors = FALSE
  )

  design <- model.matrix(~ group)
  rownames(design) <- colnames(E)
  colnames(design) <- c("Intercept", "groupB")

  # -----------------------------------------------------------------------------
  # EList subsetting
  # -----------------------------------------------------------------------------

  el <- new("EList", list(
    E = E,
    weights = weights,
    genes = genes,
    targets = targets,
    design = design
  ))

  write_elist <- function(obj, tag) {
    write.csv(obj$E, sprintf("R_elist_%s_E.csv", tag), row.names = TRUE)
    write.csv(obj$weights, sprintf("R_elist_%s_weights.csv", tag), row.names = TRUE)
    write.csv(obj$genes, sprintf("R_elist_%s_genes.csv", tag), row.names = TRUE)
    write.csv(obj$targets, sprintf("R_elist_%s_targets.csv", tag), row.names = TRUE)
    write.csv(obj$design, sprintf("R_elist_%s_design.csv", tag), row.names = TRUE)
  }

  # Full object
  write_elist(el, "full")

  # Row subset (first 10 genes)
  write_elist(el[1:10, ], "rows")

  # Column subset (first 4 samples)
  write_elist(el[, 1:4], "cols")

  # Row + column subset
  write_elist(el[1:10, 1:4], "both")

  # String-indexed rows
  write_elist(el[c("gene3", "gene7", "gene15"), ], "rowstr")

  # Boolean row mask
  row_mask <- rep(FALSE, n_genes); row_mask[c(2, 4, 6, 8, 10)] <- TRUE
  write_elist(el[row_mask, ], "rowbool")

  # -----------------------------------------------------------------------------
  # MArrayLM subsetting
  # -----------------------------------------------------------------------------

  fit <- lmFit(el, design)
  fit <- eBayes(fit)

  write_marraylm <- function(obj, tag) {
    write.csv(obj$coefficients, sprintf("R_marraylm_%s_coefficients.csv", tag), row.names = TRUE)
    write.csv(obj$stdev.unscaled, sprintf("R_marraylm_%s_stdev_unscaled.csv", tag), row.names = TRUE)
    write.csv(obj$t, sprintf("R_marraylm_%s_t.csv", tag), row.names = TRUE)
    write.csv(obj$p.value, sprintf("R_marraylm_%s_p_value.csv", tag), row.names = TRUE)
    write.csv(obj$lods, sprintf("R_marraylm_%s_lods.csv", tag), row.names = TRUE)
    write.csv(data.frame(
      Amean = obj$Amean,
      sigma = obj$sigma,
      df_residual = obj$df.residual,
      df_total = obj$df.total,
      s2_post = obj$s2.post
    ), sprintf("R_marraylm_%s_i_slots.csv", tag), row.names = TRUE)
    write.csv(obj$genes, sprintf("R_marraylm_%s_genes.csv", tag), row.names = TRUE)
  }

  write_marraylm(fit, "full")
  write_marraylm(fit[1:10, ], "rows")
  write_marraylm(fit[, 2, drop = FALSE], "cols")
  write_marraylm(fit[1:10, 2, drop = FALSE], "both")
  write_marraylm(fit[c("gene3", "gene7", "gene15"), ], "rowstr")
})

cat("  classes fixtures complete.\n")

# =============================================================================
# fitFDist spline trend with interior knots on a boundary knot: ns() shoves
# them inwards, or fails when every interior knot is on one boundary
# =============================================================================
cat("\nGenerating fitFDist knot-shoving fixtures...\n")

set.seed(20261014)

.fks_run <- function(x, covariate) {
  shoved <- FALSE
  fit <- withCallingHandlers(fitFDist(x, df1 = 4, covariate = covariate), warning = function(w) {
    if (grepl("shoving 'interior' knots", conditionMessage(w))) {
      shoved <<- TRUE
      invokeRestart("muffleWarning")
    }
  })
  # Precondition: ns() moved an interior knot
  stopifnot(shoved)
  fit
}
.fks_write <- function(case, x, covariate) {
  fit <- .fks_run(x, covariate)
  write.csv(data.frame(x = x, covariate = covariate, scale = fit$scale, df2 = fit$df2),
            sprintf("R_fit_f_dist_shoved_%s.csv", case), row.names = FALSE)
}
.fks_x <- function(covariate) exp(0.1 * covariate + rnorm(length(covariate), sd = 0.5)) * rchisq(length(covariate), 4) / 4
.fks_cov <- list(
  left = c(rep(0, 20), runif(20, 1, 10)),
  right = c(runif(20, 1, 10), rep(10, 20)),
  both = c(rep(0, 15), runif(10, 1, 9), rep(10, 15))
)
for (.case in names(.fks_cov)) .fks_write(.case, .fks_x(.fks_cov[[.case]]), .fks_cov[[.case]])
# Excluded points are predicted with the shoved knots
.fks_x_na <- .fks_x(.fks_cov$left)
.fks_x_na[c(3, 25, 33)] <- NA
.fks_write("left_na", .fks_x_na, .fks_cov$left)

# More than two thirds tied at the minimum: every interior knot is on the
# left boundary, ns() stops and fitFDist reports a problem with the covariate
.fks_cov_err <- c(rep(0, 30), runif(10, 1, 10))
.fks_x_err <- .fks_x(.fks_cov_err)
.fks_err <- tryCatch({ fitFDist(.fks_x_err, df1 = 4, covariate = .fks_cov_err); NA_character_ },
                     error = function(e) conditionMessage(e))
stopifnot(identical(.fks_err, "Problem with covariate"))
write.csv(data.frame(x = .fks_x_err, covariate = .fks_cov_err), "R_fit_f_dist_shoved_error_input.csv",
          row.names = FALSE)

cat("  fitFDist knot-shoving fixtures complete.\n")

# =============================================================================
# fitFDist spline trend on a numerically rank-deficient basis: lm.fit drops
# aliased columns (tol 1e-7), evar uses the effects beyond the rank, and
# predictions at excluded points are NA
# =============================================================================
cat("\nGenerating fitFDist rank-deficient trend fixtures...\n")

set.seed(20261015)

.frd_write <- function(case, x, covariate) {
  ok <- is.finite(x)
  # Precondition: the ns() basis for the usable points is rank deficient
  design <- splines::ns(covariate[ok], df = 4, intercept = TRUE)
  stopifnot(lm.fit(design, log(x[ok]))$rank < ncol(design))
  fit <- fitFDist(x, df1 = 4, covariate = covariate)
  write.csv(data.frame(x = x, covariate = covariate, scale = fit$scale, df2 = fit$df2),
            sprintf("R_fit_f_dist_rank_deficient_%s.csv", case), row.names = FALSE)
  fit
}
.frd_cov <- rep(c(1, 1 + 1e-9, 2, 3), each = 10)
.frd_x <- exp(rnorm(40, sd = 0.5)) * rchisq(40, 4) / 4
.frd_write("low", .frd_x, .frd_cov)
.frd_cov_mid <- rep(c(1, 2, 2 + 1e-9, 3), each = 10)
.frd_write("mid", exp(rnorm(40, sd = 0.5)) * rchisq(40, 4) / 4, .frd_cov_mid)
.frd_x_na <- .frd_x
.frd_x_na[c(5, 35)] <- NA
.frd_fit_na <- .frd_write("na", .frd_x_na, .frd_cov)
stopifnot(all(is.na(.frd_fit_na$scale[c(5, 35)])), all(is.finite(.frd_fit_na$scale[-c(5, 35)])))

cat("  fitFDist rank-deficient trend fixtures complete.\n")

# =============================================================================
# contrasts.fit branches: coefficients= returns fit[, coefficients] (keeping
# test statistics, regenerating F, subsetting cov.coefficients / contrasts /
# var.prior), the stored contrasts matrix, no cov.coefficients, and a
# near-orthogonal design
# =============================================================================
cat("\nGenerating contrasts.fit branch fixtures...\n")

set.seed(20261016)

.cfb_grp <- rep(1:3, each = 4)
.cfb_y <- matrix(rnorm(40 * 12), 40, 12)
.cfb_y[1:8, .cfb_grp == 3] <- .cfb_y[1:8, .cfb_grp == 3] + 2
write.csv(.cfb_y, "R_cfb_y.csv", row.names = FALSE)
.cfb_design <- cbind(Intercept = 1, B = (.cfb_grp == 2) * 1, C = (.cfb_grp == 3) * 1)
.cfb_eb <- eBayes(lmFit(.cfb_y, .cfb_design), proportion = 0.1)
stopifnot(length(.cfb_eb$var.prior) == 3)

.cfb_write <- function(case, fit) {
  for (nm in c("coefficients", "stdev.unscaled", "t", "p.value", "lods", "cov.coefficients", "contrasts")) {
    if (!is.null(fit[[nm]])) {
      write.csv(unname(as.matrix(fit[[nm]])), sprintf("R_cfb_%s_%s.csv", case, gsub(".", "_", nm, fixed = TRUE)),
                row.names = FALSE)
    }
  }
  vec <- data.frame(F = if (is.null(fit$F)) NA else fit$F, F_p_value = if (is.null(fit$F.p.value)) NA else fit$F.p.value)
  write.csv(vec, sprintf("R_cfb_%s_f.csv", case), row.names = FALSE)
  if (!is.null(fit$var.prior)) write.csv(data.frame(var_prior = fit$var.prior), sprintf("R_cfb_%s_var_prior.csv", case),
                                         row.names = FALSE)
}

# coefficients= on an eBayes fit keeps t / p.value / lods and regenerates F
.cfb_sub <- contrasts.fit(.cfb_eb, coefficients = c(2, 3))
stopifnot(!is.null(.cfb_sub$t), !is.null(.cfb_sub$F), length(.cfb_sub$var.prior) == 2,
          !isTRUE(all.equal(.cfb_sub$F, .cfb_eb$F)))
.cfb_write("coef_subset", .cfb_sub)
write.csv(topTable(.cfb_sub, coef = 1, number = Inf, sort.by = "none"), "R_cfb_coef_subset_toptable.csv",
          row.names = FALSE)
# a single coefficient, in reversed order
.cfb_write("coef_reversed", contrasts.fit(.cfb_eb, coefficients = c(3, 2)))

# coefficients= on a fit that already has contrasts subsets that matrix; the
# stored contrasts matrix is the one supplied, all-zero Intercept row included
.cfb_cm <- cbind(CvsB = c(0, -1, 1), C = c(0, 0, 1))
.cfb_cf <- eBayes(contrasts.fit(lmFit(.cfb_y, .cfb_design), .cfb_cm))
stopifnot(identical(unname(.cfb_cf$contrasts), unname(.cfb_cm)))
.cfb_write("contrasted", .cfb_cf)
.cfb_write("contrasted_coef2", contrasts.fit(.cfb_cf, coefficients = 2))

# Rank-deficient fit: cov.coefficients is subset through pivot, in pivot order
.cfb_d_sing <- cbind(.cfb_design[, 1:2], dup = .cfb_design[, 2], C = .cfb_design[, 3])
.cfb_eb_sing <- eBayes(suppressMessages(lmFit(.cfb_y, .cfb_d_sing)))
stopifnot(ncol(.cfb_eb_sing$cov.coefficients) == 3)
.cfb_write("singular_coef41", contrasts.fit(.cfb_eb_sing, coefficients = c(4, 1)))
.cfb_sing_err <- tryCatch({ contrasts.fit(.cfb_eb_sing, coefficients = 3); NA_character_ },
                          error = function(e) conditionMessage(e))
stopifnot(identical(.cfb_sing_err, "Subsetting to non-estimable coefficients is not allowed."))

# No cov.coefficients: warn and assume orthogonal coefficients
.cfb_nocov <- lmFit(.cfb_y, .cfb_design)
.cfb_nocov$cov.coefficients <- NULL
.cfb_nocov_warn <- NULL
.cfb_nocov_cf <- withCallingHandlers(contrasts.fit(.cfb_nocov, .cfb_cm), warning = function(w) {
  .cfb_nocov_warn <<- conditionMessage(w)
  invokeRestart("muffleWarning")
})
stopifnot(identical(.cfb_nocov_warn, "cov.coefficients not found in fit - assuming coefficients are orthogonal"))
.cfb_write("nocov", .cfb_nocov_cf)

# Near-orthogonal design: one coefficient correlation in [1e-14, 1e-12), where
# the deciding test all(abs(lower.tri) < 1e-14) is FALSE
.cfb_x1 <- rep(c(-1, 1), 6)
.cfb_x2 <- rep(c(-1, -1, 1, 1), 3) + 3e-13 * .cfb_x1
.cfb_d_orth <- cbind(1, .cfb_x1, .cfb_x2)
.cfb_fit_orth <- lmFit(.cfb_y, .cfb_d_orth)
.cfb_cor <- abs(cov2cor(.cfb_fit_orth$cov.coefficients)[lower.tri(diag(3))])
stopifnot(max(.cfb_cor) >= 1e-14, max(.cfb_cor) < 1e-12)
write.csv(.cfb_d_orth, "R_cfb_orth_design.csv", row.names = FALSE)
.cfb_write("orth", contrasts.fit(.cfb_fit_orth, cbind(c(0, 1, 1), c(0, 1, -1))))

cat("  contrasts.fit branch fixtures complete.\n")

# =============================================================================
# lmFit correlation: missing stops, an explicit NULL is estimated by
# gls.series via duplicateCorrelation; voom passes correlation = NULL
# =============================================================================
cat("\nGenerating lmFit correlation = NULL fixtures...\n")

set.seed(20261017)

.lcn_counts <- matrix(rpois(200 * 8, 50), 200, 8)
.lcn_counts[1:20, 5:8] <- .lcn_counts[1:20, 5:8] * 2
.lcn_design <- cbind(1, rep(0:1, each = 4))
.lcn_block <- rep(1:4, 2)
.lcn_y <- log2(.lcn_counts + 1)
write.csv(.lcn_counts, "R_lcn_counts.csv", row.names = FALSE)
.lcn_err <- tryCatch({ lmFit(.lcn_y, .lcn_design, block = .lcn_block); NA_character_ },
                     error = function(e) conditionMessage(e))
stopifnot(identical(.lcn_err, "the correlation must be set, see duplicateCorrelation"))
.lcn_write <- function(case, fit) {
  stopifnot(fit$correlation != 0)
  write.csv(data.frame(coefficients = fit$coefficients, stdev_unscaled = fit$stdev.unscaled,
                       sigma = fit$sigma, correlation = fit$correlation),
            sprintf("R_lcn_%s.csv", case), row.names = FALSE)
}
.lcn_write("block", lmFit(.lcn_y, .lcn_design, block = .lcn_block, correlation = NULL))
.lcn_write("ndups", lmFit(.lcn_y, .lcn_design, ndups = 2, correlation = NULL))
.lcn_v <- voom(.lcn_counts, .lcn_design, block = .lcn_block)
write.csv(.lcn_v$weights, "R_lcn_voom_weights.csv", row.names = FALSE)

cat("  lmFit correlation = NULL fixtures complete.\n")

# =============================================================================
# eBayes(proportion = 1): log(proportion / (1 - proportion)) is Inf, so every
# lods is Inf. predFCm reaches it when propTrueNull(method = "hist") is 0.
# =============================================================================
cat("\nGenerating eBayes proportion = 1 fixtures...\n")

set.seed(20261020)

.ep1_y <- matrix(rnorm(6 * 3, mean = rep(c(5, 10, 8, 11, 6, 7.5), 3), sd = 0.05), 6, 3)
write.csv(.ep1_y, "R_ep1_y.csv", row.names = FALSE)
.ep1_fit <- lmFit(.ep1_y, matrix(1, 3, 1))
.ep1_lods <- eBayes(.ep1_fit, proportion = 1)$lods
stopifnot(all(.ep1_lods == Inf))
.ep1_eb <- eBayes(.ep1_fit, proportion = 0.01)
stopifnot(propTrueNull(.ep1_eb$p.value[, 1], method = "hist") == 0)
write.csv(data.frame(pred_fc = predFCm(.ep1_eb, coef = 1, prop.true.null.method = "hist")),
          "R_ep1_predfcm_hist.csv", row.names = FALSE)

cat("  eBayes proportion = 1 fixtures complete.\n")

# =============================================================================
# predFCm(all.de = FALSE) when propTrueNull is 0: p = 1, A = p / (1 - p) is
# Inf, lods is Inf and probDE is set to 1 (uses .ep1_eb from the eBayes
# proportion = 1 section)
# =============================================================================
cat("\nGenerating predFCm all.de = FALSE fixtures...\n")

write.csv(data.frame(
  indep = predFCm(.ep1_eb, coef = 1, all.de = FALSE, prop.true.null.method = "hist"),
  dep = predFCm(.ep1_eb, coef = 1, var.indep.of.fc = FALSE, all.de = FALSE, prop.true.null.method = "hist")
), "R_ep1_predfcm_notallde.csv", row.names = FALSE)

cat("  predFCm all.de = FALSE fixtures complete.\n")

# =============================================================================
# duplicateCorrelation when a gene's REML fit hits the iteration cap:
# statmod::glmgam.fit breaks on iter > maxit, so it runs maxit + 1 iterations
# =============================================================================
cat("\nGenerating duplicateCorrelation iteration-cap fixtures...\n")

set.seed(20261021)

.dcc_y <- matrix(rnorm(200 * 12), 200, 12)
write.csv(.dcc_y, "R_dcc_y.csv", row.names = FALSE)
.dcc_design <- matrix(0, 12, 3)
.dcc_design[cbind(1:12, rep(1:3, each = 4))] <- 1
.dcc_block <- rep(1:4, each = 3)
.dcc_Z <- model.matrix(~ 0 + factor(.dcc_block))
.dcc_atanh <- function(g, maxit) {
  v <- statmod::mixedModel2Fit(.dcc_y[g, ], .dcc_design, .dcc_Z, only.varcomp = TRUE, maxit = maxit)$varcomp
  atanh(v[2] / sum(v))
}
# at least one gene's estimate depends on the cap, so the fixture exercises it
stopifnot(any(sapply(1:200, function(g) abs(.dcc_atanh(g, 20) - .dcc_atanh(g, 21)) > 1e-8)))
.dcc <- duplicateCorrelation(.dcc_y, .dcc_design, block = .dcc_block)
write.csv(data.frame(atanh_correlations = .dcc$atanh.correlations,
                     consensus_correlation = .dcc$consensus.correlation),
          "R_dcc_dupcor.csv", row.names = FALSE)

cat("  duplicateCorrelation iteration-cap fixtures complete.\n")

# =============================================================================
# fitGammaIntercept iteration control: the loop tests convergence, then
# iter > maxit (so maxit + 1 updates run), warning when the limit is hit;
# inputs with no root run to the limit and return -Inf
# =============================================================================
cat("\nGenerating fitGammaIntercept fixtures...\n")

.fgi_y <- c(0.5, 1.2, 3.0, 0.1, 2.2, 4.0)
.fgi_offset <- c(0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
.fgi_run <- function(case, y, offset, maxit) {
  warned <- FALSE
  x <- withCallingHandlers(fitGammaIntercept(y, offset = offset, maxit = maxit), warning = function(w) {
    stopifnot(identical(conditionMessage(w), "iteration limit maxit exceeded"))
    warned <<- TRUE
    invokeRestart("muffleWarning")
  })
  data.frame(case = case, maxit = maxit, x = x, warned = warned)
}
.fgi <- rbind(
  .fgi_run("maxit", .fgi_y, .fgi_offset, 0),
  .fgi_run("maxit", .fgi_y, .fgi_offset, 1),
  .fgi_run("maxit", .fgi_y, .fgi_offset, 2),
  .fgi_run("maxit", .fgi_y, .fgi_offset, 1000),
  .fgi_run("zero_y", c(0, 0, 0), c(1, 2, 3), 1000),
  .fgi_run("no_root", c(0, 0, 0, 5), c(1, 2, 3, 4), 1000)
)
stopifnot(.fgi$warned == c(TRUE, TRUE, TRUE, FALSE, TRUE, TRUE), .fgi$x[5:6] == -Inf)
write.csv(.fgi, "R_fgi.csv", row.names = FALSE)
# Empty y with a scalar offset: the constant-offset shortcut gives
# mean(numeric(0)) - offset, NaN with no warning
stopifnot(is.nan(withCallingHandlers(fitGammaIntercept(numeric(0)), warning = function(w) stop(w))))
# A 0 / 0 Newton step (zero offset where y is zero) stops R obscurely;
# pylimma raises a clear error (known_differences.rst)
stopifnot(identical(tryCatch(fitGammaIntercept(c(0, 1), offset = c(0, 1)), error = conditionMessage),
                    "missing value where TRUE/FALSE needed"))

cat("  fitGammaIntercept fixtures complete.\n")

# =============================================================================
# decideTests(method = "hierarchical" / "nestedF") without F-statistics.
# eBayes computes F only for a full-rank design. With F.p.value NULL, R's
# p.adjust(NULL) is numeric(0), so nothing is selected and the result is all
# zero by accident; pylimma raises an error instead (known_differences.rst).
# No CSV: this records the literal R behaviour.
# =============================================================================
cat("\nChecking decideTests without F-statistics...\n")

set.seed(20261022)

.dtf_y <- matrix(rnorm(30 * 6), 30, 6)
.dtf_y[1:5, 4:6] <- .dtf_y[1:5, 4:6] + 4
.dtf_design <- cbind(1, rep(0:1, each = 3), rep(0:1, each = 3))
.dtf_eb <- eBayes(suppressMessages(lmFit(.dtf_y, .dtf_design)))
stopifnot(is.null(.dtf_eb$F.p.value))
for (.dtf_method in c("hierarchical", "nestedF")) {
  stopifnot(all(decideTests(.dtf_eb, method = .dtf_method) == 0, na.rm = TRUE))
}

cat("  decideTests without F-statistics checked.\n")

# =============================================================================
# topTable(number=): capped at the number of rows, number < 1 returns an
# empty data.frame, and a non-integer number is truncated by ord[1:number]
# (both .topTableT and .topTableF)
# =============================================================================
cat("\nGenerating topTable number fixtures...\n")

set.seed(20261023)

.ttn_y <- matrix(rnorm(20 * 6), 20, 6)
write.csv(.ttn_y, "R_ttn_y.csv", row.names = FALSE)
.ttn_eb <- eBayes(lmFit(.ttn_y, cbind(1, rep(0:1, each = 3), c(0, 1, 0, 1, 0, 1))))
.ttn_numbers <- c(-5, -1, 0, 0.5, 1.5, 2.5, 3, 25, Inf)
write.csv(data.frame(
  number = .ttn_numbers,
  nrow_t = sapply(.ttn_numbers, function(n) nrow(topTable(.ttn_eb, coef = 2, number = n))),
  nrow_f = sapply(.ttn_numbers, function(n) nrow(topTable(.ttn_eb, coef = 2:3, number = n)))
), "R_ttn_nrow.csv", row.names = FALSE)

cat("  topTable number fixtures complete.\n")

# =============================================================================
# duplicateCorrelation(trim=): the consensus is mean(arho, trim = trim), which
# is the median for trim >= 0.5 and the plain mean for trim <= 0
# (uses .dcc_y from the iteration-cap section)
# =============================================================================
cat("\nGenerating duplicateCorrelation trim fixtures...\n")

.dct_trims <- c(-1, 0, 0.15, 0.4, 0.5, 0.6)
write.csv(data.frame(
  trim = .dct_trims,
  consensus_correlation = sapply(.dct_trims, function(tr)
    duplicateCorrelation(.dcc_y, .dcc_design, block = .dcc_block, trim = tr)$consensus.correlation)
), "R_dct_consensus.csv", row.names = FALSE)

cat("  duplicateCorrelation trim fixtures complete.\n")

# =============================================================================
# chooseLowessSpan: pmin(min.span + (1 - min.span) * (small.n / n)^power, 1)
# in double arithmetic, so n = 0 gives Inf, clamped to 1
# =============================================================================
cat("\nGenerating chooseLowessSpan fixtures...\n")

.cls_n <- c(0, 1, 10, 50, 100, 1000, 1e5)
write.csv(data.frame(n = .cls_n, span = chooseLowessSpan(.cls_n)), "R_cls_span.csv", row.names = FALSE)

cat("  chooseLowessSpan fixtures complete.\n")

# =============================================================================
# fitmixture with 2 arrays: narrays / (narrays - 2) is Inf in R's double
# arithmetic, so stdev is Inf or NaN (0 * Inf) while A and M are estimated
# =============================================================================
cat("\nGenerating fitmixture 2-array fixtures...\n")

.fm2_y <- matrix(c(5, 6, 4, 5.5, 5.2, 4.8), 3, 2)
.fm2 <- fitmixture(.fm2_y, mixprop = c(1, 0.5))
stopifnot(all(is.infinite(.fm2$stdev) | is.nan(.fm2$stdev)))
write.csv(data.frame(A = .fm2$A, M = .fm2$M, stdev = .fm2$stdev), "R_fm2.csv", row.names = FALSE)

cat("  fitmixture 2-array fixtures complete.\n")

# =============================================================================
# classifyTestsF with one coefficient: sign(t) * (p < p.value), so a gene
# with NA t gives NA
# =============================================================================
cat("\nGenerating classifyTestsF single-coefficient NA fixtures...\n")

set.seed(20261025)

.ctf_y <- matrix(rnorm(8 * 4, mean = 3), 8, 4)
.ctf_y[3, ] <- NA
write.csv(.ctf_y, "R_ctf_y.csv", row.names = FALSE)
.ctf_res <- classifyTestsF(eBayes(lmFit(.ctf_y, matrix(1, 4, 1))))
stopifnot(is.na(.ctf_res[3, 1]))
write.csv(data.frame(result = as.vector(.ctf_res)), "R_ctf_result.csv", row.names = FALSE)

cat("  classifyTestsF single-coefficient NA fixtures complete.\n")

# =============================================================================
# cumOverlap: upper-tail phyper(lower.tail = FALSE) keeps p-values below
# 1e-16, which decide n.min and p.min for concordant lists
# =============================================================================
cat("\nGenerating cumOverlap fixtures...\n")

set.seed(20261026)

.cov_ids <- paste0("g", 1:100)
.cov_cases <- list(identical = .cov_ids, shuffled = c(sample(.cov_ids[1:30]), sample(.cov_ids[31:100])))
for (.cov_case in names(.cov_cases)) {
  .cov <- cumOverlap(.cov_ids, .cov_cases[[.cov_case]])
  write.csv(data.frame(n_overlap = .cov$n.overlap, p_value = .cov$p.value, adj_p_value = .cov$adj.p.value),
            sprintf("R_cov_%s.csv", .cov_case), row.names = FALSE)
  write.csv(data.frame(n_min = .cov$n.min, p_min = .cov$p.min),
            sprintf("R_cov_%s_min.csv", .cov_case), row.names = FALSE)
}
write.csv(data.frame(id = .cov_cases$shuffled), "R_cov_shuffled_ids.csv", row.names = FALSE)
stopifnot(min(cumOverlap(.cov_ids, .cov_ids)$p.value) < 1e-20)

cat("  cumOverlap fixtures complete.\n")

# =============================================================================
# Sparse input: lmFit and voom coerce a sparse Matrix with as.matrix(), so the
# results equal those for the dense matrix (no CSV: pylimma tests compare its
# sparse and dense results directly)
# =============================================================================
cat("\nChecking sparse Matrix input...\n")

set.seed(20261027)

.spm_counts <- matrix(rpois(30 * 6, 5) * rbinom(30 * 6, 1, 0.6), 30, 6)
.spm_design <- cbind(1, rep(0:1, each = 3))
stopifnot(isTRUE(all.equal(lmFit(Matrix::Matrix(log2(.spm_counts + 1), sparse = TRUE), .spm_design)$coefficients,
                           lmFit(log2(.spm_counts + 1), .spm_design)$coefficients)))
stopifnot(isTRUE(all.equal(voom(Matrix::Matrix(.spm_counts, sparse = TRUE), .spm_design)$weights,
                           voom(.spm_counts, .spm_design)$weights)))

cat("  sparse Matrix input checked.\n")

# =============================================================================
# uniquegenelist on a data.frame keeps the row names of the rows it selects
# =============================================================================
cat("\nGenerating uniquegenelist row-name fixtures...\n")

.ugl_genes <- data.frame(sym = paste0("s", 1:8), row.names = paste0("p", 1:8))
for (.ugl_spacing in 1:2) {
  .ugl <- uniquegenelist(.ugl_genes, ndups = 2, spacing = .ugl_spacing)
  write.csv(data.frame(rowname = rownames(.ugl), sym = .ugl$sym),
            sprintf("R_ugl_spacing%d.csv", .ugl_spacing), row.names = FALSE)
}

cat("  uniquegenelist row-name fixtures complete.\n")

# =============================================================================
# lmFit(ndups > 1): fit$genes is uniquegenelist(y$probes, ndups, spacing), so
# annotation lines up with the fitted (unwrapped) rows
# =============================================================================
cat("\nGenerating lmFit ndups gene-annotation fixtures...\n")

set.seed(20261028)

.ndg_y <- matrix(rnorm(40 * 6), 40, 6)
.ndg_y[c(33, 34), 4:6] <- .ndg_y[c(33, 34), 4:6] + 5
write.csv(.ndg_y, "R_ndg_y.csv", row.names = FALSE)
.ndg_design <- cbind(1, rep(0:1, each = 3))
.ndg_genes <- data.frame(sym = paste0("s", 1:40))
for (.ndg_spacing in 1:2) {
  .ndg_fit <- eBayes(lmFit(new("EList", list(E = .ndg_y, genes = .ndg_genes)), .ndg_design,
                           ndups = 2, spacing = .ndg_spacing, correlation = 0.3))
  stopifnot(nrow(.ndg_fit$genes) == 20)
  write.csv(data.frame(sym = .ndg_fit$genes$sym, t = .ndg_fit$t[, 2]),
            sprintf("R_ndg_spacing%d.csv", .ndg_spacing), row.names = FALSE)
}

cat("  lmFit ndups gene-annotation fixtures complete.\n")

# =============================================================================
# Subsetting a voom EList by samples, v[, idx], subsets the design rows with
# the samples; lmFit(v[, idx]) then fits the matching design
# =============================================================================
cat("\nGenerating voom EList sample-subset fixtures...\n")

set.seed(20261030)

.vsi_counts <- matrix(rpois(50 * 8, 30), 50, 8)
.vsi_counts[1:10, 5:8] <- .vsi_counts[1:10, 5:8] * 3
write.csv(.vsi_counts, "R_vsi_counts.csv", row.names = FALSE)
.vsi_v <- voom(.vsi_counts, cbind(1, rep(0:1, each = 4)))
for (.vsi_case in c("reorder", "subset")) {
  .vsi_idx <- if (.vsi_case == "reorder") c(8, 1:7) else c(1, 2, 5, 6, 7)
  .vsi_sub <- .vsi_v[, .vsi_idx]
  stopifnot(identical(unname(.vsi_sub$design), unname(.vsi_v$design[.vsi_idx, , drop = FALSE])))
  .vsi_fit <- lmFit(.vsi_sub)
  write.csv(data.frame(coefficients = .vsi_fit$coefficients, sigma = .vsi_fit$sigma),
            sprintf("R_vsi_%s.csv", .vsi_case), row.names = FALSE)
}

cat("  voom EList sample-subset fixtures complete.\n")

# =============================================================================
# asMatrixWeights decides whether weights are array weights (its
# "arrayweights" attribute), which picks lm.series / gls.series' fast path:
# a full-size matrix is never array weights (1 gene + 1 x N matrix), and with
# ngenes == narrays a length-N vector is read as probe weights
# =============================================================================
cat("\nGenerating weight-shape fixtures...\n")

set.seed(20261031)

.wsh_y1 <- matrix(rnorm(4, 10), 1, 4)
.wsh_d1 <- cbind(1, c(0, 0, 1, 1))
.wsh_w <- c(1, 0.5, 2, 0.8)
.wsh_y6 <- matrix(rnorm(36, 8), 6, 6)
.wsh_d6 <- cbind(1, rep(0:1, each = 3))
.wsh_w6 <- c(1.2, 0.8, 0.7, 1.1, 1.0, 1.3)
write.csv(.wsh_y1, "R_wsh_y1.csv", row.names = FALSE)
write.csv(.wsh_y6, "R_wsh_y6.csv", row.names = FALSE)
.wsh_cases <- list(
  one_gene_matrix = lmFit(.wsh_y1, .wsh_d1, weights = matrix(.wsh_w, 1)),
  one_gene_vector = lmFit(.wsh_y1, .wsh_d1, weights = .wsh_w),
  square_vector = lmFit(.wsh_y6, .wsh_d6, weights = .wsh_w6),
  square_vector_block = lmFit(.wsh_y6, .wsh_d6, weights = .wsh_w6, block = rep(1:3, 2), correlation = 0.3)
)
stopifnot(isTRUE(all.equal(unname(.wsh_cases$one_gene_matrix$cov.coefficients), unname(solve(crossprod(.wsh_d1))))))
for (.wsh_case in names(.wsh_cases)) {
  .wsh_fit <- .wsh_cases[[.wsh_case]]
  write.csv(data.frame(coefficients = .wsh_fit$coefficients, stdev_unscaled = .wsh_fit$stdev.unscaled,
                       sigma = .wsh_fit$sigma), sprintf("R_wsh_%s.csv", .wsh_case), row.names = FALSE)
  write.csv(.wsh_fit$cov.coefficients, sprintf("R_wsh_%s_cov.csv", .wsh_case), row.names = FALSE)
}

cat("  weight-shape fixtures complete.\n")

# =============================================================================
# voomWithQualityWeights passes its sample weights to voom as a vector, which
# asMatrixWeights reads as probe weights when ngenes == nsamples (an accident
# in R). pylimma passes them as a row matrix, always array weights. Reference:
# R's steps with that one change; it equals voomWithQualityWeights whenever
# ngenes != nsamples (known_differences.rst)
# =============================================================================
cat("\nGenerating voomWithQualityWeights sample-weight fixtures...\n")

set.seed(20261101)

.vqw_intended <- function(counts, design) {
  v <- voom(counts, design)
  aw <- arrayWeights(v, design)
  v <- voom(counts, design, weights = matrix(aw, 1))
  aw <- arrayWeights(v, design)
  v$weights <- t(aw * t(v$weights))
  v$targets$sample.weights <- aw
  v
}
.vqw_design <- cbind(1, rep(0:1, each = 3))
.vqw_counts_rect <- matrix(rpois(30 * 6, 50), 30, 6)
.vqw_counts_rect[, 2] <- .vqw_counts_rect[, 2] * rpois(30, 2)
stopifnot(isTRUE(all.equal(.vqw_intended(.vqw_counts_rect, .vqw_design)$weights,
                           voomWithQualityWeights(.vqw_counts_rect, .vqw_design)$weights)))
.vqw_counts_square <- .vqw_counts_rect[1:6, ]
write.csv(.vqw_counts_square, "R_vqw_counts_square.csv", row.names = FALSE)
.vqw_square <- .vqw_intended(.vqw_counts_square, .vqw_design)
stopifnot(!isTRUE(all.equal(.vqw_square$weights, voomWithQualityWeights(.vqw_counts_square, .vqw_design)$weights)))
write.csv(.vqw_square$weights, "R_vqw_square_weights_intended.csv", row.names = FALSE)
write.csv(data.frame(sample_weights = .vqw_square$targets$sample.weights), "R_vqw_square_sample_weights_intended.csv",
          row.names = FALSE)

cat("  voomWithQualityWeights sample-weight fixtures complete.\n")

# =============================================================================
# arrayWeights(method = "genebygene") when missing values leave a gene's
# design rank-deficient: lm.wfit pivots (dqrdc2) and hat(fit$qr) uses the
# first rank columns of Q only
# =============================================================================
cat("\nGenerating arrayWeights gene-by-gene rank-deficient fixtures...\n")

set.seed(20261102)

.awg_y <- matrix(rnorm(40 * 6), 40, 6)
# design 1: intercept + group; NA in both group-0 samples -> last column redundant
.awg_d1 <- cbind(1, c(0, 0, 1, 1, 1, 1))
# design 2: intercept + two indicators; NA in both "b" samples -> middle column all zero
.awg_d2 <- cbind(1, c(0, 1, 1, 0, 0, 0), c(0, 0, 0, 1, 1, 0))
.awg_y1 <- .awg_y
.awg_y1[c(3, 9, 20), 1:2] <- NA
.awg_y2 <- .awg_y
.awg_y2[c(1, 9, 20), 2:3] <- NA
write.csv(.awg_y1, "R_awg_y1.csv", row.names = FALSE)
write.csv(.awg_y2, "R_awg_y2.csv", row.names = FALSE)
stopifnot(qr(.awg_d2[-(2:3), ])$rank == 2)
write.csv(data.frame(
  d1 = arrayWeights(.awg_y1, .awg_d1, method = "genebygene"),
  d2 = arrayWeights(.awg_y2, .awg_d2, method = "genebygene")
), "R_awg_weights.csv", row.names = FALSE)

cat("  arrayWeights gene-by-gene rank-deficient fixtures complete.\n")

# =============================================================================
# arrayWeights(method = "genebygene") skips a gene whose observed design is
# saturated (residual df < 2) before using its residual variance
# =============================================================================
cat("\nGenerating arrayWeights saturated-gene fixtures...\n")

set.seed(20261108)

.aws_y <- matrix(rnorm(30 * 6), 30, 6)
.aws_y[4, c(1, 3, 5)] <- NA
write.csv(.aws_y, "R_aws_y.csv", row.names = FALSE)
.aws_d <- cbind(1, c(0, 1, 0, 1, 0, 0), c(0, 0, 0, 1, 0, 1))
stopifnot(qr(.aws_d[c(2, 4, 6), ])$rank == 3)
write.csv(data.frame(aw = arrayWeights(.aws_y, .aws_d, method = "genebygene")), "R_aws_weights.csv",
          row.names = FALSE)

cat("  arrayWeights saturated-gene fixtures complete.\n")

# =============================================================================
# Rank decisions use R's qr() (LINPACK dqrdc2, tolerance 1e-7 relative to the
# column norms): a tiny-scale column and a near-collinear column are both
# dropped by vooma's lm.fit and arrayWeights' design check
# =============================================================================
cat("\nGenerating rank-tolerance fixtures...\n")

set.seed(20261103)

.rtl_E <- matrix(rnorm(200 * 8), 200, 8)
write.csv(.rtl_E, "R_rtl_E.csv", row.names = FALSE)
.rtl_x <- rep(0:1, each = 4)
.rtl_designs <- list(
  tiny = cbind(1, .rtl_x * 3e-11),
  collinear = cbind(1, .rtl_x, .rtl_x + 1e-8 * c(1, -1, 2, 0, 1, -2, 0, 1))
)
for (.rtl_case in names(.rtl_designs)) {
  .rtl_d <- .rtl_designs[[.rtl_case]]
  write.csv(.rtl_d, sprintf("R_rtl_design_%s.csv", .rtl_case), row.names = FALSE)
  write.csv(vooma(.rtl_E, .rtl_d)$weights, sprintf("R_rtl_vooma_%s.csv", .rtl_case), row.names = FALSE)
  write.csv(data.frame(aw = arrayWeights(.rtl_E, .rtl_d)), sprintf("R_rtl_aw_%s.csv", .rtl_case), row.names = FALSE)
}

cat("  rank-tolerance fixtures complete.\n")

# =============================================================================
# Numeric batch labels: as.factor() merges values whose as.character() labels
# coincide (0.3 and 0.1 + 0.2 are one level, "0.3")
# =============================================================================
cat("\nGenerating removeBatchEffect numeric-label fixtures...\n")

set.seed(20261107)

.rbl_y <- matrix(rnorm(20 * 6), 20, 6)
write.csv(.rbl_y, "R_rbl_y.csv", row.names = FALSE)
.rbl_batch <- c(0.3, 0.1 + 0.2, 0.3, 0.1 + 0.2, 1, 1)
stopifnot(nlevels(as.factor(.rbl_batch)) == 2)
write.csv(removeBatchEffect(.rbl_y, batch = .rbl_batch), "R_rbl_corrected.csv", row.names = FALSE)

cat("  removeBatchEffect numeric-label fixtures complete.\n")

# =============================================================================
# modifyWeights compares as.character(status) with as.character(values), so a
# double status 1 matches the value 1 and a logical status matches "TRUE"
# =============================================================================
cat("\nGenerating modifyWeights fixtures...\n")

.mwt_w <- c(1, 1, 1, 1)
write.csv(data.frame(
  double_status = modifyWeights(.mwt_w, status = c(0, 1, 0, 1), values = 1, multipliers = 0),
  logical_status = modifyWeights(.mwt_w, status = c(TRUE, FALSE, TRUE, FALSE), values = "TRUE", multipliers = 0),
  integer_status_double_value = modifyWeights(.mwt_w, status = c(0L, 1L, 0L, 1L), values = 1.0, multipliers = 0),
  fraction_status = modifyWeights(.mwt_w, status = c(0.5, 0.1 + 0.2, 0.5, 2), values = c(0.3, 2), multipliers = 0),
  character_status = modifyWeights(.mwt_w, status = c("gene", "ctrl", "gene", "blank"), values = c("ctrl", "blank"),
                                   multipliers = c(0, 2))
), "R_mwt.csv", row.names = FALSE)

cat("  modifyWeights fixtures complete.\n")

# =============================================================================
# voomaByGroup fits each group with its own rows of the design, block[i] and
# the correlation (designi <- design[i, , drop = FALSE])
# =============================================================================
cat("\nGenerating voomaByGroup fixtures...\n")

set.seed(20261104)

.vbg_y <- matrix(rnorm(100 * 8, mean = 6), 100, 8)
.vbg_y[1:10, 5:8] <- .vbg_y[1:10, 5:8] + 2
write.csv(.vbg_y, "R_vbg_y.csv", row.names = FALSE)
.vbg_group <- rep(c("a", "b"), 4)
.vbg_design <- cbind(1, rep(0:1, each = 4))
write.csv(voomaByGroup(.vbg_y, group = .vbg_group, design = .vbg_design)$weights, "R_vbg_design.csv",
          row.names = FALSE)
write.csv(voomaByGroup(.vbg_y, group = .vbg_group, design = .vbg_design, block = rep(1:4, each = 2),
                       correlation = 0.4)$weights, "R_vbg_block.csv", row.names = FALSE)

cat("  voomaByGroup fixtures complete.\n")

# =============================================================================
# voomaByGroup with a singleton level: its weights come from one vooma fit to
# all samples (voomall)
# =============================================================================
cat("\nGenerating voomaByGroup singleton fixtures...\n")

write.csv(voomaByGroup(.vbg_y, group = c("a", "b", "a", "b", "c", "b", "a", "b"), design = .vbg_design)$weights,
          "R_vbg_singleton.csv", row.names = FALSE)

cat("  voomaByGroup singleton fixtures complete.\n")

# =============================================================================
# arrayWeights.Rd: design is "extracted from the data object if available",
# but the code never reads y$design (an omission in R). pylimma follows the
# documentation (known_differences.rst); this records the literal behaviour.
# No CSV.
# =============================================================================
cat("\nChecking arrayWeights with an EList design slot...\n")

set.seed(20261106)

.awd_E <- matrix(rnorm(50 * 6), 50, 6)
.awd_E[1:10, 4:6] <- .awd_E[1:10, 4:6] + 3
.awd_design <- cbind(1, rep(0:1, each = 3))
.awd_el <- new("EList", list(E = .awd_E, design = .awd_design))
stopifnot(!is.null(getEAWP(.awd_el)$design))
stopifnot(isTRUE(all.equal(arrayWeights(.awd_el), arrayWeights(.awd_E))))
stopifnot(!isTRUE(all.equal(arrayWeights(.awd_el), arrayWeights(.awd_E, .awd_design))))

cat("  arrayWeights EList design slot checked.\n")

# =============================================================================
# removeBatchEffect zeroes only the NA batch coefficients
# (beta[is.na(beta)] <- 0): a gene whose design-of-interest coefficient is
# not estimable keeps its estimable batch effect
# =============================================================================
cat("\nGenerating removeBatchEffect NA-coefficient fixtures...\n")

set.seed(20261109)

.rbn_y <- matrix(rnorm(6 * 4, mean = 8), 6, 4)
.rbn_y[3, 1:2] <- NA
.rbn_y[5, 1] <- NA
write.csv(.rbn_y, "R_rbn_y.csv", row.names = FALSE)
.rbn_design <- cbind(1, c(0, 0, 1, 1))
.rbn_fit <- suppressWarnings(lmFit(.rbn_y, cbind(.rbn_design, c(1, -1, 1, -1))))
stopifnot(is.na(.rbn_fit$coefficients[3, 2]), !is.na(.rbn_fit$coefficients[3, 3]))
write.csv(suppressWarnings(removeBatchEffect(.rbn_y, batch = c(1, 0, 1, 0), design = .rbn_design)),
          "R_rbn_corrected.csv", row.names = FALSE)

cat("  removeBatchEffect NA-coefficient fixtures complete.\n")

# =============================================================================
# removeBatchEffect with a single-level batch or batch2: contr.sum stops
# ("not enough degrees of freedom to define contrasts"). No CSV.
# =============================================================================
cat("\nChecking removeBatchEffect single-level batch...\n")

.rb1_y <- matrix(rnorm(5 * 3), 5, 3)
for (.rb1_args in list(list(batch = c("a", "a", "a")), list(batch = c(1, 2, 1), batch2 = c("x", "x", "x")))) {
  .rb1_err <- tryCatch({ do.call(removeBatchEffect, c(list(.rb1_y), .rb1_args)); NA_character_ },
                       error = function(e) conditionMessage(e))
  stopifnot(identical(.rb1_err, "not enough degrees of freedom to define contrasts"))
}

cat("  removeBatchEffect single-level batch checked.\n")

# =============================================================================
# wsva stops for a rank-deficient design (.lmEffects: "design must be full
# column rank"). No CSV.
# =============================================================================
cat("\nChecking wsva rank-deficient design...\n")

set.seed(20261110)

.wrd_y <- matrix(rnorm(10 * 8), 10, 8)
.wrd_g1 <- c(0, 1, 0, 1, 0, 1, 0, 1)
.wrd_g2 <- c(0, 0, 1, 1, 0, 0, 1, 1)
.wrd_err <- tryCatch({ wsva(.wrd_y, cbind(1, .wrd_g1, .wrd_g2, .wrd_g1 + .wrd_g2)); NA_character_ },
                     error = function(e) conditionMessage(e))
stopifnot(identical(.wrd_err, "design must be full column rank"))

cat("  wsva rank-deficient design checked.\n")

# =============================================================================
# removeBatchEffect with a missing batch label: as.factor() leaves NA out of
# the levels, so R stops (contr.sum, or a row mismatch from model.matrix);
# pylimma raises a clear error (known_differences.rst). A numeric NaN is a
# level ("NaN") and runs.
# =============================================================================
cat("\nGenerating removeBatchEffect missing-label fixtures...\n")

set.seed(20261111)

.rbm_y <- matrix(rnorm(5 * 3), 5, 3)
write.csv(.rbm_y, "R_rbm_y.csv", row.names = FALSE)
for (.rbm_b in list(c("a", NA, "a"), c("a", NA, "b"))) {
  stopifnot(inherits(tryCatch(suppressMessages(removeBatchEffect(.rbm_y, batch = .rbm_b)), error = identity), "error"))
}
write.csv(suppressMessages(removeBatchEffect(.rbm_y, batch = c(1, NaN, 1))), "R_rbm_nan_level.csv", row.names = FALSE)

cat("  removeBatchEffect missing-label fixtures complete.\n")

# =============================================================================
# Input errors: wsva with NA in y stops in .lmEffects ("All y values must be
# finite and non-NA"); removeBatchEffect with a batch of the wrong length
# stops in cbind. pylimma raises the first verbatim and a clearer message
# for the second. No CSV.
# =============================================================================
cat("\nChecking wsva / removeBatchEffect input errors...\n")

.wie_y <- matrix(rnorm(10 * 8), 10, 8)
.wie_y_na <- .wie_y
.wie_y_na[2, 3] <- NA
stopifnot(identical(tryCatch(wsva(.wie_y_na, cbind(1, rep(0:1, 4))), error = conditionMessage),
                    "All y values must be finite and non-NA"))
stopifnot(identical(tryCatch(suppressMessages(removeBatchEffect(.wie_y, batch = rep(c("a", "b"), 4)[1:7])),
                             error = conditionMessage),
                    "number of rows of matrices must match (see arg 2)"))

cat("  wsva / removeBatchEffect input errors checked.\n")

# =============================================================================
# Gene-set index rules: a character set is which(geneid %in% set) (every row
# whose ID is in the set, so duplicated IDs all count), a logical set its TRUE
# rows; camera / cameraPR / mroast resolve sets this way
# =============================================================================
cat("\nGenerating gene-set index fixtures...\n")

set.seed(20261112)

.gsi_y <- matrix(rnorm(40 * 6), 40, 6)
.gsi_y[1:8, 4:6] <- .gsi_y[1:8, 4:6] + 1.5
.gsi_ids <- paste0("g", 1:40)
.gsi_ids[12] <- "g3"   # duplicated ID: row 12 is also "g3"
rownames(.gsi_y) <- .gsi_ids
write.csv(data.frame(id = .gsi_ids, .gsi_y), "R_gsi_y.csv", row.names = FALSE)
.gsi_d <- cbind(1, rep(0:1, each = 3))
.gsi_sets <- list(a = c("g1", "g2", "g3", "g5", "nope"), b = paste0("g", 20:30))
stopifnot(identical(which(.gsi_ids %in% .gsi_sets$a), c(1L, 2L, 3L, 5L, 12L)))
write.csv(camera(.gsi_y, .gsi_sets, .gsi_d, sort = FALSE), "R_gsi_camera_character.csv")
.gsi_mask <- seq_len(40) %in% c(1:6, 15)
write.csv(camera(.gsi_y, list(m = .gsi_mask), .gsi_d), "R_gsi_camera_logical.csv")
.gsi_t <- setNames(rnorm(40), .gsi_ids)
.gsi_t[1:6] <- .gsi_t[1:6] + 2
write.csv(data.frame(t = unname(.gsi_t)), "R_gsi_t.csv", row.names = FALSE)
write.csv(cameraPR(.gsi_t, list(m = .gsi_mask, a = .gsi_sets$a), sort = FALSE), "R_gsi_camerapr.csv")
.gsi_mr <- mroast(.gsi_y, list(a = .gsi_sets$a, w = data.frame(id = c("g7", "g1", "g3"), w = c(1, 2, -1))),
                  .gsi_d, nrot = 99, sort = "none")
write.csv(data.frame(set = rownames(.gsi_mr), NGenes = .gsi_mr$NGenes), "R_gsi_mroast_ngenes.csv", row.names = FALSE)

cat("  gene-set index fixtures complete.\n")

# =============================================================================
# geneSetTest / rankSumTestWithCorrelation subscript statistics[index]: a
# character index selects by name (repeats count), a logical index its TRUE
# positions
# =============================================================================
cat("\nGenerating geneSetTest index fixtures...\n")

set.seed(20261113)

.gst_t <- setNames(rnorm(30), paste0("g", 1:30))
.gst_t[1:5] <- .gst_t[1:5] + 1.5
write.csv(data.frame(t = unname(.gst_t)), "R_gst_t.csv", row.names = FALSE)
.gst_mask <- seq_len(30) %in% c(1:5, 12)
write.csv(data.frame(
  case = c("character", "character_repeat", "logical"),
  gst = c(geneSetTest(c("g1", "g2", "g3", "g12"), .gst_t),
          geneSetTest(c("g1", "g2", "g2", "g12"), .gst_t),
          geneSetTest(.gst_mask, .gst_t)),
  rank_less = c(rankSumTestWithCorrelation(c("g1", "g2", "g3", "g12"), .gst_t)["less"],
                rankSumTestWithCorrelation(c("g1", "g2", "g2", "g12"), .gst_t)["less"],
                rankSumTestWithCorrelation(.gst_mask, .gst_t)["less"]),
  wilcox = c(wilcoxGST(c("g1", "g2", "g3", "g12"), .gst_t), wilcoxGST(c("g1", "g2", "g2", "g12"), .gst_t),
             wilcoxGST(.gst_mask, .gst_t))
), "R_gst_index.csv", row.names = FALSE)

# A position past the end subscripts NA, giving NA p-values; a negative
# position drops that gene; a shorter logical set is recycled, a longer one
# gives NA (pylimma raises for all)
stopifnot(is.na(geneSetTest(c(1, 31), .gst_t)),
          all(is.na(rankSumTestWithCorrelation(c(1, 31), .gst_t))),
          geneSetTest(-(1:28), .gst_t) == geneSetTest(29:30, .gst_t),
          geneSetTest(c(TRUE, FALSE), .gst_t) == geneSetTest(seq(1, 29, by = 2), .gst_t),
          is.na(geneSetTest(c(rep(FALSE, 30), TRUE), .gst_t)))

cat("  geneSetTest index fixtures complete.\n")

# =============================================================================
# ids2indices: which(identifiers %in% as.character(x)) per set, so duplicated
# identifiers all count and repeated members do not; a non-list is Set1
# =============================================================================
cat("\nGenerating ids2indices fixtures...\n")

.i2i_ids <- c("a", "b", "a", "c", "10", "d")
.i2i <- list(
  dup = ids2indices(list(s = c("a", "a", "b")), .i2i_ids),
  unnamed = ids2indices(list(c("b", "c"), "d", "zz"), .i2i_ids),
  single = ids2indices("a", .i2i_ids),
  numeric = ids2indices(list(n = 10), .i2i_ids),
  keep_empty = ids2indices(list(e = "zz", s = "c"), .i2i_ids, remove.empty = FALSE)
)
write.csv(do.call(rbind, lapply(names(.i2i), function(case) {
  x <- .i2i[[case]]
  nms <- if (is.null(names(x))) rep("", length(x)) else names(x)
  data.frame(case = case, set = seq_along(x), name = nms,
             rows = vapply(x, function(v) paste(v - 1L, collapse = ";"), ""))
})), "R_i2i.csv", row.names = FALSE)

cat("  ids2indices fixtures complete.\n")

# =============================================================================
# interGeneCorrelation: vif <- m * mean(colMeans(y)^2) after scaling each
# gene's residual effects, i.e. averaging across genes; rank from qr(design);
# an all-zero gene gives NaN (0/0)
# =============================================================================
cat("\nGenerating interGeneCorrelation fixtures...\n")

set.seed(20261114)

.igc_y <- matrix(rnorm(25 * 8), 25, 8) + rnorm(8)   # shared sample effect: positive correlation
write.csv(.igc_y, "R_igc_y.csv", row.names = FALSE)
.igc_d <- cbind(1, rep(0:1, each = 4))
.igc_d_rd <- cbind(.igc_d, 2 * .igc_d[, 2])
.igc_y_zero <- .igc_y
.igc_y_zero[3, ] <- 0
.igc <- list(full = interGeneCorrelation(.igc_y, .igc_d), rank_deficient = interGeneCorrelation(.igc_y, .igc_d_rd),
             zero_gene = interGeneCorrelation(.igc_y_zero, .igc_d))
stopifnot(isTRUE(all.equal(.igc$full, .igc$rank_deficient)), is.nan(.igc$zero_gene$vif))
write.csv(data.frame(case = names(.igc), vif = sapply(.igc, `[[`, "vif"), correlation = sapply(.igc, `[[`, "correlation")),
          "R_igc.csv", row.names = FALSE)

cat("  interGeneCorrelation fixtures complete.\n")

# =============================================================================
# camera(trend.var = TRUE) with array weights: y is scaled by sqrt(weights)
# before the trend's average expression is computed
# =============================================================================
cat("\nGenerating camera weighted trend fixtures...\n")

set.seed(20261115)

.cwt_y <- matrix(rnorm(300 * 6, mean = 6), 300, 6)
.cwt_y[1:30, 4:6] <- .cwt_y[1:30, 4:6] + 0.8
write.csv(.cwt_y, "R_cwt_y.csv", row.names = FALSE)
.cwt_d <- cbind(1, rep(0:1, each = 3))
.cwt_w <- c(0.3, 1, 2, 0.5, 1.5, 1)
.cwt_sets <- list(a = 1:30, b = 31:70)
write.csv(camera(.cwt_y, .cwt_sets, .cwt_d, weights = .cwt_w, trend.var = TRUE, sort = FALSE), "R_cwt_camera.csv")

cat("  camera weighted trend fixtures complete.\n")

# =============================================================================
# roast / mroast propagate NA: an empty set, a NA gene weight or a negative
# df.prior give NA statistics, so sum(statrot > statobs) and the p-values are
# NA; mroast sorts such sets last. Only the NA pattern and NGenes are
# recorded (the finite p-values depend on the rotations).
# =============================================================================
cat("\nGenerating roast NA-propagation fixtures...\n")

set.seed(20261116)

.rna_y <- matrix(rnorm(20 * 6), 20, 6)
write.csv(.rna_y, "R_rna_y.csv", row.names = FALSE)
.rna_d <- cbind(1, rep(0:1, each = 3))
.rna_mr <- mroast(.rna_y, list(A = 1:5, C = integer(0), B = 6:10), .rna_d, nrot = 99)
.rna_gw <- roast(.rna_y, 1:5, .rna_d, gene.weights = c(1, NA, 1, 1, 1), nrot = 99)$p.value
.rna_df <- suppressWarnings(roast(.rna_y, 1:5, .rna_d, var.prior = 1, df.prior = -3, nrot = 99))$p.value
write.csv(data.frame(set = rownames(.rna_mr), NGenes = .rna_mr$NGenes, PValue_na = is.na(.rna_mr$PValue),
                     FDR_na = is.na(.rna_mr$FDR)), "R_rna_mroast.csv", row.names = FALSE)
write.csv(data.frame(case = rep(c("gene_weight_na", "df_prior_negative"), each = 4),
                     active_prop = c(.rna_gw$Active.Prop, .rna_df$Active.Prop),
                     p_value_na = c(is.na(.rna_gw$P.Value), is.na(.rna_df$P.Value))), "R_rna_roast.csv", row.names = FALSE)

cat("  roast NA-propagation fixtures complete.\n")

# =============================================================================
# camera / cameraPR with a set of all genes (m2 = 0) or no genes (m = 0):
# double arithmetic gives a NaN p-value row, not an error
# =============================================================================
cat("\nGenerating camera degenerate-set fixtures...\n")

set.seed(20261117)

.cds_y <- matrix(rnorm(20 * 6), 20, 6)
write.csv(.cds_y, "R_cds_y.csv", row.names = FALSE)
.cds_d <- cbind(1, rep(0:1, each = 3))
.cds_cam <- camera(.cds_y, list(all = 1:20, none = integer(0), some = 1:5), .cds_d, sort = FALSE)
.cds_pr <- cameraPR(.cds_y[, 1], list(all = 1:20, some = 1:5), sort = FALSE)
stopifnot(is.na(.cds_cam["all", "PValue"]), is.na(.cds_cam["none", "PValue"]), is.na(.cds_pr["all", "PValue"]))
write.csv(.cds_cam, "R_cds_camera.csv")
write.csv(.cds_pr, "R_cds_camerapr.csv")

cat("  camera degenerate-set fixtures complete.\n")

# =============================================================================
# topRomer orders a romer table by Up / Down / Mixed with ties broken by
# Mixed (or min(Up, Down)) and then -NGenes
# =============================================================================
cat("\nGenerating topRomer fixtures...\n")

.trm <- cbind(NGenes = c(5, 12, 8, 12, 3), Up = c(0.01, 0.2, 0.01, 0.05, 0.2),
              Down = c(0.9, 0.3, 0.5, 0.01, 0.01), Mixed = c(0.02, 0.1, 0.02, 0.03, 0.4))
rownames(.trm) <- paste0("s", 1:5)
write.csv(.trm, "R_trm_table.csv")
write.csv(data.frame(alternative = rep(c("up", "down", "mixed"), each = 3),
                     set = c(rownames(topRomer(.trm, 3, "up")), rownames(topRomer(.trm, 3, "down")),
                             rownames(topRomer(.trm, 3, "mixed")))), "R_trm_order.csv", row.names = FALSE)

cat("  topRomer fixtures complete.\n")

# =============================================================================
# camera / fry (.lmEffects) check full rank with qr(), whose tolerance is
# relative to each column's norm: a tiny-scale column is still full rank
# =============================================================================
cat("\nGenerating gene-set rank-tolerance fixtures...\n")

set.seed(20261118)

.gsr_y <- matrix(rnorm(30 * 6), 30, 6)
.gsr_y[1:6, 4:6] <- .gsr_y[1:6, 4:6] + 1
write.csv(.gsr_y, "R_gsr_y.csv", row.names = FALSE)
.gsr_d <- cbind(1, rep(0:1, each = 3) * 3e-11)
write.csv(.gsr_d, "R_gsr_design.csv", row.names = FALSE)
write.csv(camera(.gsr_y, list(a = 1:6, b = 7:15), .gsr_d, sort = FALSE), "R_gsr_camera.csv")
write.csv(fry(.gsr_y, list(a = 1:6, b = 7:15), .gsr_d, sort = "none"), "R_gsr_fry.csv")

cat("  gene-set rank-tolerance fixtures complete.\n")

# =============================================================================
# goana / kegga: a single gene ID is one DE set; a data.frame de is an error
# =============================================================================
cat("\nGenerating enrichment de-input checks...\n")

.ed_single <- kegga("gene001", universe = goana_universe, gene.pathway = kegga_gp, pathway.names = kegga_pn)
stopifnot(identical(.ed_single, kegga(list(DE = "gene001"), universe = goana_universe,
                                      gene.pathway = kegga_gp, pathway.names = kegga_pn)))
.ed_df <- tryCatch(kegga(data.frame(x = de_up), universe = goana_universe, gene.pathway = kegga_gp,
                         pathway.names = kegga_pn), error = conditionMessage)
stopifnot(identical(.ed_df, "de should be a list of character vectors. It should not be a data.frame."))
.ed_null <- tryCatch(kegga(NULL, universe = goana_universe, gene.pathway = kegga_gp, pathway.names = kegga_pn),
                     error = conditionMessage)
stopifnot(identical(.ed_null, "components of de should be vectors"))
.ed_list <- kegga(list(c("gene001", "gene002"), "gene003"), universe = goana_universe, gene.pathway = kegga_gp,
                  pathway.names = kegga_pn)
stopifnot(identical(colnames(.ed_list), c("Pathway", "N", "DE1", "DE2", "P.DE1", "P.DE2")))

cat("  enrichment de-input checks complete.\n")

# =============================================================================
# kegga(restrict.universe = TRUE) drops universe genes absent from
# gene.pathway; an empty explicit universe stops
# =============================================================================
cat("\nGenerating kegga restrict.universe fixtures...\n")

write.csv(kegga(goana_de_list, universe = goana_universe, restrict.universe = TRUE,
                gene.pathway = kegga_gp, pathway.names = kegga_pn), "R_kegga_restrict.csv")
.kr_empty <- tryCatch(kegga(de_up, universe = character(0), gene.pathway = kegga_gp, pathway.names = kegga_pn),
                      error = conditionMessage)
stopifnot(identical(.kr_empty, "No genes in universe"))

cat("  kegga restrict.universe fixtures complete.\n")

# =============================================================================
# goana / kegga: null.prob and covariate are ignored (with a message) when
# universe is not set, and a covariate is dropped when no gene is DE;
# otherwise null.prob switches to Wallenius' test (not ported to pylimma)
# =============================================================================
cat("\nGenerating enrichment null.prob checks...\n")

.np_plain <- kegga(de_up, gene.pathway = kegga_gp, pathway.names = kegga_pn)
stopifnot(identical(suppressMessages(kegga(de_up, gene.pathway = kegga_gp, pathway.names = kegga_pn,
                                           covariate = 1:3, null.prob = 1:3)), .np_plain))
.np_cov <- seq_along(goana_universe) / 200
stopifnot(identical(
  suppressMessages(kegga("not_a_gene", universe = goana_universe, covariate = .np_cov,
                         gene.pathway = kegga_gp, pathway.names = kegga_pn)),
  kegga("not_a_gene", universe = goana_universe, gene.pathway = kegga_gp, pathway.names = kegga_pn)))
.np_wallenius <- kegga(de_up, universe = goana_universe, null.prob = .np_cov,
                       gene.pathway = kegga_gp, pathway.names = kegga_pn)
stopifnot(!isTRUE(all.equal(.np_wallenius$P.DE,
                            kegga(de_up, universe = goana_universe, gene.pathway = kegga_gp,
                                  pathway.names = kegga_pn)$P.DE)))

cat("  enrichment null.prob checks complete.\n")

# =============================================================================
# goana / kegga on a fit define their own universe and covariate
# =============================================================================
cat("\nGenerating enrichment fit-argument checks...\n")

set.seed(20261121)
.efa_y <- matrix(rnorm(200 * 6), 200, 6, dimnames = list(goana_universe, NULL))
.efa_y[1:15, 4:6] <- .efa_y[1:15, 4:6] + 3
.efa_fit <- eBayes(lmFit(.efa_y, cbind(1, rep(0:1, each = 3))))
.efa_msg <- function(...) tryCatch(kegga(.efa_fit, gene.pathway = kegga_gp, pathway.names = kegga_pn, ...),
                                   error = conditionMessage)
stopifnot(identical(.efa_msg(universe = goana_universe), "kegga.MArrayLM defines its own universe"))
# with trend = FALSE the covariate collides with the one kegga.MArrayLM passes on
stopifnot(grepl("matched by multiple actual arguments", .efa_msg(covariate = rnorm(200))))

cat("  enrichment fit-argument checks complete.\n")

# =============================================================================
# kegga pathway.names: a duplicated pathway ID takes its first row (match);
# rows with an NA ID or name are dropped
# =============================================================================
cat("\nGenerating kegga pathway.names fixtures...\n")

.kpn <- rbind(kegga_pn, data.frame(path_id = c("path:hsa00010", "path:hsa00020"),
                                   description = c("Duplicate glycolysis", "Duplicate citrate")))
.kpn$description[.kpn$path_id == "path:hsa00030"] <- NA
write.csv(.kpn, "R_kpn_names.csv", row.names = FALSE)
.kpn_res <- kegga(de_up, universe = goana_universe, gene.pathway = kegga_gp, pathway.names = .kpn)
stopifnot(.kpn_res["path:hsa00010", "Pathway"] == "Glycolysis", is.na(.kpn_res["path:hsa00030", "Pathway"]))
write.csv(.kpn_res, "R_kpn_kegga.csv")

cat("  kegga pathway.names fixtures complete.\n")

# =============================================================================
# topGO / topKEGG order by P, N and name; a missing name sorts last, and so
# does a missing p-value (pmin gives NA)
# =============================================================================
cat("\nGenerating topGO / topKEGG NA-name fixtures...\n")

.tna <- data.frame(Term = c("b term", NA, "a term", NA, "c term", "d term"), Ont = "BP",
                   N = c(5, 5, 5, 3, 5, 2), Up = 1, Down = 0,
                   P.Up = c(0.01, 0.01, 0.01, 0.01, 0.2, 1e-10), P.Down = c(1, 1, 1, 1, 1, NA),
                   row.names = paste0("GO:", 1:6))
write.csv(.tna, "R_tna_table.csv")
write.csv(data.frame(set = rownames(topGO(.tna))), "R_tna_topgo.csv", row.names = FALSE)
.tna_kegg <- data.frame(Pathway = .tna$Term, .tna[, c("N", "Up", "Down", "P.Up", "P.Down")],
                        row.names = paste0("path:", 1:6))
write.csv(data.frame(set = rownames(topKEGG(.tna_kegg))), "R_tna_topkegg.csv", row.names = FALSE)

cat("  topGO / topKEGG NA-name fixtures complete.\n")

# =============================================================================
# topGO: ontology may be one string or an abbreviation (match.arg); number
# is capped at the row count, so Inf gives every row
# =============================================================================
cat("\nGenerating topGO argument checks...\n")

.tga <- goana_result
.tga_bp <- topGO(.tga, ontology = c("BP", "BP"))
stopifnot(identical(topGO(.tga, ontology = "BP"), .tga_bp), identical(topGO(.tga, ontology = "B"), .tga_bp),
          all(.tga_bp$Ont == "BP"))
# literal R: an element that matches nothing is dropped while another matches
stopifnot(identical(topGO(.tga, ontology = c("BP", "XX")), .tga_bp))
stopifnot(identical(topGO(.tga, number = Inf), topGO(.tga, number = nrow(.tga))),
          identical(topKEGG(res_kegga, number = Inf), topKEGG(res_kegga, number = nrow(res_kegga))))

cat("  topGO argument checks complete.\n")

# =============================================================================
# goana / kegga keep set names as given, so "A" and "a" are two sets
# (pylimma lower-cases its output columns and raises on such a clash)
# =============================================================================
.kcase <- kegga(list(A = de_up, a = de_up[1:2]), universe = goana_universe, gene.pathway = kegga_gp,
                pathway.names = kegga_pn)
stopifnot(identical(colnames(.kcase), c("Pathway", "N", "A", "a", "P.A", "P.a")))

# =============================================================================
# goanaTrend index.de: literal R behaviour for positions pylimma rejects
# (a negative position excludes, a fraction is truncated, a position past
# the end extends isDE; pylimma raises for all three)
# =============================================================================
.gti_cov <- (1:10) / 10
stopifnot(identical(goanaTrend(-1, .gti_cov), goanaTrend(2:10, .gti_cov)),
          identical(goanaTrend(2.7, .gti_cov), goanaTrend(2, .gti_cov)),
          length(goanaTrend(11, .gti_cov)) == 10)

# =============================================================================
# goana / kegga compare gene IDs as as.character(): numeric IDs match their
# character form, and NA IDs (is.na) are dropped from the universe
# =============================================================================
.eid_gp <- data.frame(gene = c(1, 2, 3, 4, 5), path = c("p1", "p1", "p1", "p2", "p2"))
.eid_num <- kegga(c(1, 2), universe = c(1:5, NA), gene.pathway = .eid_gp, pathway.names = data.frame(c("p1", "p2"), c("one", "two")))
.eid_chr <- kegga(c("1", "2"), universe = as.character(1:5), gene.pathway = data.frame(gene = as.character(1:5), path = .eid_gp$path),
                  pathway.names = data.frame(c("p1", "p2"), c("one", "two")))
stopifnot(identical(.eid_num, .eid_chr), .eid_num["p1", "DE"] == 2)

# =============================================================================
# squeezeVar with a covariate at genome scale (30,000 genes). Inputs are a
# formula of the gene index, so the Python test rebuilds them exactly.
# =============================================================================
cat("\nGenerating large-scale squeezeVar trend fixtures...\n")

.svl_i <- 1:30000
.svl_x <- exp(1.2 * sin(.svl_i * 0.37) + 0.8 * cos(.svl_i * 1.3) + 0.5 * sin(.svl_i * 0.011))
.svl_cov <- (.svl_i %% 997) / 997 + .svl_i / 30000
.svl <- squeezeVar(.svl_x, df = 4, covariate = .svl_cov)
.svl_pick <- c(1, 2, 500, 9999, 15000, 29999, 30000)
write.csv(data.frame(index = .svl_pick, var_prior = .svl$var.prior[.svl_pick], var_post = .svl$var.post[.svl_pick],
                     df_prior = .svl$df.prior), "R_svl_trend.csv", row.names = FALSE)

cat("  large-scale squeezeVar trend fixtures complete.\n")

# =============================================================================
# Extreme t tails on the log scale: pt(..., log.p = TRUE) and
# qt(..., log.p = TRUE) beyond double-precision underflow, and
# tmixture.vector with unequal df and |t| in that range (eBayes robust)
# =============================================================================
cat("\nGenerating extreme t-tail fixtures...\n")

.ett <- expand.grid(log_p = c(-50, -300, -700, -745, -760, -800, -1500, -3000, -1e5),
                    df = c(1, 3, 10, 98, 988, 1e4, 5e5, 1e6, Inf))
.ett$qt <- qt(.ett$log_p, .ett$df, lower.tail = FALSE, log.p = TRUE)
write.csv(.ett, "R_ett_qt.csv", row.names = FALSE)
.etp <- expand.grid(t = c(2, 10, 38, 40, 60, 100, 1e3, 1e6, 1e30), df = c(1, 3, 10, 98, 988, 1e4, 5e5, 1e6, Inf))
.etp$log_sf <- pt(.etp$t, .etp$df, lower.tail = FALSE, log.p = TRUE)
write.csv(.etp, "R_ett_pt.csv", row.names = FALSE)

.etm_t <- c(seq(-4, 4, length.out = 180), c(60, 75, 90, 120, 150, 200, 250, 300, 400, 500) * rep(c(1, -1), 5),
            seq(5, 14, length.out = 10))
.etm_df <- rep(c(988, 980), length.out = length(.etm_t))
.etm_su <- rep(c(0.5, 0.7, 1), length.out = length(.etm_t))
write.csv(data.frame(t = .etm_t, df = .etm_df, stdev_unscaled = .etm_su), "R_etm_input.csv", row.names = FALSE)
# very large df (df.total can reach the pooled residual df)
.etm_big_t <- rep(c(38, 3, 2, 1, 0.5), 20)
.etm_big_df <- c(rep(1e6, 99), 2e6)
write.csv(data.frame(v0 = c(limma:::tmixture.vector(.etm_t, .etm_su, .etm_df, proportion = 0.1),
                            limma:::tmixture.vector(.etm_big_t, rep(1, 100), .etm_big_df, proportion = 0.1))),
          "R_etm_v0.csv", row.names = FALSE)

cat("  extreme t-tail fixtures complete.\n")

# =============================================================================
# zscoreT (default quantile method) for t-statistics whose tail probability
# underflows double precision
# =============================================================================
.zte <- expand.grid(x = c(-500, -100, -50, -5, 0, 2, 50, 100, 500, 1e4), df = c(3, 10, 98, 988, 1e6, Inf))
.zte$z <- zscoreT(.zte$x, df = .zte$df)
write.csv(.zte, "R_zte_zscoret.csv", row.names = FALSE)

# =============================================================================
# fitFDistRobustly with unequal df1 when the non-robust df2 is Inf: pf / qf
# with df2 = Inf (the chi-squared limit) transform x to the largest df1
# =============================================================================
# chi-squared quantiles shrunk towards 1 (under-dispersed), so df2 is Inf
.fri_df1 <- rep(c(3, 6), each = 30)
.fri_x <- 0.5 * sqrt(qchisq(rep(ppoints(30), 2), df = .fri_df1) / .fri_df1)
.fri <- fitFDistRobustly(.fri_x, df1 = .fri_df1)
stopifnot(limma:::fitFDist(.fri_x, df1 = .fri_df1)$df2 == Inf, .fri$df2 == Inf)
write.csv(data.frame(x = .fri_x, df1 = .fri_df1, scale = .fri$scale, tail_p_value = .fri$tail.p.value,
                     df2_shrunk = .fri$df2.shrunk), "R_fri_fitfdistrobustly.csv", row.names = FALSE)

# =============================================================================
# squeezeVar and the fitFDist family take a single variance as a length-1
# vector
# =============================================================================
stopifnot(identical(squeezeVar(1.5, 5), list(var.post = 1.5, var.prior = 1.5, df.prior = 0)),
          identical(limma:::fitFDist(1.5, 5), list(scale = 1.5, df2 = 0)),
          all(is.na(unlist(fitFDistRobustly(1.5, 5)))),
          all(is.na(unlist(limma:::fitFDistUnequalDF1(1.5, 5)))))

# =============================================================================
# F upper tail on the log scale (pf(..., log.p = TRUE)) where it underflows,
# including large df2
# =============================================================================
.eft <- expand.grid(f = c(50, 400, 1000, 1e8), df1 = c(3, 6), df2 = c(50, 1e3, 1e5, 1e6))
.eft$log_sf <- pf(.eft$f, .eft$df1, .eft$df2, lower.tail = FALSE, log.p = TRUE)
write.csv(.eft, "R_eft_pf.csv", row.names = FALSE)

# =============================================================================
# makeContrasts takes the levels of a factor, not its values
# =============================================================================
.mcl <- makeContrasts(B - A, C - B, levels = factor(c("A", "B", "C", "A"), levels = c("C", "A", "B")))
write.csv(.mcl, "R_mcl_factor.csv")
# no parameter names: unnamed matrix, numeric levels, numeric factor levels
stopifnot(inherits(try(makeContrasts(B - A, levels = matrix(0, 2, 2)), silent = TRUE), "try-error"),
          inherits(try(makeContrasts(B - A, levels = c(1, 2)), silent = TRUE), "try-error"),
          inherits(try(makeContrasts(B - A, levels = factor(c(1, 2))), silent = TRUE), "try-error"))
# duplicated level names: the last copy gets the indicator (pylimma raises)
stopifnot(identical(unname(makeContrasts(A, levels = c("A", "B", "A"))[, 1]), c(0, 0, 1)))

# =============================================================================
# makeContrasts expressions: any syntactically valid level name (dots, Python
# keywords), R number forms, ^ / ** and R operator precedence, IEEE arithmetic
# =============================================================================
.mce_levels <- c("WT.Ctrl", "KO.Ctrl", "not", "True", "lambda", ".a", "if.")
.mce_contrasts <- c("WT.Ctrl-KO.Ctrl", "not-True", "(lambda+.a)/2-if.", "2^-1*WT.Ctrl",
                    "-.a^2+3**2*not", "1e-1*True+.5*KO.Ctrl-2L*if.", "WT.Ctrl - (KO.Ctrl - not)/0x2",
                    "2^3^-1*lambda", "--True", "1/0*True", "not*(-8)^(1/3)", "if. - KO.Ctrl*10^400")
write.csv(makeContrasts(contrasts = .mce_contrasts, levels = .mce_levels), "R_mce_matrix.csv")
# the syntactic-name rule: name == make.names(name)
.mce_names <- c("_x", ".5a", "a b", "if", "TRUE", "..1", ".a", "...", "A_b", "in", "T", "Inf", "NA",
                "NA_real_", "a-b", "1a", "a.1", ".", "A.x", "function", "while.", "x__.")
write.csv(data.frame(name = .mce_names, valid = .mce_names == make.names(.mce_names)),
          "R_mce_names.csv", row.names = FALSE)
# R evaluates the contrast in the R session, so other names resolve there (pylimma
# raises): pi and functions work; an unknown name finds an unrelated object
# ... and ..1 pass make.names() but are reserved for the dots arguments, so R
# cannot evaluate them (pylimma rejects them as level names)
stopifnot(inherits(try(makeContrasts(contrasts = "... - A", levels = c("A", "...")), silent = TRUE),
                   "try-error"),
          inherits(try(makeContrasts(contrasts = "..1 - A", levels = c("A", "..1")), silent = TRUE),
                   "try-error"))
# an expression over several lines parses as several expressions and R keeps the
# last one only (pylimma raises)
stopifnot(identical(unname(makeContrasts(contrasts = "A + B\n- C", levels = c("A", "B", "C"))[, 1]),
                    c(0, 0, -1)))
stopifnot(all.equal(unname(makeContrasts(B - A * pi, levels = c("A", "B"))[, 1]), c(-pi, 1)),
          identical(unname(makeContrasts(mean(c(A, B)), levels = c("A", "B"))[, 1]), c(0.5, 0.5)),
          inherits(try(makeContrasts(B - D, levels = c("A", "B")), silent = TRUE), "try-error"))

# =============================================================================
# makeContrasts(contrasts=): names of a named vector are the column names (an
# empty name falls back to the expression); a single string is one contrast
# =============================================================================
write.csv(makeContrasts(contrasts = c(BvsA = "B-A", "C-B"), levels = c("A", "B", "C")),
          "R_mcc_named.csv")
stopifnot(identical(colnames(makeContrasts(contrasts = "B-A", levels = c("A", "B"))), "B-A"),
          inherits(try(makeContrasts(BvsA = B - A, contrasts = "B-A", levels = c("A", "B")),
                       silent = TRUE), "try-error"),
          # no contrasts: an empty matrix for contrasts= (pylimma raises), an error for ...
          identical(dim(makeContrasts(contrasts = character(0), levels = c("A", "B"))), c(2L, 0L)),
          inherits(try(makeContrasts(levels = c("A", "B")), silent = TRUE), "try-error"))

# =============================================================================
# model.matrix with ^: a power inside I(), crossing of terms outside it (patsy
# writes both as **)
# =============================================================================
.mmp <- data.frame(age = 1:6, g = rep(c("a", "b", "c"), 2), h = rep(c("x", "y"), each = 3))
write.csv(model.matrix(~ I(age^2), .mmp), "R_mmp_power.csv", row.names = FALSE)
write.csv(model.matrix(~ (g + h)^2, .mmp), "R_mmp_crossing.csv", row.names = FALSE)

# =============================================================================
# model.matrix with an ordered factor: contr.poly where contrasts apply
# (patsy: C(o, Poly)); indicator columns without an intercept
# =============================================================================
.mmo <- data.frame(o = factor(rep(c("a", "b", "c", "d"), 2), ordered = TRUE),
                   g = rep(c("x", "y"), each = 4))
write.csv(model.matrix(~ g + o, .mmo), "R_mmo_poly.csv", row.names = FALSE)
write.csv(model.matrix(~ 0 + o, .mmo), "R_mmo_cellmeans.csv", row.names = FALSE)

# =============================================================================
# normalizeVSN (vsn::vsnMatrix): optimiser starting at pstartHeuristic's b = 1
# with lmm = 5; data with many missing values in one array, which runs away to
# a worse optimum from b = 0
# =============================================================================
set.seed(55)
.vso_x <- exp(runif(150, 3, 10) + matrix(rnorm(450, 0, 0.2), 150)) %*% diag(exp(rnorm(3, 0, 0.2))) +
  matrix(rnorm(450, 0, 20), 150)
.vso_x[1:100, 2] <- NA
write.csv(.vso_x, "R_vso_input.csv", row.names = FALSE)
write.csv(normalizeVSN(.vso_x, lts.quantile = 1), "R_vso_ml.csv", row.names = FALSE)
# optimpar cvg.eps stops the LTS iterations early (4 of 7 here)
write.csv(normalizeVSN(.vso_x, optimpar = list(cvg.eps = 1e-2)), "R_vso_cvg.csv", row.names = FALSE)

# =============================================================================
# vsn's LTS row selection (vsn:::vsnLTS, one stratum): rows with a missing
# value have rvar NA and are dropped outside slice 1; slices come from
# cut(rank(hmean), breaks = 5), right-closed, with average ranks for ties
# =============================================================================
set.seed(11)
.vsl_hy <- matrix(rnorm(501 * 4, mean = rep(seq(5, 12, length.out = 501), 4)), 501)
.vsl_hy[sample(length(.vsl_hy), 300)] <- NA
.vsl_hy[2:3, ] <- .vsl_hy[1, ]  # tied row means
.vsl_hy[4, ] <- NA              # all missing: hmean NaN, ranked last
write.csv(.vsl_hy, "R_vsl_hy.csv", row.names = FALSE)
.vsl_hmean <- suppressWarnings(rowMeans(.vsl_hy, na.rm = TRUE))
.vsl_rvar <- rowSums((.vsl_hy - .vsl_hmean)^2)
.vsl_slice <- cut(rank(.vsl_hmean, na.last = TRUE), breaks = 5L)
.vsl_q <- tapply(.vsl_rvar, list(.vsl_slice, rep(1L, 501)), quantile, probs = 0.9, na.rm = TRUE)
.vsl_whsel <- which((.vsl_rvar <= .vsl_q[cbind(as.integer(.vsl_slice), 1L)]) | (as.integer(.vsl_slice) == 1L))
write.csv(data.frame(row = .vsl_whsel), "R_vsl_whsel.csv", row.names = FALSE)
# all ranks equal: cut() puts them in slice 3, so a single row with a missing
# value is dropped
stopifnot(identical(as.integer(cut(rank(5), breaks = 5L)), 3L),
          identical(as.integer(cut(rank(c(2, 2)), breaks = 5L)), c(3L, 3L)))
# end to end with missing values scattered over all arrays
set.seed(12)
.vsl_x <- exp(runif(300, 3, 10) + matrix(rnorm(1500, 0, 0.2), 300)) %*% diag(exp(rnorm(5, 0, 0.2))) +
  matrix(rnorm(1500, 0, 20), 300)
.vsl_x[sample(length(.vsl_x), 300)] <- NA
write.csv(.vsl_x, "R_vsl_input.csv", row.names = FALSE)
write.csv(normalizeVSN(.vsl_x), "R_vsl_output.csv", row.names = FALSE)

# =============================================================================
# normalizeVSN input checks (vsn::vsnMatrix): too few rows unless
# minDataPointsPerStratum is lowered, lts.quantile in [0.5, 1], optimpar
# settings in range, and a likelihood that is not finite (infinite or
# identical arrays)
# =============================================================================
set.seed(13)
.vsv_x <- matrix(exp(rnorm(120, 6)), 30)
write.csv(.vsv_x, "R_vsv_input.csv", row.names = FALSE)
write.csv(normalizeVSN(.vsv_x, minDataPointsPerStratum = 20L), "R_vsv_output.csv", row.names = FALSE)
.vsv_fails <- function(...) inherits(try(suppressWarnings(normalizeVSN(...)), silent = TRUE), "try-error")
.vsv_big <- matrix(exp(rnorm(400, 6)), 100)
.vsv_inf <- .vsv_big
.vsv_inf[1, 1] <- Inf
stopifnot(.vsv_fails(.vsv_x), .vsv_fails(.vsv_big[0, ]),
          .vsv_fails(.vsv_big, lts.quantile = 0.3), .vsv_fails(.vsv_big, lts.quantile = 1.5),
          !.vsv_fails(.vsv_big, lts.quantile = 0.5),
          .vsv_fails(.vsv_big, optimpar = list(factr = -1)),
          .vsv_fails(.vsv_big, optimpar = list(pgtol = NA_real_)),
          .vsv_fails(.vsv_big, optimpar = list(maxit = 0L)),
          .vsv_fails(.vsv_big, optimpar = list(cvg.niter = 0L)),
          .vsv_fails(.vsv_big, optimpar = list(cvg.niter = 2.5)),
          .vsv_fails(.vsv_inf), .vsv_fails(matrix(0, 100, 4)), .vsv_fails(matrix(5, 100, 4)))

# =============================================================================
# normalizeVSN (vsn:::vsnSample) leaves rows that are all NA out of the fit,
# with a warning; an all-NA matrix then fails obscurely (pylimma raises)
# =============================================================================
set.seed(14)
.vsa_x <- exp(runif(120, 3, 10) + matrix(rnorm(480, 0, 0.2), 120)) %*% diag(exp(rnorm(4, 0, 0.2))) +
  matrix(rnorm(480, 0, 20), 120)
.vsa_x[sample(length(.vsa_x), 40)] <- NA
.vsa_x[c(5, 60, 61, 118), ] <- NA
write.csv(.vsa_x, "R_vsa_input.csv", row.names = FALSE)
.vsa_warning <- tryCatch(normalizeVSN(.vsa_x), warning = conditionMessage)
stopifnot(identical(.vsa_warning, "4 rows were removed since they contained only NA elements."))
write.csv(suppressWarnings(normalizeVSN(.vsa_x)), "R_vsa_output.csv", row.names = FALSE)
stopifnot(inherits(try(suppressWarnings(normalizeVSN(matrix(NA_real_, 100, 4))), silent = TRUE),
                   "try-error"))

# =============================================================================
# normalizeBetweenArrays(method = "vsn") points to normalizeVSN
# =============================================================================
stopifnot(identical(tryCatch(normalizeBetweenArrays(matrix(1:8, 4), method = "vsn"),
                             error = conditionMessage),
                    "vsn method no longer supported. Please use normalizeVSN instead."))

# =============================================================================
# stats::lowess, which loessFit, voom and vooma call (pylimma ports it): ties
# that fill the window, constant x, x on a tiny scale, outliers, f > 1,
# iter = 0 and delta = 0. Every window holds at least four points: through
# fewer, a fit is exact and the robustness weights follow last-bit rounding,
# which differs between builds of R (on macOS arm64 R fuses multiply-adds)
# =============================================================================
set.seed(21)
.lw_case <- function(name, x, y, f = 2/3, iter = 3L, delta = NA) {
  fit <- if (is.na(delta)) lowess(x, y, f = f, iter = iter) else
    lowess(x, y, f = f, iter = iter, delta = delta)
  fitted <- numeric(length(x))
  fitted[order(x)] <- fit$y
  data.frame(case = name, f = f, iter = iter, delta = delta, x = x, y = y, fitted = fitted)
}
.lw_x <- runif(200, 4, 14)
.lw_dom <- ifelse(runif(200) < 0.55, 1, sample(2:8, 200, replace = TRUE))
.lw_out <- runif(100)
.lw_yout <- .lw_out + rnorm(100, sd = 0.1)
.lw_yout[seq(5, 100, 10)] <- .lw_yout[seq(5, 100, 10)] + 5
write.csv(rbind(
  .lw_case("smooth", .lw_x, sin(.lw_x) + rnorm(200, sd = 0.3)),
  .lw_case("outliers", .lw_out, .lw_yout, f = 0.1),
  .lw_case("dominant_ties", .lw_dom, 1 / .lw_dom + rnorm(200, sd = 0.3), f = 0.3),
  .lw_case("constant_x", rep(5, 50), rnorm(50)),
  .lw_case("tiny_scale", .lw_x[1:100] * 1e-6, sin(.lw_x[1:100]) + rnorm(100, sd = 0.3), f = 0.3),
  .lw_case("f_above_1", .lw_x[1:60], cos(.lw_x[1:60]) + rnorm(60, sd = 0.2), f = 2),
  .lw_case("iter_0", .lw_x[1:80], cos(.lw_x[1:80]) + rnorm(80, sd = 0.2), f = 0.3, iter = 0L),
  .lw_case("delta_0", .lw_x[1:80], cos(.lw_x[1:80]) + rnorm(80, sd = 0.2), f = 0.3, delta = 0)),
  "R_lowess_cases.csv", row.names = FALSE)
.lw_fails <- function(...) inherits(try(lowess(...), silent = TRUE), "try-error")
# approx(rule = 2, ties = mean), as voom's approxfun and squeezeVar use it:
# NA pairs dropped, ties averaged (an infinite one stays infinite), constant
# beyond the ends; fewer than two x values fail
.lw_ax <- c(3, 1, 1, NA, 2, 2, 5, 4)
.lw_ay <- c(9, 1, 2, 7, 3, Inf, NA, 16)
.lw_aout <- c(0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 6, NA)
write.csv(data.frame(x = c(.lw_ax, rep(NA, 2)), y = c(.lw_ay, rep(NA, 2)), xout = .lw_aout,
                     y_out = approx(.lw_ax, .lw_ay, .lw_aout, rule = 2, ties = mean)$y),
          "R_lowess_approx.csv", row.names = FALSE)
stopifnot(inherits(try(approx(c(1, 1, NA), c(1, 2, 3), 2, rule = 2, ties = mean), silent = TRUE),
                   "try-error"))
stopifnot(.lw_fails(1:10, 1:10, f = 0), .lw_fails(1:10, 1:10, iter = -1L), .lw_fails(1:10, 1:10, delta = -1),
          identical(lowess(.lw_x, .lw_x^2, iter = -0.5), lowess(.lw_x, .lw_x^2, iter = 0)),
          suppressWarnings(.lw_fails(1:10, 1:10, iter = 1e10)))  # NA from asInteger

# Callers of lowess: loessFit (constant x, x * 1e-6, span = 2; iterations = 0
# fails), eBayes(trend, robust) on 12 genes, squeezeVar with a two-valued
# covariate, voom and vooma with small spans (voom span = 0 fails),
# voomaLmFit with an NA sigma, cyclic loess pairs / affy
.lwc_y <- rnorm(100)
.lwc_x <- runif(100, 4, 14)
write.csv(data.frame(y = .lwc_y, x = .lwc_x,
                     constant_x = loessFit(.lwc_y, rep(5, 100))$fitted,
                     tiny_scale = loessFit(.lwc_y, .lwc_x * 1e-6)$fitted,
                     span_2 = loessFit(.lwc_y, .lwc_x, span = 2)$fitted),
          "R_lowess_loessfit.csv", row.names = FALSE)
stopifnot(inherits(try(loessFit(.lwc_y, .lwc_x, iterations = 0L), silent = TRUE), "try-error"))
.lwc_design <- cbind(1, rep(0:1, 6), rep(c(0, 0, 1, 1), 3))
.lwc_e <- matrix(rnorm(12 * 12), 12)
write.csv(.lwc_e, "R_lowess_ebayes_input.csv", row.names = FALSE)
.lwc_eb <- eBayes(lmFit(.lwc_e, .lwc_design), trend = TRUE, robust = TRUE)
write.csv(data.frame(s2_prior = .lwc_eb$s2.prior, df_prior = .lwc_eb$df.prior),
          "R_lowess_ebayes.csv", row.names = FALSE)
.lwc_v <- rchisq(60, 6) / 6 * exp(rnorm(60, 0, 0.3))
.lwc_sv <- squeezeVar(.lwc_v, 5, covariate = rep(0:1, each = 30), robust = TRUE)
write.csv(data.frame(var = .lwc_v, var_prior = .lwc_sv$var.prior, df_prior = .lwc_sv$df.prior,
                     var_post = .lwc_sv$var.post),
          "R_lowess_squeezevar.csv", row.names = FALSE)
.lwc_sim <- function(n_genes, n_samples) {
  list(counts = matrix(rnbinom(n_genes * n_samples, mu = rep(2^seq(2, 12, length.out = n_genes), n_samples),
                               size = 5), n_genes),
       design = cbind(1, rep(0:1, n_samples / 2)))
}
set.seed(4)
.lwc_voom <- .lwc_sim(25, 10)
write.csv(.lwc_voom$counts, "R_lowess_voom_counts.csv", row.names = FALSE)
write.csv(voom(.lwc_voom$counts, .lwc_voom$design, span = 0.2, adaptive.span = FALSE)$weights,
          "R_lowess_voom_weights.csv", row.names = FALSE)
stopifnot(inherits(try(voom(.lwc_voom$counts, .lwc_voom$design, span = 0, adaptive.span = FALSE),
                       silent = TRUE), "try-error"))
set.seed(2)
.lwc_vooma <- .lwc_sim(20, 8)
write.csv(log2(.lwc_vooma$counts + 1), "R_lowess_vooma_input.csv", row.names = FALSE)
write.csv(vooma(log2(.lwc_vooma$counts + 1), .lwc_vooma$design, span = 0.2)$weights,
          "R_lowess_vooma_weights.csv", row.names = FALSE)
# voomaLmFit with a gene that has no residual df: approxfun drops its NA
# point of the trend, so no weight is NA
set.seed(8)
.lwc_na <- matrix(rnorm(200 * 6, 8), 200)
.lwc_na[1, 1:4] <- NA
.lwc_na[2, 3] <- NA
write.csv(.lwc_na, "R_lowess_voomalmfit_input.csv", row.names = FALSE)
.lwc_vlf <- voomaLmFit(.lwc_na, cbind(1, rep(0:1, 3)))
stopifnot(sum(is.na(.lwc_vlf$sigma)) == 1, !anyNA(.lwc_vlf$EList$weights))
write.csv(.lwc_vlf$EList$weights, "R_lowess_voomalmfit_weights.csv", row.names = FALSE)
set.seed(12)
.lwc_cyc <- matrix(rnorm(300 * 4, 8, 2), 300, 4)
.lwc_cyc[, 2] <- .lwc_cyc[, 2] + 1 + 0.1 * .lwc_cyc[, 1]
write.csv(.lwc_cyc, "R_lowess_cyclic_input.csv", row.names = FALSE)
write.csv(normalizeCyclicLoess(.lwc_cyc, method = "pairs"), "R_lowess_cyclic_pairs.csv", row.names = FALSE)
write.csv(normalizeCyclicLoess(.lwc_cyc, method = "affy"), "R_lowess_cyclic_affy.csv", row.names = FALSE)

# =============================================================================
# normalizeCyclicLoess takes each fit from loessFit: a column with one
# observed value in a row is fitted exactly, a zero weight still gets a
# fitted value and a row holding an infinite value turns NA. A missing value
# turns the whole row NA in "pairs" and "affy" (not in "fast"); pylimma gives
# such a row no adjustment from that fit instead, as .cyl_keep does here with
# R's own loessFit
# =============================================================================
set.seed(31)
.cyl_x <- matrix(rnorm(40 * 4, 8, 2), 40)
.cyl_x[, 2] <- .cyl_x[, 2] + 0.5 + 0.1 * .cyl_x[, 1]
.cyl_w <- runif(40)
.cyl_w[c(4, 17)] <- 0
.cyl_na <- .cyl_x
.cyl_na[c(3, 10), 2] <- NA
.cyl_na[22, c(1, 4)] <- NA
.cyl_inf <- .cyl_x
.cyl_inf[5, 3] <- Inf
.cyl_inf[12, 1] <- -Inf
.cyl_keep <- function(x, method, span = 0.7, iterations = 3) {
  n <- ncol(x)
  fit <- function(m, a) {
    f <- loessFit(m, a, span = span)$fitted
    f[is.na(f)] <- 0
    f
  }
  for (k in 1:iterations) {
    adjustment <- matrix(0, nrow(x), n)
    for (i in 1:(n - 1)) for (j in (i + 1):n) {
      f <- fit(x[, j] - x[, i], 0.5 * (x[, j] + x[, i]))
      if (method == "pairs") {
        x[, i] <- x[, i] + f / 2
        x[, j] <- x[, j] - f / 2
      } else {
        adjustment[, j] <- adjustment[, j] + f
        adjustment[, i] <- adjustment[, i] - f
      }
    }
    if (method == "affy") x <- x - adjustment / n
  }
  x
}
stopifnot(all.equal(.cyl_keep(.cyl_x, "pairs"), normalizeCyclicLoess(.cyl_x, method = "pairs")),
          all.equal(.cyl_keep(.cyl_x, "affy"), normalizeCyclicLoess(.cyl_x, method = "affy")),
          all(is.na(normalizeCyclicLoess(.cyl_na, method = "pairs")[c(3, 10, 22), ])),
          all(is.na(normalizeCyclicLoess(.cyl_na, method = "affy")[c(3, 10, 22), ])))
write.csv(.cyl_x, "R_cyl_x.csv", row.names = FALSE)
write.csv(data.frame(w = .cyl_w), "R_cyl_w.csv", row.names = FALSE)
write.csv(.cyl_na, "R_cyl_na.csv", row.names = FALSE)
write.csv(.cyl_inf, "R_cyl_inf.csv", row.names = FALSE)
for (.cyl_m in c("fast", "pairs", "affy")) {
  write.csv(normalizeCyclicLoess(.cyl_x, weights = .cyl_w, method = .cyl_m),
            paste0("R_cyl_weighted_", .cyl_m, ".csv"), row.names = FALSE)
  write.csv(normalizeCyclicLoess(.cyl_inf, method = .cyl_m), paste0("R_cyl_inf_", .cyl_m, ".csv"),
            row.names = FALSE)
}
write.csv(normalizeCyclicLoess(.cyl_na, method = "fast"), "R_cyl_na_fast.csv", row.names = FALSE)
write.csv(.cyl_keep(.cyl_na, "pairs"), "R_cyl_na_pairs.csv", row.names = FALSE)
write.csv(.cyl_keep(.cyl_na, "affy"), "R_cyl_na_affy.csv", row.names = FALSE)
# two genes: in the columns where gene 1 is missing, gene 2 is fitted exactly
.cyl_two <- rbind(rnorm(16, 7.5, 0.3), rnorm(16, 6, 0.3))
.cyl_two[1, c(7, 11)] <- NA
write.csv(.cyl_two, "R_cyl_two.csv", row.names = FALSE)
write.csv(normalizeCyclicLoess(.cyl_two), "R_cyl_two_fast.csv", row.names = FALSE)

# =============================================================================
# normalizeCyclicLoess(adaptive.span = TRUE) chooses its span with
# chooseLowessSpan(nrow(x), small.n = 200, min.span = 0.6)
# =============================================================================
stopifnot(all.equal(normalizeCyclicLoess(.lwc_cyc, adaptive.span = TRUE),
                    normalizeCyclicLoess(.lwc_cyc,
                                         span = chooseLowessSpan(300, small.n = 200, min.span = 0.6))))
write.csv(normalizeCyclicLoess(.lwc_cyc, adaptive.span = TRUE), "R_lowess_cyclic_adaptive.csv", row.names = FALSE)

# =============================================================================
# normalizeCyclicLoess input: iterations = 0 runs two passes (1:0 is c(1, 0);
# pylimma raises) and 2.5 runs two, "pairs" / "affy" on one column, "fast"
# on none and weights with more than one value per row fail, an n x 1
# matrix does not
# =============================================================================
.cyi_fails <- function(...) inherits(try(normalizeCyclicLoess(...), silent = TRUE), "try-error")
stopifnot(isTRUE(all.equal(normalizeCyclicLoess(.cyl_x, iterations = 0),
                           normalizeCyclicLoess(.cyl_x, iterations = 2))),
          .cyi_fails(.cyl_x[, 1, drop = FALSE], method = "pairs"),
          .cyi_fails(.cyl_x[, 1, drop = FALSE], method = "affy"),
          !.cyi_fails(.cyl_x[, 1, drop = FALSE], method = "fast"),
          .cyi_fails(.cyl_x[, 0, drop = FALSE], method = "fast"),
          .cyi_fails(.cyl_x, weights = matrix(1, 40, 4)),
          isTRUE(all.equal(normalizeCyclicLoess(.cyl_x, iterations = 2.5),
                           normalizeCyclicLoess(.cyl_x, iterations = 2))),
          isTRUE(all.equal(normalizeCyclicLoess(.cyl_x, iterations = 0.5),
                           normalizeCyclicLoess(.cyl_x, iterations = 1))),
          isTRUE(all.equal(normalizeCyclicLoess(.cyl_x, weights = matrix(.cyl_w)),
                           normalizeCyclicLoess(.cyl_x, weights = .cyl_w))))

# =============================================================================
# Recording what limma's plot functions draw: .plot_record(fname, ...) runs
# the limma function with its base-graphics calls replaced by recorders and
# returns the calls as a long data frame (call number, function, argument,
# position, value); .also names further functions to record, such as
# plotWithHighlights. Used by the plotting sections below.
# =============================================================================
.plot_record <- function(fname, ..., .also = character()) {
  calls <- list()
  record <- function(what) {
    force(what)
    function(...) {
    # skip empty arguments (legend(..., pch=pch,, col=col) in plotWithHighlights)
    dots <- match.call(expand.dots = FALSE)$...
    given <- !vapply(dots, function(d) identical(d, quote(expr = )), TRUE)
    args <- lapply(dots[given], eval, envir = parent.frame())
    if (is.null(names(args))) names(args) <- rep("", length(args))
    unnamed <- names(args) == ""
    names(args)[unnamed] <- paste0("arg", which(unnamed))
    calls[[length(calls) + 1]] <<- list(what = what, args = args)
    invisible(NULL)
  }}
  f <- get(fname, asNamespace("limma"))
  e <- new.env(parent = environment(f))
  # par() stays real (functions query it, e.g. par("col")), on a null device
  pdf(NULL)
  on.exit(dev.off())
  for (g in c("plot", "points", "lines", "segments", "rect", "text", "legend", "abline", "axis",
              "mtext", "title", "polygon", "image", "layout", "barplot", "matplot", "box",
              .also))
    assign(g, record(g), e)
  environment(f) <- e
  f(...)
  rows <- list()
  for (k in seq_along(calls)) for (a in names(calls[[k]]$args)) {
    v <- calls[[k]]$args[[a]]
    if (is.function(v) || is.language(v) || is.list(v) || is.null(v)) next
    # an empty vector (e.g. points() for a status value with no points) is one
    # row with pos 0
    rows[[length(rows) + 1]] <- data.frame(call = k, fun = calls[[k]]$what, arg = a,
                                           pos = if (length(v)) seq_along(v) else 0L,
                                           value = if (length(v)) as.character(v) else NA)
  }
  do.call(rbind, rows)
}

# =============================================================================
# plotWithHighlights: as.character(status) keeps NA (background, not a group
# of its own), values come from table(status) by decreasing frequency (ties in
# level order: numeric for numbers) and a float status matches values = 1
# =============================================================================
.pwh_x <- seq_len(160)
.pwh_y <- sqrt(.pwh_x)
.pwh_cases <- list(
  character_na = list(status = c(rep("a", 100), rep(NA, 50), rep("b", 10))),
  na_most_frequent = list(status = c(rep("a", 10), rep(NA, 50), rep("b", 5), rep("c", 95))),
  float_values = list(status = c(rep(1, 100), rep(0, 50), rep(-1, 10)), values = c(1, -1)),
  numeric_ties = list(status = c(rep(0, 100), rep(10, 30), rep(2, 30))),
  logical = list(status = c(rep(TRUE, 40), rep(FALSE, 120))))
write.csv(do.call(rbind, lapply(names(.pwh_cases), function(case) {
  args <- c(list("plotWithHighlights", .pwh_x, .pwh_y, legend = FALSE), .pwh_cases[[case]])
  rec <- do.call(.plot_record, args)
  data.frame(case = case, rec[rec$fun == "points" & rec$arg == "arg1", ])
})), "R_pwh_status_points.csv", row.names = FALSE)

# =============================================================================
# plotWithHighlights legend: the background value comes first ("Other" when
# the background holds several values, NA included), then every highlighted
# value, also one with no points
# =============================================================================
.pwl_cases <- list(
  single_background = list(status = c(rep("notDE", 150), rep("up", 6), rep("down", 4))),
  other_background = list(status = c(rep("a", 100), rep("b", 50), rep("c", 10)), values = "c"),
  na_background = list(status = c(rep("a", 100), rep(NA, 50), rep("c", 10)), values = "c"),
  empty_value = list(status = c(rep("a", 150), rep("c", 10)), values = c("c", "z")))
write.csv(do.call(rbind, lapply(names(.pwl_cases), function(case) {
  args <- c(list("plotWithHighlights", seq_len(160), sqrt(seq_len(160))), .pwl_cases[[case]])
  rec <- do.call(.plot_record, args)
  data.frame(case = case, rec[rec$fun == "legend" & rec$arg == "legend", c("pos", "value")])
})), "R_pwh_legend.csv", row.names = FALSE)

# =============================================================================
# plotMD / plotMA defaults: for an MArrayLM the last coefficient, its name as
# title and genes$Status; plotMD takes the matching column of a TestResults
# status; for an EList the column name as title and genes$Status
# =============================================================================
set.seed(41)
.pmd_y <- matrix(rnorm(60 * 6, 8), 60, dimnames = list(NULL, paste0("s", 1:6)))
.pmd_y[1:6, 4:6] <- .pmd_y[1:6, 4:6] + 3
.pmd_design <- cbind(Intercept = 1, B = rep(0:1, each = 3), C = rep(c(0, 1, 0), 2))
.pmd_genes <- data.frame(ID = paste0("g", 1:60), Status = rep(c("gene", "control"), c(54, 6)))
.pmd_fit <- eBayes(lmFit(.pmd_y, .pmd_design))
.pmd_fit$genes <- .pmd_genes
.pmd_el <- new("EList", list(E = .pmd_y, genes = .pmd_genes))
.pmd_args <- function(case, fname, ...) {
  rec <- .plot_record(fname, ..., .also = "plotWithHighlights")
  rec <- rec[rec$fun == "plotWithHighlights" & rec$arg %in% c("x", "y", "main", "ylab", "status"), ]
  data.frame(case = case, rec[, c("arg", "pos", "value")])
}
write.csv(rbind(
  .pmd_args("md_fit", "plotMD.MArrayLM", .pmd_fit),
  .pmd_args("ma_fit", "plotMA.MArrayLM", .pmd_fit),
  .pmd_args("md_fit_coef", "plotMD.MArrayLM", .pmd_fit, coef = 2),
  .pmd_args("md_fit_tests", "plotMD.MArrayLM", .pmd_fit, status = decideTests(.pmd_fit)),
  .pmd_args("md_elist", "plotMD.EList", .pmd_el, column = 2)),
  "R_pmd_defaults.csv", row.names = FALSE)
write.csv(.pmd_y, "R_pmd_y.csv", row.names = FALSE)

# =============================================================================
# volcanoplot: highlight labels default to fit$genes$ID
# =============================================================================
.vp_rec <- .plot_record("volcanoplot", .pmd_fit, coef = 2, highlight = 4)
write.csv(.vp_rec[.vp_rec$fun == "text", c("arg", "pos", "value")], "R_volcano_labels.csv",
          row.names = FALSE)

# =============================================================================
# vennCounts drops rows with NA (and names columns "Group 1", ...);
# vennDiagram puts counts[k] (table order) at fixed positions, the set-2-only
# count in the second circle, and accepts a scalar cex
# =============================================================================
.vn_x <- cbind(c(1, 0, NA, 1, -1, 0, 1), c(1, 1, 0, 0, NA, -1, 1))
write.csv(vennCounts(.vn_x), "R_venn_counts_na.csv", row.names = FALSE)
write.csv(vennCounts(.vn_x, include = "up"), "R_venn_counts_na_up.csv", row.names = FALSE)
set.seed(51)
.vn_tests <- matrix(sample(c(-1, 0, 1), 3 * 40, replace = TRUE, prob = c(0.2, 0.5, 0.3)), 40,
                    dimnames = list(NULL, c("A", "B", "C")))
.vn_text <- function(case, ...) {
  rec <- .plot_record("vennDiagram", ...)
  text_calls <- rec[rec$fun == "text" & rec$arg %in% c("arg1", "arg2", "arg3"), ]
  data.frame(case = case, text_calls[, c("call", "arg", "value")])
}
write.csv(rbind(
  .vn_text("one", .vn_tests[, 1, drop = FALSE]),
  .vn_text("two", .vn_tests[, 1:2]),
  .vn_text("three", .vn_tests),
  .vn_text("two_up_down", .vn_tests[, 1:2], include = c("up", "down")),
  .vn_text("two_scalar_cex", .vn_tests[, 1:2], cex = 1)),
  "R_venn_text.csv", row.names = FALSE)
write.csv(.vn_tests, "R_venn_tests.csv", row.names = FALSE)

# =============================================================================
# heatDiagram: names are numbered over all rows, then the DE rows taken; rows
# are ordered (stably) on the untruncated primary coefficient, then clipped
# =============================================================================
.hd_results <- cbind(c(0, 1, 0, 1, -1, 1, 1), c(1, 0, 1, 1, 0, 0, 1))
.hd_coef <- cbind(c(0.2, 3, 0.1, 2, -2, 2, 1.5), c(1, 0.5, -1, 1.2, 0.3, 0.2, 2))
write.csv(data.frame(results = .hd_results, coef = .hd_coef), "R_hd_input.csv", row.names = FALSE)
.hd_out <- function(case, ...) {
  pdf(NULL)
  on.exit(dev.off())
  out <- heatDiagram(.hd_results, .hd_coef, ...)
  data.frame(case = case, name = rownames(out), out)
}
write.csv(rbind(.hd_out("default"), .hd_out("limit", limit = 1.6),
                .hd_out("names", names = paste0("gene", 1:7))),
          "R_hd_output.csv", row.names = FALSE)
# names go through as.character; names shorter than the rows give NA names
# (pylimma raises)
stopifnot(identical(.hd_out("numeric", names = as.numeric(1:7))$name, .hd_out("default")$name),
          anyNA(.hd_out("short", names = c(1.5, 2, 3))$name))

# =============================================================================
# plotMA3by2 on a matrix: each array's M = E - rowMeans(E) against
# A = rowMeans(E), common limits over all A and M, the legend on the first
# plot of a page (plotMA's arguments recorded, files not written)
# =============================================================================
set.seed(61)
.p32_E <- matrix(rnorm(20 * 8, 8, 1.5), 20, dimnames = list(NULL, paste0("a", 1:8)))
write.csv(.p32_E, "R_p32_E.csv", row.names = FALSE)
.p32_calls <- list()
.p32_plotMA <- function(object, array, ...) {
  args <- list(...)
  .p32_calls[[length(.p32_calls) + 1]] <<- data.frame(
    array = array, gene = seq_len(nrow(object$M)), A = object$A[, array], M = object$M[, array],
    xlim1 = args$xlim[1], xlim2 = args$xlim[2], ylim1 = args$ylim[1], ylim2 = args$ylim[2],
    legend = args$legend, main = args$main)
}
.p32 <- plotMA3by2
environment(.p32) <- list2env(list(plotMA = .p32_plotMA, png = function(...) NULL,
                                   dev.off = function(...) NULL, par = function(...) NULL),
                              parent = asNamespace("limma"))
.p32(.p32_E)
write.csv(do.call(rbind, .p32_calls), "R_p32_calls.csv", row.names = FALSE)

# =============================================================================
# plotDensities: group levels sorted (factor), each array coloured by its
# level, colours recycled over the arrays when group is shorter, legend in
# level order (default col = 1:ngroup, so col values index the levels)
# =============================================================================
set.seed(71)
.pd_E <- matrix(rnorm(50 * 4, 8), 50, dimnames = list(NULL, c("s1", "s2", "s3", "s4")))
write.csv(.pd_E, "R_pd_E.csv", row.names = FALSE)
.pd_rec <- function(case, ...) {
  rec <- .plot_record("plotDensities.default", .pd_E, ...)
  keep <- (rec$fun == "matplot" & rec$arg == "col") | (rec$fun == "legend" & rec$arg %in% c("legend", "col"))
  data.frame(case = case, rec[keep, c("fun", "arg", "pos", "value")])
}
write.csv(rbind(.pd_rec("character", group = c("b", "a", "b", "a")),
                .pd_rec("short", group = c(1, 2)),
                .pd_rec("numeric", group = c(10, 2, 2, 10))),
          "R_pd_groups.csv", row.names = FALSE)

# =============================================================================
# plotlines recycles col over the rows (col <- rep(col, ngenes))
# =============================================================================
.pl_rec <- .plot_record("plotlines", matrix(seq_len(20), 5), col = c("red", "blue"))
write.csv(.pl_rec[.pl_rec$fun == "lines" & .pl_rec$arg == "col", c("call", "value")],
          "R_plotlines_col.csv", row.names = FALSE)

# =============================================================================
# plotMDS(plot = FALSE) returns the MDS object, x and y for dim.plot; the MDS
# object can be plotted again (plotMDS.MDS); dim.plot must name two
# dimensions
# =============================================================================
set.seed(81)
.mds_x <- matrix(rnorm(100 * 6), 100)
.mds_x[1:20, 4:6] <- .mds_x[1:20, 4:6] + 2
write.csv(.mds_x, "R_mds_x.csv", row.names = FALSE)
.mds <- plotMDS(.mds_x, plot = FALSE, dim.plot = c(1, 3))
pdf(NULL)
.mds_re <- plotMDS(.mds, dim.plot = c(2, 3))
invisible(dev.off())
write.csv(data.frame(x = .mds$x, y = .mds$y, re_x = .mds_re$x, re_y = .mds_re$y),
          "R_mds_dims.csv", row.names = FALSE)
.mds_fails <- function(...) inherits(try(plotMDS(.mds_x, plot = FALSE, ...), silent = TRUE), "try-error")
stopifnot(.mds_fails(dim.plot = c(1, 1)), .mds_fails(dim.plot = 1), .mds_fails(top = 0))

# =============================================================================
# Plot argument errors: volcanoplot(list(x = 1)) stops with "fit must be an
# MArrayLM"; mdplot(columns = 1) and vennDiagram(NULL) fail obscurely
# ("need finite 'xlim' values", "'data' must be of a vector type"); pylimma
# raises clear errors for the last two
# =============================================================================
pdf(NULL)
.pae <- function(expr) tryCatch({ suppressWarnings(expr); "" }, error = conditionMessage)
stopifnot(identical(.pae(volcanoplot(list(x = 1))), "fit must be an MArrayLM"),
          nzchar(.pae(mdplot(matrix(rnorm(20), 10), columns = 1))),
          nzchar(.pae(vennDiagram(NULL))))
invisible(dev.off())

# =============================================================================
# plotRLDF: R's argument checks; a design column on a tiny scale is still
# estimable (qr(tol = 1e-7) is relative); with two groups R pads svd(nu = 2)
# with an arbitrary second dimension (pylimma raises)
# =============================================================================
set.seed(91)
.rl_y <- matrix(rnorm(150 * 12), 150)
.rl_y[1:15, 5:8] <- .rl_y[1:15, 5:8] + 2
.rl_y[16:30, 9:12] <- .rl_y[16:30, 9:12] + 2
write.csv(.rl_y, "R_rl_y.csv", row.names = FALSE)
.rl_design <- cbind(1, rep(c(0, 1, 0), each = 4), rep(c(0, 0, 1), each = 4))
.rl_tiny <- .rl_design
.rl_tiny[, 3] <- .rl_tiny[, 3] * 1e-11
.rl_out <- plotRLDF(.rl_y, .rl_tiny, plot = FALSE)
.rl_sign <- function(m) sweep(m, 2, sign(m[cbind(apply(abs(m), 2, which.max), seq_len(ncol(m)))]), "*")
write.csv(.rl_sign(.rl_out$training), "R_rl_tiny_training.csv", row.names = FALSE)
write.csv(data.frame(d = .rl_out$singular.values), "R_rl_tiny_singular_values.csv", row.names = FALSE)
.rl_fails <- function(...) inherits(try(plotRLDF(.rl_y, plot = FALSE, ...), silent = TRUE), "try-error")
stopifnot(.rl_fails(design = .rl_design, labels.y = c("a", "b")),
          .rl_fails(design = .rl_design, z = .rl_y[1:10, ]),
          .rl_fails(design = .rl_design, show.dimensions = c(1, 1)),
          .rl_fails(design = .rl_design, var.prior = c(1, 2), df.prior = 3),
          ncol(plotRLDF(.rl_y, cbind(1, rep(0:1, each = 6)), plot = FALSE)$training) == 2)

# =============================================================================
# plotExons: the rank-th gene among distinct genes, exons ordered and
# labelled by exoncolname, DE exons red / blue sized by FDR, dashed line at
# the mean logFC, title from the annotation columns and strand
# =============================================================================
set.seed(101)
.pe_genes <- data.frame(GeneID = rep(paste0("G", 1:6), each = 4), Exon = rep(c(9, 3, 7, 1), 6),
                        Symbol = rep(paste0("sym", 1:6), each = 4), Strand = "+")
.pe_y <- matrix(rnorm(24 * 6), 24)
.pe_y[5:8, 4:6] <- .pe_y[5:8, 4:6] + c(3, -2.5, 0.2, 4)
.pe_y[1:4, 4:6] <- .pe_y[1:4, 4:6] + c(5, 4, 3.5, 4.5)
write.csv(.pe_y, "R_pe_y.csv", row.names = FALSE)
write.csv(.pe_genes, "R_pe_genes.csv", row.names = FALSE)
.pe_fit <- eBayes(lmFit(.pe_y, cbind(1, rep(0:1, each = 3))))
.pe_fit$genes <- .pe_genes
.pe_rec <- function(case, ...) {
  rec <- .plot_record("plotExons", .pe_fit, ...)
  keep <- (rec$fun == "plot" & rec$arg %in% c("arg1", "main")) |
    (rec$fun == "axis" & rec$arg == "labels") |
    (rec$fun == "points" & rec$arg %in% c("arg1", "arg2", "col", "cex")) |
    (rec$fun == "abline" & rec$arg == "h") | (rec$fun == "mtext" & rec$arg == "text")
  data.frame(case = case, rec[keep, c("fun", "arg", "pos", "value")])
}
write.csv(rbind(.pe_rec("rank1"), .pe_rec("rank2_exon", rank = 2, exoncolname = "Exon"),
                .pe_rec("geneid", geneid = "G3", exoncolname = "Exon")),
          "R_pe_calls.csv", row.names = FALSE)

# =============================================================================
# coolmap (through gplots::heatmap.2): rowInd / colInd, the row order from
# the bottom up (nr:1 without a row dendrogram) and the column order (1:nc
# without one); the column dendrogram uses the unscaled matrix
# =============================================================================
set.seed(111)
.cm_x <- matrix(rnorm(30 * 6, 8, 1), 30, dimnames = list(sprintf("g%02d", 1:30), paste0("s", 1:6)))
.cm_x[1:8, 4:6] <- .cm_x[1:8, 4:6] + 2
.cm_x[9:14, c(1, 4)] <- .cm_x[9:14, c(1, 4)] - 1.5
write.csv(.cm_x, "R_cm_x.csv")
.cm_ind <- function(case, x = .cm_x, ...) {
  pdf(NULL)
  on.exit(dev.off())
  out <- coolmap(x, ...)
  rbind(data.frame(case = case, which = "row", pos = seq_along(out$rowInd), ind = out$rowInd),
        data.frame(case = case, which = "col", pos = seq_along(out$colInd), ind = out$colInd))
}
write.csv(rbind(
  .cm_ind("de_pattern"),
  .cm_ind("expression", cluster.by = "expression level"),
  .cm_ind("rows_none", cluster.by = "expression level", linkage.row = "none"),
  .cm_ind("cols_none", cluster.by = "expression level", linkage.col = "none")),
  "R_cm_ind.csv", row.names = FALSE)

# coolmap with each linkage method: R's ward.D, centroid and median update
# the distances as given (scipy squares them)
write.csv(do.call(rbind, lapply(c("single", "complete", "average", "mcquitty", "ward.D", "ward.D2",
                                  "centroid", "median"), function(m)
  .cm_ind(m, linkage.row = m, linkage.col = m))), "R_cm_linkage.csv", row.names = FALSE)

# coolmap with missing values: dist() sums over the coordinates both rows
# have and scales by ncol / count
.cm_na <- .cm_x
.cm_na[2, 1] <- NA
.cm_na[c(5, 17), 4] <- NA
.cm_na[9, 2:3] <- NA
write.csv(.cm_na, "R_cm_na.csv")
write.csv(rbind(.cm_ind("na_de_pattern", .cm_na), .cm_ind("na_expression", .cm_na, cluster.by = "expression level")),
          "R_cm_na_ind.csv", row.names = FALSE)

# coolmap takes as.matrix() of an EList (its E) and fails for a single row
.cm_el <- new("EList", list(E = .cm_x))
stopifnot(identical(.cm_ind("el", .cm_el)$ind, .cm_ind("x", .cm_x)$ind),
          inherits(try(.cm_ind("one", .cm_x[1, , drop = FALSE]), silent = TRUE), "try-error"))

# =============================================================================
# barcodeplot: index positions in the order given, gene.weights following
# that order (an unsorted index with unequal weights); the worm is recorded
# =============================================================================
set.seed(121)
.bc_stat <- rnorm(200)
.bc_index <- sample(200, 30)
.bc_stat[.bc_index[1:15]] <- .bc_stat[.bc_index[1:15]] + 1.5
.bc_weights <- runif(30, 0.5, 3)
write.csv(data.frame(stat = .bc_stat), "R_bc_stat.csv", row.names = FALSE)
write.csv(data.frame(index = .bc_index, weight = .bc_weights), "R_bc_index.csv", row.names = FALSE)
.bc_rec <- .plot_record("barcodeplot", .bc_stat, index = .bc_index, gene.weights = .bc_weights)
write.csv(.bc_rec[.bc_rec$fun == "lines" & .bc_rec$arg == "y", c("pos", "value")], "R_bc_worm_weighted.csv",
          row.names = FALSE)

# barcodeplot's drawing: shaded regions (rect), bars (segments), worms
# (lines), their reference lines (abline) and axis labels, for one set, two
# sets, positive / mixed / negative weights and without the worm
.bc_two <- sample(setdiff(seq_len(200), .bc_index), 25)
.bc_stat[.bc_two[1:12]] <- .bc_stat[.bc_two[1:12]] - 1.5
write.csv(data.frame(stat = .bc_stat), "R_bc_stat2.csv", row.names = FALSE)
write.csv(data.frame(index2 = .bc_two), "R_bc_index2.csv", row.names = FALSE)
.bc_mixed <- .bc_weights * rep(c(1, -1), 15)
.bc_draw <- function(case, ...) {
  rec <- .plot_record("barcodeplot", .bc_stat, ...)
  keep <- rec$fun %in% c("rect", "segments", "lines", "abline", "axis") &
    rec$arg %in% c("arg1", "arg2", "arg3", "arg4", "col", "lwd", "x", "y", "h", "side", "at", "labels", "tick")
  data.frame(case = case, rec[keep, ])
}
write.csv(rbind(
  .bc_draw("one", index = .bc_index),
  .bc_draw("two", index = .bc_index, index2 = .bc_two),
  .bc_draw("weighted", index = .bc_index, gene.weights = .bc_weights),
  .bc_draw("mixed", index = .bc_index, gene.weights = .bc_mixed),
  .bc_draw("negative", index = .bc_index, gene.weights = -.bc_weights),
  .bc_draw("no_worm", index = .bc_index, worm = FALSE)),
  "R_bc_draw.csv", row.names = FALSE)

# =============================================================================
# propTrueNull(method="hist"): cut() bins are (left, right], so a p-value on a
# bin edge counts in the bin below it
# =============================================================================
set.seed(1211)
.ptn_p <- c(round(runif(150), 2), (1:20) / 20, 0)
write.csv(data.frame(p = .ptn_p), "R_ptn_edges_p.csv", row.names = FALSE)
write.csv(data.frame(pi0 = propTrueNull(.ptn_p, method = "hist")), "R_ptn_edges.csv", row.names = FALSE)

# =============================================================================
# qqt / qqf: theoretical quantiles at ppoints(n) (a = 3/8 for n <= 10), in the
# order of the data (ties keep their order); qf handles infinite df
# =============================================================================
.qq_y <- c(2.1, -0.4, 0.7, 0.7, 1.5, -1.2, 3.3, 0.2, -0.9, 1.1, 0.5)
write.csv(data.frame(y = .qq_y), "R_qq_y.csv", row.names = FALSE)
.qq_case <- function(case, fun, n, ...) data.frame(case = case, x = fun(.qq_y[1:n], plot.it = FALSE, ...)$x)
write.csv(rbind(
  .qq_case("t3_n5", qqt, 5, df = 3),
  .qq_case("t3_n10", qqt, 10, df = 3),
  .qq_case("t3_n11", qqt, 11, df = 3),
  .qq_case("tinf_n11", qqt, 11),
  .qq_case("f_3_10_n5", qqf, 5, df1 = 3, df2 = 10),
  .qq_case("f_3_inf_n11", qqf, 11, df1 = 3, df2 = Inf),
  .qq_case("f_inf_5_n5", qqf, 5, df1 = Inf, df2 = 5),
  .qq_case("f_inf_inf_n5", qqf, 5, df1 = Inf, df2 = Inf),
  .qq_case("f_5_1e6_n11", qqf, 11, df1 = 5, df2 = 1e6),
  .qq_case("f_5e5_10_n5", qqf, 5, df1 = 5e5, df2 = 10),
  .qq_case("f_vector_df1_n5", qqf, 5, df1 = 2:6, df2 = 10)),
  "R_qq_x.csv", row.names = FALSE)

# =============================================================================
# zscoreGamma in the far tails: pgamma(log.p=TRUE) stays finite where the
# tail probability underflows
# =============================================================================
.zg_q <- c(0.5, 2, 100, 1000, 5000, 1e-200, 1e-320)
write.csv(data.frame(q = .zg_q,
                     z_shape2 = zscoreGamma(.zg_q, shape = 2, rate = 1.5),
                     z_shape50 = zscoreGamma(.zg_q, shape = 50, scale = 0.2)),
          "R_zscore_gamma_tails.csv", row.names = FALSE)
stopifnot(is.nan(suppressWarnings(zscoreGamma(5, shape = -0.5))))

# =============================================================================
# Subsetting limma objects: head/tail take a negative n as "all but the last
# (first) -n" and truncate a fractional n; TRUE selects every row; a name
# not among the row or column names stops (subsetListOfArrays)
# =============================================================================
.sub_el <- new("EList", list(E = matrix(as.numeric(0:8), 3, 3)))
.sub_fit <- lmFit(matrix(rnorm(60), 10), cbind(Intercept = 1, x = c(0, 0, 0, 1, 1, 1)))
stopifnot(nrow(head(.sub_el, -1)) == 2, nrow(tail(.sub_el, -1)) == 2,
          nrow(tail(.sub_el, 2.5)) == 3, nrow(tail(.sub_el, -1.5)) == 2,
          identical(tail(.sub_el, -1)$E, .sub_el$E[2:3, ]),
          nrow(head(.sub_el, 2.5)) == 2, nrow(head(.sub_el, -5)) == 0,
          nrow(.sub_el[TRUE, ]) == 3,
          inherits(try(.sub_fit[, "z"], silent = TRUE), "try-error"),
          grepl("Subscript not found in colnames", try(.sub_fit[, "z"], silent = TRUE)),
          grepl("Subscript not found in rownames", try(.sub_el["a", ], silent = TRUE)))

# =============================================================================
# A fit keeps the expression matrix's row names, so genes can be selected by
# name
# =============================================================================
.rn_y <- matrix(rnorm(48), 8, dimnames = list(paste0("g", 1:8), NULL))
.rn_fit <- eBayes(lmFit(.rn_y, cbind(1, c(0, 0, 0, 1, 1, 1))))
stopifnot(identical(rownames(.rn_fit[c("g3", "g1"), ]), c("g3", "g1")),
          identical(.rn_fit[c("g3", "g1"), ]$t, .rn_fit$t[c(3, 1), , drop = FALSE]))

# =============================================================================
# poolVar: Satterthwaite pooling; the df is var^2 / sum(m^2 s2^2 / df)
# divided directly (0/0 is NaN, an NA variance gives NA, infinite df Inf)
# =============================================================================
.pv_case <- function(case, ...) {
  out <- poolVar(...)
  data.frame(case = case, var = out$var, df = out$df, multiplier = out$multiplier)
}
write.csv(rbind(
  .pv_case("n", var = c(1.2, 0.7, 2.5), n = 3:5),
  .pv_case("df_multiplier", var = c(1.2, 0.7), df = c(4, 9), multiplier = c(0.5, 2)),
  .pv_case("zero_var", var = c(0, 0, 0), n = 3:5),
  .pv_case("na_var", var = c(1, NA, 3), n = 3:5),
  .pv_case("inf_df", var = c(1, 2), df = c(Inf, Inf), multiplier = c(1, 1))),
  "R_pool_var.csv", row.names = FALSE)
# a matrix of variances is read as its values (as.vector)
stopifnot(identical(poolVar(matrix(c(1.2, 0.7, 2.5, 1.1), 2), df = 3:6, multiplier = c(1, 2, 1, 2)),
                    poolVar(c(1.2, 0.7, 2.5, 1.1), df = 3:6, multiplier = c(1, 2, 1, 2))))

# =============================================================================
# auROC: ties averaged, stat=NULL, matrices read as vectors (column-major),
# logical and character stat ordered by order(); truth through as.logical
# (a string that is not a logical stops obscurely; pylimma raises)
# =============================================================================
.ar_truth <- c(1, 0, 1, 1, 0, 0, 1, 0)
.ar_stat <- c(2.5, 0.3, 1.1, 1.1, 0.9, 1.1, -0.2, 0.0)
write.csv(data.frame(truth = .ar_truth, stat = .ar_stat), "R_auroc_input.csv", row.names = FALSE)
write.csv(data.frame(
  case = c("stat", "no_stat", "matrix", "logical_stat", "character_stat", "character_truth"),
  auc = c(auROC(.ar_truth, .ar_stat), auROC(.ar_truth),
          auROC(matrix(.ar_truth, 4), matrix(.ar_stat, 4)),
          auROC(.ar_truth, .ar_stat > 1),
          auROC(.ar_truth, c("d", "a", "c", "c", "b", "c", "a", "b")),
          auROC(c("TRUE", "F", "T", "true", "FALSE", "false", "True", "F"), .ar_stat))),
  "R_auroc.csv", row.names = FALSE)
stopifnot(inherits(try(auROC(c("a", "b", "a"), 1:3), silent = TRUE), "try-error"))

# =============================================================================
# duplicateCorrelation input forms: a character block (factor(block)) and a
# design given as a vector (as.matrix) give the same correlation as their
# numeric / matrix forms
# =============================================================================
set.seed(1213)
.dci_y <- matrix(rnorm(20 * 12), 20, 12)
.dci_y[, 1:6] <- .dci_y[, 1:6] + rnorm(20)
.dci_block <- rep(c("b", "a", "c"), each = 4)
.dci_x <- rep(c(0, 1), 6)
write.csv(.dci_y, "R_dci_y.csv", row.names = FALSE)
.dci_cor <- duplicateCorrelation(.dci_y, design = cbind(1, .dci_x), block = .dci_block)$consensus.correlation
stopifnot(grepl("2 or more levels", try(duplicateCorrelation(.dci_y, block = rep(1, 12)), silent = TRUE)),
          suppressWarnings(duplicateCorrelation(.dci_y, diag(12), block = .dci_block)$consensus.correlation) == 0,
          all.equal(duplicateCorrelation(.dci_y, design = cbind(1, .dci_x), block = match(.dci_block, c("b", "a", "c")))$consensus.correlation, .dci_cor),
          all.equal(duplicateCorrelation(.dci_y, design = rep(1, 12), block = .dci_block)$consensus.correlation,
                    duplicateCorrelation(.dci_y, design = matrix(1, 12, 1), block = .dci_block)$consensus.correlation))
write.csv(data.frame(
  case = c("character_block", "vector_design"),
  cor = c(.dci_cor, duplicateCorrelation(.dci_y, design = rep(1, 12), block = .dci_block)$consensus.correlation)),
  "R_dci.csv", row.names = FALSE)

# =============================================================================
# duplicateCorrelation with a design column on a tiny scale: qr() and
# statmod's mixedModel2Fit (lm.fit) decide rank relative to each column
# (tol 1e-7), so the column is kept
# =============================================================================
set.seed(1214)
.dct_y <- matrix(rnorm(60), 10, 6)
.dct_block <- rep(1:3, each = 2)
write.csv(.dct_y, "R_dct_y.csv", row.names = FALSE)
write.csv(data.frame(
  case = c("trend", "block_column"),
  cor = c(duplicateCorrelation(.dct_y, cbind(1, 1e-12 * (1:6)), block = .dct_block)$consensus.correlation,
          duplicateCorrelation(.dct_y, cbind(1, 1e-12 * (.dct_block == 2)), block = .dct_block)$consensus.correlation)),
  "R_dct.csv", row.names = FALSE)

# =============================================================================
# duplicateCorrelation with no gene that can be fitted (too few observations):
# the consensus is mean(numeric(0)), NaN
# =============================================================================
.dcn <- suppressWarnings(duplicateCorrelation(matrix(c(1, 3, 2, 5), 1), cbind(1, c(0, 1, 0, 1)), block = c(1, 1, 2, 2)))
stopifnot(is.nan(.dcn$consensus.correlation), is.nan(.dcn$cor), all(is.na(.dcn$atanh.correlations)))

# =============================================================================
# avereps on an EList averages E, every matrix in other and the weights (a
# vector as one column, as.matrix)
# =============================================================================
.ar_X <- matrix(as.numeric(0:35), 12, 3, byrow = TRUE)
.ar_el <- avereps(new("EList", list(E = .ar_X, other = list(x = .ar_X * 2), weights = as.numeric(0:11),
                                    genes = data.frame(ID = rep(letters[1:6], each = 2)))))
write.csv(data.frame(E = .ar_el$E, other = .ar_el$other$x, weights = .ar_el$weights),
          "R_avereps_elist.csv", row.names = FALSE)
stopifnot(grepl("incorrect length for 'group'",
                try(avereps(new("EList", list(E = .ar_X, weights = c(1, 2, 3), genes = data.frame(ID = rep(letters[1:6], each = 2))))),
                    silent = TRUE)))

# =============================================================================
# contrasts.fit argument forms: coefficients = 1 (a double) subsets; a
# cov.coefficients of the wrong size fails in chol() %*% (pylimma raises a
# clear error); a NULL fit has no coefficients
# =============================================================================
.cf_fit <- lmFit(matrix(rnorm(60), 10), cbind(1, c(0, 0, 0, 1, 1, 1)))
.cf_bad <- .cf_fit
.cf_bad$cov.coefficients <- diag(3)
stopifnot(identical(dim(contrasts.fit(.cf_fit, coefficients = 1)$coefficients), c(10L, 1L)),
          grepl("non-conformable", try(contrasts.fit(.cf_bad, cbind(c(0, 1))), silent = TRUE)),
          grepl("fit must contain coefficients component", try(contrasts.fit(NULL, cbind(c(0, 1))), silent = TRUE)))

# =============================================================================
# topTable coef forms: a double subscript is truncated (coef = 2.0 is column
# 2); an empty coef gives an empty data frame (pylimma raises)
# =============================================================================
.tt_fit <- eBayes(lmFit(matrix(rnorm(60), 10), cbind(1, c(0, 0, 0, 1, 1, 1))))
stopifnot(identical(topTable(.tt_fit, coef = 2.0), topTable(.tt_fit, coef = 2L)),
          inherits(try(suppressWarnings(topTable(.tt_fit, coef = TRUE)), silent = TRUE), "try-error"),
          identical(dim(topTable(.tt_fit, coef = integer(0))), c(0L, 0L)))

# =============================================================================
# eBayes / treat argument checks: a fit with no genes fails "No finite
# residual standard deviations" (max(numeric(0)) is -Inf, so the df check
# passes); treat with fc = NULL and no lfc fails in log2()
# =============================================================================
.eb_fit <- lmFit(matrix(rnorm(60), 10), cbind(1, c(0, 0, 0, 1, 1, 1)))
stopifnot(grepl("No finite residual standard deviations", try(suppressWarnings(eBayes(.eb_fit[0, ])), silent = TRUE)),
          grepl("No finite residual standard deviations", try(suppressWarnings(treat(.eb_fit[0, ])), silent = TRUE)),
          grepl("non-numeric argument", try(treat(.eb_fit, fc = NULL), silent = TRUE)))

# =============================================================================
# squeezeVar input forms: a matrix of variances is its values (length(var));
# NULL is empty; a df of the wrong length is recycled into var[df == 0] and
# fails in fitFDist (pylimma raises that message up front)
# =============================================================================
set.seed(1215)
.sv_var <- matrix(exp(rnorm(20)) * rchisq(20, df = 4) / 4, 4, 5)
write.csv(.sv_var, "R_sv_matrix_var.csv", row.names = FALSE)
.sv_out <- squeezeVar(.sv_var, df = 4)
write.csv(data.frame(var_post = as.vector(.sv_out$var.post), var_prior = .sv_out$var.prior,
                     df_prior = .sv_out$df.prior), "R_sv_matrix.csv", row.names = FALSE)
stopifnot(grepl("var is empty", try(squeezeVar(NULL, 5), silent = TRUE)),
          identical(dim(squeezeVar(matrix(c(0.5, 0.7)), df = 4)$var.post), c(2L, 1L)),
          identical(dim(.sv_out$var.post), c(4L, 5L)),
          grepl("different lengths", try(suppressWarnings(squeezeVar(as.vector(.sv_var), df = numeric(0))), silent = TRUE)),
          grepl("x and df1 have different lengths", try(squeezeVar(as.vector(.sv_var), df = c(5, 5)), silent = TRUE)))

# =============================================================================
# lmFit(method="robust", maxit=): MASS::rlm loops over 1L:maxit, so 10.0 is
# 10; with a fraction, a gene that does not converge fails in rlm's warning
# ("invalid format '%d'") (pylimma raises for a fraction)
# =============================================================================
.rm_y <- matrix(rnorm(60), 10)
.rm_d <- cbind(1, c(0, 0, 0, 1, 1, 1))
stopifnot(identical(suppressWarnings(lmFit(.rm_y, .rm_d, method = "robust", maxit = 10.0))$coefficients,
                    suppressWarnings(lmFit(.rm_y, .rm_d, method = "robust", maxit = 10L))$coefficients),
          grepl("invalid format", try(lmFit(.rm_y, .rm_d, method = "robust", maxit = 1.5), silent = TRUE)))

# =============================================================================
# genas subset rules: the genes each rule keeps (fit[genes, ], R's MArrayLM
# subset), also for a fit with two genes
# =============================================================================
set.seed(1216)
.gn_X <- cbind(1, (0:11) %% 2, ((0:11) %/% 2) %% 2)
.gn_y <- matrix(rnorm(80 * 12), 80, 12)
.gn_y[1:8, .gn_X[, 2] == 1] <- .gn_y[1:8, .gn_X[, 2] == 1] + 2
.gn_y[5:14, .gn_X[, 3] == 1] <- .gn_y[5:14, .gn_X[, 3] == 1] + 1.5
write.csv(.gn_y, "R_genas_subset_y.csv", row.names = FALSE)
.gn_n <- function(y) {
  fit <- eBayes(lmFit(y, .gn_X))
  vapply(c("all", "Fpval", "p.union", "p.int", "logFC", "predFC"),
         function(s) genas(fit, coef = c(2, 3), subset = s)$n, 0)
}
write.csv(data.frame(subset = c("all", "Fpval", "p.union", "p.int", "logFC", "predFC"),
                     n80 = .gn_n(.gn_y), n2 = .gn_n(.gn_y[c(1, 20), ])),
          "R_genas_subset_n.csv", row.names = FALSE)

# =============================================================================
# genas takes coef by name as fit[, coef] does
# =============================================================================
.gc_y <- as.matrix(read.csv("R_genas_subset_y.csv"))
.gc_fit <- eBayes(lmFit(.gc_y, cbind(Intercept = 1, A = (0:11) %% 2, B = ((0:11) %/% 2) %% 2)))
stopifnot(identical(genas(.gc_fit, coef = c("A", "B")), genas(.gc_fit, coef = c(2, 3))),
          grepl("cannot be evaluated at initial parameters", try(optim(c(1, 1), function(p) Inf), silent = TRUE)))

# =============================================================================
# genas results (optim's Nelder-Mead): data with a strong biological
# correlation (the null optimum is far from the alternative's) and the
# subset-rule data
# =============================================================================
.gr_u <- function(n, s = 0) { i <- 1:n; z <- sin(i * 12.9898 + s) * 43758.5453; (z - floor(z)) - 0.5 }
.gr_y <- matrix(.gr_u(1800), 300, 6)
.gr_y[, 3:4] <- .gr_y[, 3:4] + .gr_y[, 1:2] * 0.8
write.csv(.gr_y, "R_genas_y.csv", row.names = FALSE)
.gr_case <- function(case, fit, ...) {
  r <- genas(fit, ...)
  data.frame(case = case, technical = r$technical.correlation, biological = r$biological.correlation,
             deviance = r$deviance, p_value = r$p.value, n = r$n,
             v11 = r$covariance.matrix[1, 1], v21 = r$covariance.matrix[2, 1], v22 = r$covariance.matrix[2, 2])
}
.gr_sub <- eBayes(lmFit(as.matrix(read.csv("R_genas_subset_y.csv")),
                        cbind(1, (0:11) %% 2, ((0:11) %/% 2) %% 2)))
write.csv(rbind(
  .gr_case("strong", eBayes(lmFit(.gr_y, model.matrix(~ 0 + factor(c(1, 1, 2, 2, 3, 3))))), coef = c(1, 2)),
  .gr_case("subset_all", .gr_sub, coef = c(2, 3)),
  .gr_case("subset_Fpval", .gr_sub, coef = c(2, 3), subset = "Fpval")),
  "R_genas.csv", row.names = FALSE)

# =============================================================================
# backgroundCorrect on an EListRaw uses its Eb (and drops it)
# =============================================================================
.bc_E <- as.matrix(read.csv("R_intensities_foreground.csv"))
.bc_Eb <- as.matrix(read.csv("R_intensities_background.csv"))
.bc_raw <- backgroundCorrect(new("EListRaw", list(E = .bc_E, Eb = .bc_Eb)), method = "subtract", verbose = FALSE)
stopifnot(identical(.bc_raw$E, backgroundCorrect.matrix(.bc_E, .bc_Eb, method = "subtract")),
          is.null(.bc_raw$Eb),
          identical(backgroundCorrect(new("EListRaw", list(E = .bc_E, Eb = .bc_Eb)), verbose = FALSE)$E, .bc_raw$E))

# =============================================================================
# normexp.signal where the normal part is narrow next to the exponential
# mean (large sigma^2 / alpha): the result is a small difference of large
# terms
# =============================================================================
.ns_x <- c(60, 80, 100, 120, 150, 500)
write.csv(data.frame(x = .ns_x,
                     alpha_0.01 = normexp.signal(c(100, log(10), log(0.01)), .ns_x),
                     alpha_0.002 = normexp.signal(c(100, log(10), log(0.002)), .ns_x),
                     alpha_20 = normexp.signal(c(100, log(10), log(20)), .ns_x)),
          "R_normexp_signal_tails.csv", row.names = FALSE)

# =============================================================================
# normexp.fit(method="mle") on an Illumina-like array (narrow normal part,
# large exponential mean): nlminb's default x.tol (1.5e-8) is needed to
# reach R's optimum on this flat likelihood
# =============================================================================
set.seed(1219)
.nm_x <- sapply(1:4, function(i) rnorm(10000, 30, 1.2 + 0.04 * i) + rexp(10000, 1 / (150 + 5 * i)))[, 4]
write.csv(data.frame(x = .nm_x), "R_normexp_mle_flat_x.csv", row.names = FALSE)
write.csv(data.frame(par = normexp.fit(.nm_x, method = "mle")$par), "R_normexp_mle_flat.csv", row.names = FALSE)

# =============================================================================
# diffSplice with no residual df: the genewise variance is 0/0 (NaN), and so
# are the posterior variance, t and F
# =============================================================================
set.seed(1220)
.dsz_y <- matrix(rnorm(18), 6, 3)
write.csv(.dsz_y, "R_diffsplice_nodf_y.csv", row.names = FALSE)
.dsz <- suppressWarnings(diffSplice(lmFit(.dsz_y, cbind(1, c(0, 1, 0), c(0, 0, 1))),
                                    geneid = rep(c("G0", "G1"), each = 3), verbose = FALSE))
stopifnot(all(is.nan(.dsz$gene.s2)), all(is.na(.dsz$t)), all(is.na(.dsz$gene.F)))

# =============================================================================
# arrayWeights with a design given as a vector: as.matrix(design), one column
# =============================================================================
set.seed(1221)
.aw_E <- matrix(rnorm(40 * 9), 40, 9)
.aw_E[, 3] <- .aw_E[, 3] * 3
.aw_d <- c(0, 0, 0, 1, 1, 1, 0, 0, 0)
write.csv(.aw_E, "R_aw_vector_design_E.csv", row.names = FALSE)
write.csv(data.frame(genebygene = arrayWeights(.aw_E, .aw_d, method = "genebygene"),
                     reml = arrayWeights(.aw_E, .aw_d, method = "reml")),
          "R_aw_vector_design.csv", row.names = FALSE)
stopifnot(all.equal(arrayWeights(.aw_E, .aw_d, method = "genebygene"), arrayWeights(.aw_E, matrix(.aw_d), method = "genebygene")))

# =============================================================================
# Column subset of a rank-deficient fit: R takes cov.coefficients in pivot
# order (jj <- which(pivot %in% j)), so for columns out of that order the
# covariance does not line up with the coefficients (pylimma follows the
# requested order)
# =============================================================================
set.seed(1301)
.rk_d <- cbind(A = 1, B = rep(0:1, 4), C = rep(c(0, 0, 1, 1), 2))
.rk_d <- cbind(.rk_d, D = .rk_d[, "B"] + .rk_d[, "C"])
.rk_fit <- suppressWarnings(suppressMessages(lmFit(matrix(rnorm(80), 10), .rk_d)))
.rk_sub <- .rk_fit[, c("C", "A")]
stopifnot(identical(colnames(.rk_sub$coefficients), c("C", "A")),
          identical(rownames(.rk_sub$cov.coefficients), c("A", "C")))

# =============================================================================
# qf in both tails at large df (the chi-squared limits above 4e5), as
# fitFDistRobustly uses it
# =============================================================================
.qfl_p <- c(1e-6, 0.05, 0.5, 0.9)
.qfl <- expand.grid(p = .qfl_p, case = c("df2_1e6", "df1_5e5", "df2_inf", "finite"), lower = c(TRUE, FALSE),
                    stringsAsFactors = FALSE)
.qfl_df <- list(df2_1e6 = c(5, 1e6), df1_5e5 = c(5e5, 10), df2_inf = c(5, Inf), finite = c(5, 20))
.qfl$q <- mapply(function(p, case, lower) qf(p, .qfl_df[[case]][1], .qfl_df[[case]][2], lower.tail = lower),
                 .qfl$p, .qfl$case, .qfl$lower)
write.csv(.qfl, "R_qf_large_df.csv", row.names = FALSE)

# =============================================================================
# A missing block label: lmFit and voom fail in chol(), duplicateCorrelation
# in qr.qty() (pylimma raises a clear error); a numeric NaN label is a
# level in duplicateCorrelation, which then runs (pylimma raises too)
# =============================================================================
set.seed(1303)
.nb_y <- matrix(rnorm(60), 10)
.nb_d <- cbind(1, c(0, 0, 0, 1, 1, 1))
.nb_block <- c(1, 1, 2, 2, 3, NA)
stopifnot(grepl("chol", try(lmFit(.nb_y, .nb_d, block = .nb_block, correlation = 0.3), silent = TRUE)),
          grepl("qr.qty", try(duplicateCorrelation(.nb_y, .nb_d, block = .nb_block), silent = TRUE)),
          grepl("chol", try(voom(matrix(rpois(60, 20), 10), .nb_d, block = .nb_block, correlation = 0.3),
                            silent = TRUE)),
          is.finite(duplicateCorrelation(.nb_y, .nb_d, block = c(1, 1, 2, 2, NaN, NaN))$consensus.correlation))

# =============================================================================
# lmFit input checks: NA in the design stops explicitly; no samples and a
# non-finite matrix in is.fullrank fail inside lm.fit / eigen (pylimma
# raises clear errors)
# =============================================================================
.lc_y <- matrix(rnorm(40), 10)
.lc_d <- cbind(1, c(0, 0, 1, NA))
stopifnot(grepl("NAs not allowed in design matrix", try(lmFit(.lc_y, .lc_d), silent = TRUE)),
          grepl("0 \\(non-NA\\) cases", try(lmFit(matrix(numeric(0), 10, 0), matrix(numeric(0), 0, 1)), silent = TRUE)),
          grepl("infinite or missing values", try(is.fullrank(cbind(1, c(1, NA, 3))), silent = TRUE)))

# =============================================================================
# getEAWP input forms: an EList's vector E is one column (as.matrix), a
# bare vector one row; complex values are not numeric; an all-NA gene has
# Amean NaN without a warning
# =============================================================================
stopifnot(identical(dim(getEAWP(new("EList", list(E = c(1, 2, 3, 4))))$exprs), c(4L, 1L)),
          identical(dim(getEAWP(c(1, 2, 3, 4))$exprs), c(1L, 4L)),
          grepl("doesn't contain numeric expression values", try(getEAWP(matrix(1i, 2, 2)), silent = TRUE)),
          is.nan(getEAWP(rbind(c(1, 2), c(NA, NA)))$Amean[2]))

# =============================================================================
# lmFit(method="robust", k=): MASS::rlm fails obscurely for k <= 0
# (pylimma raises a clear error)
# =============================================================================
.rk_y <- matrix(rnorm(60), 10)
.rk_d2 <- cbind(1, c(0, 0, 0, 1, 1, 1))
stopifnot(inherits(try(suppressWarnings(lmFit(.rk_y, .rk_d2, method = "robust", k = 0)), silent = TRUE), "try-error"),
          grepl("negative weights", try(suppressWarnings(lmFit(.rk_y, .rk_d2, method = "robust", k = -1)), silent = TRUE)))

# =============================================================================
# wsva's design checks (.lmEffects): a NULL design (also for an EList with a
# design, which as.matrix(y) drops) and one with the wrong number of rows
# stop explicitly
# =============================================================================
.ws_y <- matrix(rnorm(120), 20)
stopifnot(grepl("design matrix not specified", try(wsva(.ws_y, NULL), silent = TRUE)),
          grepl("design matrix not specified",
                try(wsva(new("EList", list(E = .ws_y, design = cbind(1, rep(0:1, each = 3)))), NULL), silent = TRUE)),
          grepl("row dimension of design matrix must match", try(wsva(.ws_y, cbind(1, 1:5)), silent = TRUE)))

# =============================================================================
# avereps probe IDs: as.character(ID) (1 and 1.0 are one probe), NA IDs one
# group (rowsum warns), an ID of the wrong length stops
# =============================================================================
.ai_x <- matrix(as.numeric(1:6), 3)
stopifnot(identical(unname(suppressWarnings(avereps(.ai_x, ID = c("a", NA, NA)))), cbind(c(1, 2.5), c(4, 5.5))),
          identical(unname(avereps(.ai_x, ID = c(1, 1.0, 2))), cbind(c(1.5, 3), c(4.5, 6))),
          grepl("incorrect length for 'group'", try(avereps(.ai_x, ID = c("a", "b")), silent = TRUE)))

# =============================================================================
# asMatrixWeights(NULL) fails inside as.matrix (pylimma raises a clear error)
# =============================================================================
stopifnot(grepl("must be of a vector type", try(limma:::asMatrixWeights(NULL, c(2, 2)), silent = TRUE)))

# =============================================================================
# as.data.frame.MArrayLM names columns component.colname: coefficient
# matrices by the coefficient names, a genes data frame by its columns; a
# one-column component keeps its own column name, or the component's
# =============================================================================
.adf_fit <- eBayes(lmFit(matrix(rnorm(42), 7), cbind(a = 1, bvsa = c(0, 0, 1, 1, 0, 1))))
.adf_fit$genes <- data.frame(ID = paste0("g", 1:7), Sym = letters[1:7])
stopifnot(all(c("coefficients.a", "coefficients.bvsa", "t.bvsa", "p.value.bvsa", "genes.ID", "genes.Sym",
                "sigma", "F") %in% names(as.data.frame(.adf_fit))))
.adf_fit$genes <- data.frame(ProbeID = paste0("g", 1:7))
.adf_fit$w <- matrix(1, 7, 1)
stopifnot(all(c("ProbeID", "w") %in% names(as.data.frame(.adf_fit))))

# =============================================================================
# model.matrix needs two levels for every factor term, also without an
# intercept
# =============================================================================
.mm_d <- data.frame(f = factor(rep("a", 4)), x = 1:4)
stopifnot(grepl("2 or more levels", try(model.matrix(~ f, .mm_d), silent = TRUE)),
          grepl("2 or more levels", try(model.matrix(~ 0 + f, .mm_d), silent = TRUE)),
          identical(dim(model.matrix(~ x, .mm_d)), c(4L, 2L)))

# =============================================================================
# decideTests / classifyTestsF with unusable arguments: p.value = NaN gives
# an all-NA result; an impossible p.value or df, or a correlation matrix
# with missing values, fails deep in qf / eigen / if() (pylimma raises
# clear errors); a scalar p-value is a 1 x 1 matrix
# =============================================================================
.dt_fit <- eBayes(lmFit(matrix(rnorm(120), 20), cbind(1, c(0, 0, 0, 1, 1, 1), c(0, 1, 0, 1, 0, 1))))
stopifnot(all(is.na(decideTests(.dt_fit, p.value = NaN))),
          inherits(try(classifyTestsF(.dt_fit, p.value = 2), silent = TRUE), "try-error"),
          inherits(try(classifyTestsF(.dt_fit, df = 0), silent = TRUE), "try-error"),
          inherits(try(classifyTestsF(.dt_fit, cor.matrix = matrix(NA, 3, 3)), silent = TRUE), "try-error"),
          identical(dim(decideTests(0.5)), c(1L, 1L)))

# =============================================================================
# genas with unusable coefficients: a position out of range fails in
# fit[, coef], a repeated coefficient or identical contrasts in chol()
# (pylimma raises clear errors); with data where some genes pass genas's
# criterion, as R otherwise stops earlier
# =============================================================================
set.seed(1300)
.gu_fit <- eBayes(lmFit(matrix(rnorm(360), 60), cbind(1, c(0, 0, 0, 1, 1, 1), c(0, 1, 0, 1, 0, 1))))
stopifnot(inherits(try(genas(.gu_fit, coef = c(1, 6)), silent = TRUE), "try-error"),
          grepl("chol", try(genas(.gu_fit, coef = c(2, 2)), silent = TRUE)),
          grepl("chol", try(genas(contrasts.fit(.gu_fit, cbind(c(0, 1, 0), c(0, 1, 0)))), silent = TRUE)))

# =============================================================================
# fitmixture with a rank-deficient design (one array, or one mixprop):
# lm.fit gives NA coefficients, which fail a later assignment (pylimma
# raises a clear error)
# =============================================================================
.fm_y <- matrix(rnorm(40, 8), 10)
stopifnot(grepl("NAs are not allowed", try(fitmixture(.fm_y, rep(0.5, 4)), silent = TRUE)),
          grepl("NAs are not allowed", try(fitmixture(.fm_y[, 1, drop = FALSE], 0.3), silent = TRUE)))

# =============================================================================
# selectModel: criterion = NULL is match.arg's first choice ("aic")
# =============================================================================
.sm_y <- matrix(rnorm(36), 6)
.sm_dl <- list(cbind(1, rep(0:1, 3)), matrix(1, 6, 1))
stopifnot(identical(selectModel(.sm_y, .sm_dl, criterion = NULL)$pref, selectModel(.sm_y, .sm_dl)$pref))

# =============================================================================
# diffSplice orders genes and exons with order(geneid, exonid) on their own
# type: numeric IDs numerically (12 after 2)
# =============================================================================
set.seed(1304)
.ds_nex <- c(3, 2, 12, 4, 1, 5, 3, 3, 2, 4, 3, 3)
.ds_gid <- rep(1:12, .ds_nex)
.ds_eid <- unlist(lapply(.ds_nex, seq_len))
.ds_p <- sample(length(.ds_gid))
.ds_gid <- .ds_gid[.ds_p]
.ds_eid <- .ds_eid[.ds_p]
.ds_y <- matrix(rnorm(length(.ds_gid) * 8), ncol = 8)
write.csv(data.frame(geneid = .ds_gid, exonid = .ds_eid, .ds_y), "R_ds_numeric_input.csv", row.names = FALSE)
.ds <- diffSplice(eBayes(lmFit(.ds_y, cbind(1, rep(0:1, each = 4)))), geneid = .ds_gid, exonid = .ds_eid,
                  verbose = FALSE)
write.csv(data.frame(GeneID = .ds$genes$GeneID, ExonID = .ds$genes$ExonID, t = .ds$t[, 2]),
          "R_ds_numeric_exons.csv", row.names = FALSE)
write.csv(data.frame(GeneID = .ds$gene.genes$GeneID, F = .ds$gene.F[, 2]), "R_ds_numeric_genes.csv",
          row.names = FALSE)

# =============================================================================
# Matrix results keep the input's dimnames: shape-preserving functions,
# avedups (first name of each duplicate set) and avearrays (unique IDs)
# =============================================================================
set.seed(1305)
.dn_x <- matrix(rnorm(64, 8), 8, dimnames = list(letters[1:8], LETTERS[1:8]))
.dn_same <- list(normalizeQuantiles(.dn_x), normalizeMedianValues(.dn_x), normalizeCyclicLoess(.dn_x),
                 normalizeBetweenArrays(.dn_x, method = "scale"), removeBatchEffect(.dn_x, batch = rep(1:2, 4)),
                 voom(2^.dn_x)$E)
stopifnot(all(vapply(.dn_same, function(m) identical(dimnames(m), dimnames(.dn_x)), TRUE)),
          identical(lapply(dimnames(avedups(.dn_x, ndups = 2, spacing = 2)), as.vector),
                    list(c("a", "b", "e", "f"), LETTERS[1:8])),
          identical(dimnames(avearrays(.dn_x, rep(c("p", "q"), 4))), list(letters[1:8], c("p", "q"))),
          identical(colnames(avearrays(.dn_x, rep(c("p", "q"), 4), weights = matrix(1, 8, 8))), c("IDp", "IDq")))

# =============================================================================
# topTable with a genelist: the genelist columns come first and the rows keep
# the fit's row names (an ID / ID0 column when they are duplicated)
# =============================================================================
set.seed(1306)
.tg_y <- matrix(rnorm(120), 20, 6, dimnames = list(paste0("g", 1:20), NULL))
.tg_d <- cbind(1, rep(0:1, 3), c(0, 0, 1, 1, 0, 1))
write.csv(.tg_y, "R_toptable_genelist_expr.csv")
.tg_fit <- eBayes(lmFit(.tg_y, .tg_d))
write.csv(topTable(.tg_fit, coef = 2, number = 5, genelist = paste0("id", 1:20)), "R_toptable_genelist_unique.csv")
write.csv(topTable(.tg_fit, coef = 2:3, number = 5, genelist = paste0("id", 1:20)), "R_toptable_genelist_unique_F.csv")
rownames(.tg_y)[19:20] <- "g5"
.tg_fit <- eBayes(lmFit(.tg_y, .tg_d))
write.csv(topTable(.tg_fit, coef = 2, number = 5, genelist = data.frame(ID = paste0("id", 1:20), Sym = letters[1:20])),
          "R_toptable_genelist_dup.csv")

# =============================================================================
# backgroundCorrect: R fails or warns on a background of another shape, a
# column with no value above background (method "minimum") and a NaN or vector offset
# =============================================================================
.bce_E <- matrix(1:40 + 0, 10)
.bce_err <- function(expr) conditionMessage(tryCatch(expr, error = identity, warning = identity))
stopifnot(all(vapply(list(matrix(1, 10, 1), matrix(1, 1, 4), 1, rep(1, 10)), function(Eb)
            .bce_err(backgroundCorrect.matrix(.bce_E, Eb, method = "subtract", verbose = FALSE)), "") ==
            "non-conformable arrays"),
          .bce_err(backgroundCorrect.matrix(.bce_E, .bce_E + 1, method = "minimum", verbose = FALSE)) ==
            "no non-missing arguments to min; returning Inf",
          .bce_err(backgroundCorrect.matrix(.bce_E, matrix(1, 10, 4), offset = NaN, verbose = FALSE)) ==
            "argument is not interpretable as logical",
          .bce_err(backgroundCorrect.matrix(.bce_E, matrix(1, 10, 4), offset = 1:2, verbose = FALSE)) ==
            "the condition has length > 1")

# =============================================================================
# normalizeQuantiles with -Inf: approx() gives NaN between infinite values
# (two tied -Inf in a column with ties = TRUE)
# =============================================================================
set.seed(1307)
.nqi_x <- matrix(rnorm(48, 8, 2), 12)
.nqi_x[cbind(c(1, 6, 8, 4), c(1, 3, 3, 2))] <- c(-Inf, -Inf, -Inf, NA)
write.csv(.nqi_x, "R_nq_inf_input.csv", row.names = FALSE)
write.csv(normalizeQuantiles(.nqi_x), "R_nq_inf_ties.csv", row.names = FALSE)
write.csv(normalizeQuantiles(.nqi_x, ties = FALSE), "R_nq_inf_noties.csv", row.names = FALSE)

# =============================================================================
# zscore: any p<distribution> function, with its own argument names (hyper is
# plain phyper, not zscoreHyper)
# =============================================================================
.zs_cases <- list(
  list("norm", c(-3, 0.5, 4), list(mean = 1, sd = 2)),
  list("t", c(-30, -1, 2, 40), list(df = 5)),
  list("t", c(-3, 1, 6), list(df = 5, ncp = 1)),
  list("f", c(0.2, 1, 3, 30), list(df1 = 3, df2 = 9)),
  list("f", c(0.2, 1, 3, 30), list(df1 = 3, df2 = 9, ncp = 2)),
  list("chisq", c(0.5, 3, 25), list(df = 4)),
  list("chisq", c(0.5, 3, 25), list(df = 4, ncp = 1.5)),
  list("beta", c(0.1, 0.5, 0.9), list(shape1 = 2, shape2 = 3)),
  list("exp", c(0.1, 1, 8), list(rate = 2)),
  list("gamma", c(0.2, 2, 9), list(shape = 2, rate = 1.5)),
  list("binom", c(0, 3, 7, 10), list(size = 10, prob = 0.3)),
  list("pois", c(0, 2, 9), list(lambda = 2)),
  list("hyper", c(0, 1, 3, 5, 8, 10), list(m = 10, n = 20, k = 10)),
  list("nbinom", c(0, 3, 12), list(size = 3, prob = 0.4)),
  list("nbinom", c(0, 3, 12), list(size = 3, mu = 2)),
  list("geom", c(0, 2, 9), list(prob = 0.3)),
  list("unif", c(-0.5, 1, 2.5), list(min = -1, max = 3)),
  list("logis", c(-2, 0, 2), list(location = 0.5, scale = 2)),
  list("cauchy", c(-50, 0, 3), list(location = 1, scale = 0.5)),
  list("lnorm", c(0.1, 1, 20), list(meanlog = 0.5, sdlog = 1.5)),
  list("weibull", c(0.1, 1, 4), list(shape = 1.5, scale = 2)))
write.csv(do.call(rbind, lapply(seq_along(.zs_cases), function(i) {
  cs <- .zs_cases[[i]]
  data.frame(case = i, distribution = cs[[1]], args = paste(names(cs[[3]]), unlist(cs[[3]]), sep = "=", collapse = ";"),
             q = cs[[2]], z = do.call(zscore, c(list(cs[[2]], cs[[1]]), cs[[3]])))
})), "R_zscore_distributions.csv", row.names = FALSE)
# q is recycled against longer parameters
stopifnot(identical(zscore(1, "norm", mean = c(0, 1)), c(1, 0)))

# =============================================================================
# isNumeric: factors, logicals and complex values are not numeric
# =============================================================================
stopifnot(!isNumeric(factor(1:2)), !isNumeric(data.frame(a = factor(1:2))), !isNumeric(TRUE),
          !isNumeric(1i), isNumeric(1:2), isNumeric(data.frame(a = 1)))

# =============================================================================
# makeUnique: NA stays NA (and fails alongside a repeated value); values are
# compared as as.character gives them
# =============================================================================
stopifnot(identical(makeUnique(c(NA, "a", NA)), c(NA, "a", NA)),
          grepl("NAs are not allowed", try(makeUnique(c(NA, "a", "a")), silent = TRUE)),
          identical(makeUnique(c(1, 1.0, 2)), c("11", "12", "2")),
          identical(makeUnique(c(rep("c", 10), "d")), c(sprintf("c%02d", 1:10), "d")))

# =============================================================================
# logsumexp recycles x and y of different lengths
# =============================================================================
stopifnot(isTRUE(all.equal(logsumexp(c(1, 2), c(3, 4, 5, 6)), log(exp(c(1, 2, 1, 2)) + exp(3:6)))))

# =============================================================================
# chooseLowessSpan is vectorised in n (pmin): n = 0 gives 1, a negative n NaN
# =============================================================================
stopifnot(identical(chooseLowessSpan(c(0, -5)), c(1, NaN)),
          identical(chooseLowessSpan(c(20, 1000)), c(chooseLowessSpan(20), chooseLowessSpan(1000))))
