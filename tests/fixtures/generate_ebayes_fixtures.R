# Generate reference values for pylimma ebayes tests
# Run with: Rscript generate_ebayes_fixtures.R

library(limma)

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

cat("Generated ebayes fixtures\n")
cat(sprintf("s2.prior=%.6f, df.prior=%.6f\n", fit$s2.prior, fit$df.prior))
