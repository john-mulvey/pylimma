# Generate reference values for pylimma lmfit tests
# Run with: Rscript generate_lmfit_fixtures.R

library(limma)

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

cat("Generated lmfit fixtures\n")
cat(sprintf("n_genes=%d, n_samples=%d, n_coefs=%d\n", n_genes, n_samples, ncol(design)))
cat(sprintf("rank=%d, df.residual=%d\n", fit$rank, fit$df.residual[1]))
