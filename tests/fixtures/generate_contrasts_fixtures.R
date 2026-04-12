# Generate reference values for pylimma contrasts tests
# Run with: Rscript generate_contrasts_fixtures.R

library(limma)

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

cat("Generated contrasts fixtures\n")
cat(sprintf("n_genes=%d, n_samples=%d\n", n_genes, n_samples))
cat(sprintf("Original coefs: %d, Contrast coefs: %d\n", ncol(fit$coefficients), ncol(fit2$coefficients)))
