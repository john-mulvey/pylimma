# Generate reference values for pylimma squeeze_var tests
# Run with: Rscript generate_squeeze_var_fixtures.R

library(limma)

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

cat("Generated squeeze_var fixtures\n")
cat(sprintf("fitFDist: scale=%.6f, df2=%.6f\n", fit$scale, fit$df2))
cat(sprintf("squeezeVar: var.prior=%.6f, df.prior=%.6f\n", sv$var.prior, sv$df.prior))
