# Generate reference values for pylimma utils tests
# Run with: Rscript generate_utils_fixtures.R

library(limma)

# Test values for trigammaInverse
test_x <- c(0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1e6, 1e8)
tri_inv <- trigammaInverse(test_x)

out <- data.frame(x = test_x, trigamma_inverse = tri_inv)
write.csv(out, "trigamma_inverse.csv", row.names = FALSE)

cat("Generated trigamma_inverse.csv\n")
