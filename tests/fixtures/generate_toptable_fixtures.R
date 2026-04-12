# Generate reference values for pylimma toptable tests
# Run with: Rscript generate_toptable_fixtures.R

library(limma)

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

cat("Generated toptable fixtures\n")
cat(sprintf("Top genes: %s\n", paste(head(tt$gene, 5), collapse = ", ")))
