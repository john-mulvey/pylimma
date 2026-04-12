#!/usr/bin/env Rscript
# Fixture generation for pylimma classes.py (EList / MArrayLM subsetting)
#
# Run with: Rscript generate_classes_fixtures.R
#
# Produces CSVs capturing each slot of R EList and MArrayLM objects after
# a range of [i, j] subsetting operations. The Python tests load these and
# compare slot-by-slot to pylimma's EList/MArrayLM __getitem__ output.

library(limma)

cat("=== pylimma classes.py Fixture Generation ===\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("limma version: %s\n", packageVersion("limma")))
cat(sprintf("Date: %s\n\n", Sys.time()))

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
cat("Generating EList subsetting fixtures...\n")

el <- new("EList", list(
  E = E,
  weights = weights,
  genes = genes,
  targets = targets,
  design = design
))

write_elist <- function(obj, tag) {
  write.csv(obj$E,       sprintf("R_elist_%s_E.csv", tag),       row.names = TRUE)
  write.csv(obj$weights, sprintf("R_elist_%s_weights.csv", tag), row.names = TRUE)
  write.csv(obj$genes,   sprintf("R_elist_%s_genes.csv", tag),   row.names = TRUE)
  write.csv(obj$targets, sprintf("R_elist_%s_targets.csv", tag), row.names = TRUE)
  write.csv(obj$design,  sprintf("R_elist_%s_design.csv", tag),  row.names = TRUE)
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
cat("Generating MArrayLM subsetting fixtures...\n")

fit <- lmFit(el, design)
fit <- eBayes(fit)

write_marraylm <- function(obj, tag) {
  write.csv(obj$coefficients,   sprintf("R_marraylm_%s_coefficients.csv",   tag), row.names = TRUE)
  write.csv(obj$stdev.unscaled, sprintf("R_marraylm_%s_stdev_unscaled.csv", tag), row.names = TRUE)
  write.csv(obj$t,              sprintf("R_marraylm_%s_t.csv",              tag), row.names = TRUE)
  write.csv(obj$p.value,        sprintf("R_marraylm_%s_p_value.csv",        tag), row.names = TRUE)
  write.csv(obj$lods,            sprintf("R_marraylm_%s_lods.csv",           tag), row.names = TRUE)
  write.csv(data.frame(
    Amean       = obj$Amean,
    sigma       = obj$sigma,
    df_residual = obj$df.residual,
    df_total    = obj$df.total,
    s2_post     = obj$s2.post
  ), sprintf("R_marraylm_%s_i_slots.csv", tag), row.names = TRUE)
  write.csv(obj$genes, sprintf("R_marraylm_%s_genes.csv", tag), row.names = TRUE)
}

write_marraylm(fit, "full")
write_marraylm(fit[1:10, ], "rows")
write_marraylm(fit[, 2, drop = FALSE], "cols")
write_marraylm(fit[1:10, 2, drop = FALSE], "both")
write_marraylm(fit[c("gene3", "gene7", "gene15"), ], "rowstr")

cat("\nDone. Fixtures written to current working directory.\n")
