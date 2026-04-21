#!/usr/bin/env Rscript
# ------------------------------------------------------------------
# ONE-TIME GENERATION OF R REFERENCE PIPELINE OUTPUTS
#
# Runs the canonical R limma pipelines on the four committed benchmark
# datasets and writes top-tables to benchmarks/data/R_references/.
# The parity notebooks under examples/ load these CSVs and compare
# them to pylimma output computed live in Python.
#
# Only limma + edgeR are required; no Bioconductor data packages.
#
# Usage:
#   cd pylimma/benchmarks
#   Rscript _setup_R_references.R
# ------------------------------------------------------------------

suppressPackageStartupMessages({
    library(limma)
    library(edgeR)
})

OUT <- "data/R_references"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("R %s; limma %s; edgeR %s\n",
            R.version.string,
            packageVersion("limma"),
            packageVersion("edgeR")))
cat(sprintf("Writing R reference outputs to %s\n\n",
            normalizePath(OUT)))


read_gz <- function(path) as.matrix(read.csv(gzfile(path), row.names = 1))

two_group <- function(labels) {
    g <- factor(labels)
    X <- model.matrix(~0 + g)
    C <- matrix(0, ncol = 1, nrow = ncol(X))
    C[1, 1] <- -1; C[2, 1] <- 1
    list(design = X, contrasts = C)
}


# ------------------------------------------------------------------
# ALL (Chiaretti 2004) - pipeline_a: lmFit + contrasts.fit + eBayes
# ------------------------------------------------------------------
cat("[ALL] lmFit -> contrasts.fit -> eBayes -> topTable\n")
expr    <- read_gz("data/all_expr.csv.gz")
targets <- read.csv("data/all_targets.csv", row.names = 1)
bt      <- substr(as.character(targets$BT), 1, 1)
d       <- two_group(bt)
fit  <- lmFit(expr, d$design)
fit  <- contrasts.fit(fit, d$contrasts)
fit  <- eBayes(fit)
tt   <- topTable(fit, coef = 1, number = Inf, sort.by = "none")
tt$ProbeID <- rownames(tt)
write.csv(tt, gzfile(file.path(OUT, "all_toptable.csv.gz")), row.names = FALSE)
cat(sprintf("  %d rows x %d cols\n", nrow(tt), ncol(tt)))


# ------------------------------------------------------------------
# GSE60450 - pipeline_b: voom + lmFit + contrasts.fit + eBayes
# ------------------------------------------------------------------
cat("\n[GSE60450] voom -> lmFit -> contrasts.fit -> eBayes -> topTable\n")
counts  <- read_gz("data/gse60450_counts.csv.gz")
targets <- read.csv("data/gse60450_targets.csv", row.names = 1)
celltype <- sapply(strsplit(as.character(targets$group), "\\."),
                   function(x) x[1])
d   <- two_group(celltype)
v   <- voom(counts, d$design)
fit <- lmFit(v$E, d$design, weights = v$weights)
fit <- contrasts.fit(fit, d$contrasts)
fit <- eBayes(fit)
tt  <- topTable(fit, coef = 1, number = Inf, sort.by = "none")
tt$GeneID <- rownames(tt)
write.csv(tt, gzfile(file.path(OUT, "gse60450_toptable.csv.gz")), row.names = FALSE)
cat(sprintf("  %d rows x %d cols\n", nrow(tt), ncol(tt)))


# ------------------------------------------------------------------
# Yoruba - pipeline_b: voom on 69-sample RNA-seq (scaling test)
# ------------------------------------------------------------------
cat("\n[Yoruba] voom -> lmFit -> contrasts.fit -> eBayes -> topTable\n")
counts  <- read_gz("data/yoruba_counts.csv.gz")
targets <- read.csv("data/yoruba_targets.csv", row.names = 1)
d   <- two_group(targets$gender)
v   <- voom(counts, d$design)
fit <- lmFit(v$E, d$design, weights = v$weights)
fit <- contrasts.fit(fit, d$contrasts)
fit <- eBayes(fit)
tt  <- topTable(fit, coef = 1, number = Inf, sort.by = "none")
tt$GeneID <- rownames(tt)
write.csv(tt, gzfile(file.path(OUT, "yoruba_toptable.csv.gz")), row.names = FALSE)
cat(sprintf("  %d rows x %d cols\n", nrow(tt), ncol(tt)))


# ------------------------------------------------------------------
# Pasilla - pipeline_d: diffSplice + topSplice
# ------------------------------------------------------------------
cat("\n[Pasilla] lmFit -> eBayes -> diffSplice -> topSplice\n")
counts  <- read_gz("data/pasilla_counts.csv.gz")
targets <- read.csv("data/pasilla_targets.csv", row.names = 1)
d       <- two_group(targets$condition)
geneid  <- as.character((seq_len(nrow(counts)) - 1L) %/% 5L)  # 0-indexed (match Python)
log2_counts <- log2(counts + 1)
fit <- lmFit(log2_counts, d$design)
fit <- eBayes(fit)
ds  <- diffSplice(fit, geneid = geneid)
ts  <- topSplice(ds, number = Inf, sort.by = "none")
# topSplice already carries GeneID as a column; the row names are an
# internal exon index (e.g. "5", "10", "55") that isn't meaningful
# to a reader. Write GeneID explicitly and drop row names.
write.csv(ts, gzfile(file.path(OUT, "pasilla_topsplice.csv.gz")), row.names = FALSE)
cat(sprintf("  %d rows x %d cols\n", nrow(ts), ncol(ts)))


# ------------------------------------------------------------------
# Provenance sidecar: record the R and Bioc versions these outputs
# were computed against, so readers can see the exact toolchain.
# ------------------------------------------------------------------
writeLines(c(
    sprintf("generated_at: %s", format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
    sprintf("r_version: %s", R.version.string),
    sprintf("limma_version: %s", packageVersion("limma")),
    sprintf("edger_version: %s", packageVersion("edgeR")),
    sprintf("platform: %s-%s-%s", Sys.info()["sysname"],
            Sys.info()["release"], Sys.info()["machine"])
), file.path(OUT, "VERSIONS.txt"))

cat(sprintf("\nDone. Reference outputs in %s\n", OUT))
