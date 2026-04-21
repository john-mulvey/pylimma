#!/usr/bin/env Rscript
# ------------------------------------------------------------------
# ONE-TIME BENCHMARK DATASET EXTRACTION
#
# Installs three Bioconductor data packages (ALL, pasilla,
# tweeDEseqCountData) via BiocManager, extracts the datasets we need
# to CSV, gzips, and then *removes any package that wasn't already
# installed before this script ran* - leaving the R library as close
# to its pre-run state as possible.
#
# Run once on the maintainer's machine; the committed CSVs in
# benchmarks/data/ are then the source of truth for every subsequent
# benchmark run. Users do not need to re-run this.
#
# Usage:
#   cd pylimma/benchmarks
#   Rscript _setup_datasets.R
# ------------------------------------------------------------------

suppressPackageStartupMessages({
    library(tools)
})

OUT_DIR <- "data"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------------
# Snapshot installed packages BEFORE we install anything, so we can
# compute the exact set to remove when we're done.
# ------------------------------------------------------------------
pre_install <- rownames(installed.packages())
cat(sprintf("[snapshot] %d packages installed before setup\n",
            length(pre_install)))

# ------------------------------------------------------------------
# Install the three data packages if not already present.
# ------------------------------------------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

data_pkgs <- c("ALL", "pasilla", "tweeDEseqCountData")
to_install <- setdiff(data_pkgs, pre_install)
if (length(to_install) > 0) {
    cat(sprintf("[install] %s\n", paste(to_install, collapse = ", ")))
    BiocManager::install(to_install, update = FALSE, ask = FALSE)
} else {
    cat("[install] all three data packages already present\n")
}

# ------------------------------------------------------------------
# Dataset 1: ALL (Chiaretti 2004). Pre-RMA-normalised HG-U95Av2
# expression data plus clinical phenotype.
# ------------------------------------------------------------------
cat("\n[1/3] ALL (Chiaretti 2004)\n")
suppressPackageStartupMessages({
    library(Biobase)
    library(ALL)
})
data("ALL")
expr <- exprs(ALL)
pheno <- pData(ALL)[, c("BT", "mol.biol", "sex", "age")]
write.csv(expr, gzfile(file.path(OUT_DIR, "all_expr.csv.gz")))
write.csv(pheno, file.path(OUT_DIR, "all_targets.csv"))
cat(sprintf("    shape: %d x %d\n", nrow(expr), ncol(expr)))

# Small-overhead dataset: 50 genes from ALL (for fixed-cost floor).
write.csv(expr[1:50, , drop = FALSE],
          gzfile(file.path(OUT_DIR, "all_small_expr.csv.gz")))
write.csv(pheno, file.path(OUT_DIR, "all_small_targets.csv"))

# ------------------------------------------------------------------
# Dataset 2: Pasilla (Drosophila RNA-seq for differential splicing).
# Package ships the counts as a plain TSV - copy it in.
# ------------------------------------------------------------------
cat("\n[2/3] Pasilla\n")
suppressPackageStartupMessages(library(pasilla))
pas_dir <- system.file("extdata", package = "pasilla")
counts <- as.matrix(read.table(
    file.path(pas_dir, "pasilla_gene_counts.tsv"),
    header = TRUE, sep = "\t", row.names = 1
))
anno <- read.csv(file.path(pas_dir, "pasilla_sample_annotation.csv"),
                 row.names = 1)
# In pasilla 1.38.0 the counts columns are "treated1" / "untreated1"
# but the annotation's row names (from the "file" column) are
# "treated1fb" / "untreated1fb". Strip the trailing "fb" from the
# annotation side so the two align; then fail loudly if anything is
# still mismatched (silent mismatch is what caused the earlier
# all-NaN targets bug).
rownames(anno) <- sub("fb$", "", rownames(anno))
stopifnot(all(colnames(counts) %in% rownames(anno)))
anno_match <- anno[colnames(counts), , drop = FALSE]
write.csv(counts, gzfile(file.path(OUT_DIR, "pasilla_counts.csv.gz")))
write.csv(anno_match,  file.path(OUT_DIR, "pasilla_targets.csv"))
cat(sprintf("    shape: %d x %d\n", nrow(counts), ncol(counts)))

# ------------------------------------------------------------------
# Dataset 3: Yoruba HapMap (Pickrell 2010). tweeDEseqCountData
# ships an ExpressionSet `pickrell1` with raw counts + phenotype.
# ------------------------------------------------------------------
cat("\n[3/3] Yoruba HapMap (Pickrell 2010)\n")
suppressPackageStartupMessages(library(tweeDEseqCountData))
# The "pickrell1" data file in this package actually defines
# `pickrell1.eset`, not `pickrell1`. Load into an explicit environment
# and fetch the correctly-named object.
pk_env <- new.env()
data("pickrell1", package = "tweeDEseqCountData", envir = pk_env)
pickrell_eset <- get("pickrell1.eset", envir = pk_env)
yoruba_counts <- exprs(pickrell_eset)
yoruba_pheno  <- pData(pickrell_eset)
write.csv(yoruba_counts, gzfile(file.path(OUT_DIR, "yoruba_counts.csv.gz")))
write.csv(yoruba_pheno,  file.path(OUT_DIR, "yoruba_targets.csv"))
cat(sprintf("    shape: %d x %d\n",
            nrow(yoruba_counts), ncol(yoruba_counts)))

# ------------------------------------------------------------------
# Clean up: remove the set of packages that weren't installed before.
# Detaches loaded namespaces first so remove.packages can delete them.
# ------------------------------------------------------------------
post_install <- rownames(installed.packages())
added <- setdiff(post_install, pre_install)

# Detach anything we loaded so remove.packages can clean it up.
for (ns in intersect(loadedNamespaces(),
                     c("ALL", "pasilla", "tweeDEseqCountData", "Biobase",
                       added))) {
    try(unloadNamespace(ns), silent = TRUE)
}

if (length(added) > 0) {
    cat(sprintf("\n[cleanup] removing %d package(s) installed for setup: %s\n",
                length(added), paste(added, collapse = ", ")))
    remove.packages(added)
}

# Remove the Bioc downloaded-source cache if it was created in this run.
cache_candidates <- c(
    file.path(Sys.getenv("HOME"),
              "Library/Caches/org.R-project.R/R/BiocFileCache"),
    file.path(tempdir(), "downloaded_packages")
)
for (cache in cache_candidates) {
    if (dir.exists(cache)) {
        tarballs <- list.files(cache, pattern = "\\.tar\\.gz$",
                               full.names = TRUE)
        ours <- grep(paste0("(",
                            paste(added, collapse = "|"),
                            ")"), tarballs, value = TRUE)
        for (f in ours) {
            unlink(f)
        }
    }
}

# Verify final library state.
final <- rownames(installed.packages())
still_extra <- setdiff(final, pre_install)
if (length(still_extra) == 0) {
    cat("[cleanup] library restored to pre-run state.\n")
} else {
    cat(sprintf(
        "[cleanup] %d package(s) remain that were not in the pre-run snapshot: %s\n",
        length(still_extra), paste(still_extra, collapse = ", ")))
    cat("          Review whether these should be removed manually.\n")
}

cat(sprintf("\nDone. Data written to %s\n", normalizePath(OUT_DIR)))
