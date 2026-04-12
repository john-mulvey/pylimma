#!/usr/bin/env Rscript
# Time R limma's pipelines on the same four reference datasets used by
# run_python.py. Writes results/r_<YYYYMMDD>_<platform>.json with a
# schema compatible with run_python.py, so run_benchmarks.ipynb
# consumes both without branching.
#
# The CSVs in data/ are the SAME files Python reads - no per-runtime
# preprocessing, no drift.
#
# Reproducibility:
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

suppressPackageStartupMessages({
    library(limma)
    library(edgeR)
    library(jsonlite)
})

N_REPS      <- 5
DATA_DIR    <- "data"
RESULTS_DIR <- "results"
dir.create(RESULTS_DIR, showWarnings = FALSE)


time_and_memory <- function(expr_fn) {
    gc(reset = TRUE)
    t <- system.time(val <- expr_fn())[["elapsed"]]
    mem <- sum(gc()[, 2]) * 1024 * 1024        # Mb -> bytes (approx)
    list(elapsed = t, peak_rss_bytes = mem, val = val)
}


repeat_fn <- function(label, fn) {
    fn()                                       # warm up
    elapsed <- numeric(N_REPS); peak <- numeric(N_REPS)
    for (i in seq_len(N_REPS)) {
        r <- time_and_memory(fn)
        elapsed[i] <- r$elapsed
        peak[i]    <- r$peak_rss_bytes
    }
    cat(sprintf("  %-40s median=%.3fs min=%.3fs max=%.3fs\n",
                label, median(elapsed), min(elapsed), max(elapsed)))
    list(elapsed_seconds = elapsed, peak_rss_bytes = peak)
}


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

pipeline_a <- function(expr, design, contrasts) {
    fit <- lmFit(expr, design)
    fit <- contrasts.fit(fit, contrasts = contrasts)
    fit <- eBayes(fit)
    topTable(fit, coef = 1, number = Inf)
}

pipeline_b <- function(counts, design, contrasts) {
    v   <- voom(counts, design)
    fit <- lmFit(v$E, design, weights = v$weights)
    fit <- contrasts.fit(fit, contrasts = contrasts)
    fit <- eBayes(fit)
    topTable(fit, coef = 1, number = Inf)
}

pipeline_c <- function(counts, design, contrasts, gene_sets) {
    v   <- voom(counts, design)
    fit <- lmFit(v$E, design, weights = v$weights)
    fit <- contrasts.fit(fit, contrasts = contrasts)
    fit <- eBayes(fit)
    camera(v$E, index = gene_sets, design = design, contrast = contrasts[, 1])
}

pipeline_d_splicing <- function(expr, design, geneid) {
    fit <- lmFit(expr, design)
    fit <- eBayes(fit)
    ds  <- diffSplice(fit, geneid = geneid)
    topSplice(ds, number = Inf)
}


two_group <- function(labels) {
    g <- factor(labels)
    X <- model.matrix(~0 + g)
    C <- matrix(0, ncol = 1, nrow = ncol(X))
    C[1, 1] <- -1; C[2, 1] <- 1
    list(design = X, contrasts = C)
}


# Gzipped CSV reader.
read_gz <- function(path) {
    as.matrix(read.csv(gzfile(path), row.names = 1))
}


results <- list(
    runtime = "r",
    r_version = R.version.string,
    limma_version = as.character(packageVersion("limma")),
    edger_version = as.character(packageVersion("edgeR")),
    platform = sprintf("%s-%s-%s", Sys.info()["sysname"],
                       Sys.info()["release"], Sys.info()["machine"]),
    n_reps = N_REPS,
    thread_counts = list(
        OMP_NUM_THREADS      = Sys.getenv("OMP_NUM_THREADS"),
        OPENBLAS_NUM_THREADS = Sys.getenv("OPENBLAS_NUM_THREADS"),
        MKL_NUM_THREADS      = Sys.getenv("MKL_NUM_THREADS")
    ),
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    datasets  = list()
)


# ---------------------------------------------------------------------------
# ALL (Chiaretti)
# ---------------------------------------------------------------------------
run_all <- function(small) {
    key <- if (small) "all_small" else "all"
    expr_f <- file.path(DATA_DIR, sprintf("%s_expr.csv.gz", key))
    targ_f <- file.path(DATA_DIR, sprintf("%s_targets.csv", key))
    if (!file.exists(expr_f)) { cat(sprintf("SKIP %s\n", key)); return() }
    expr    <- read_gz(expr_f)
    targets <- read.csv(targ_f, row.names = 1)
    bt <- substr(as.character(targets$BT), 1, 1)        # "B" or "T"
    d  <- two_group(bt)
    cat(sprintf("[%s] shape=%dx%d\n", key, nrow(expr), ncol(expr)))
    results$datasets[[key]] <<- list(
        shape = dim(expr),
        pipeline_a = repeat_fn("pipeline_a",
                               function() pipeline_a(expr, d$design, d$contrasts))
    )
}

run_all(small = TRUE)
run_all(small = FALSE)


# ---------------------------------------------------------------------------
# GSE60450
# ---------------------------------------------------------------------------
f <- file.path(DATA_DIR, "gse60450_counts.csv.gz")
if (file.exists(f)) {
    counts  <- read_gz(f)
    targets <- read.csv(file.path(DATA_DIR, "gse60450_targets.csv"),
                        row.names = 1)
    celltype <- sapply(strsplit(as.character(targets$group), "\\."),
                       function(x) x[1])
    d <- two_group(celltype)
    cat(sprintf("[gse60450] shape=%dx%d\n", nrow(counts), ncol(counts)))
    set.seed(7)
    gene_sets <- lapply(seq_len(50),
                        function(i) sample.int(nrow(counts), 30))
    names(gene_sets) <- paste0("set_", seq_len(50))
    log2_counts <- log2(counts + 1)
    results$datasets$gse60450 <- list(
        shape = dim(counts),
        pipeline_a = repeat_fn("pipeline_a",
                               function() pipeline_a(log2_counts, d$design, d$contrasts)),
        pipeline_b = repeat_fn("pipeline_b (voom)",
                               function() pipeline_b(counts, d$design, d$contrasts)),
        pipeline_c = repeat_fn("pipeline_c (voom+camera)",
                               function() pipeline_c(counts, d$design, d$contrasts, gene_sets))
    )
} else cat("SKIP gse60450\n")


# ---------------------------------------------------------------------------
# Yoruba
# ---------------------------------------------------------------------------
f <- file.path(DATA_DIR, "yoruba_counts.csv.gz")
if (file.exists(f)) {
    counts  <- read_gz(f)
    targets <- read.csv(file.path(DATA_DIR, "yoruba_targets.csv"),
                        row.names = 1)
    d <- two_group(targets$gender)
    cat(sprintf("[yoruba] shape=%dx%d\n", nrow(counts), ncol(counts)))
    log2_counts <- log2(counts + 1)
    results$datasets$yoruba <- list(
        shape = dim(counts),
        pipeline_a = repeat_fn("pipeline_a",
                               function() pipeline_a(log2_counts, d$design, d$contrasts)),
        pipeline_b = repeat_fn("pipeline_b (voom)",
                               function() pipeline_b(counts, d$design, d$contrasts))
    )
} else cat("SKIP yoruba\n")


# ---------------------------------------------------------------------------
# Pasilla
# ---------------------------------------------------------------------------
f <- file.path(DATA_DIR, "pasilla_counts.csv.gz")
if (file.exists(f)) {
    counts  <- read_gz(f)
    targets <- read.csv(file.path(DATA_DIR, "pasilla_targets.csv"),
                        row.names = 1)
    d <- two_group(targets$condition)
    geneid <- as.character(seq_len(nrow(counts)) %/% 5)
    cat(sprintf("[pasilla] shape=%dx%d\n", nrow(counts), ncol(counts)))
    log2_counts <- log2(counts + 1)
    results$datasets$pasilla <- list(
        shape = dim(counts),
        pipeline_d = repeat_fn("pipeline_d (splicing)",
                               function() pipeline_d_splicing(log2_counts, d$design, geneid))
    )
} else cat("SKIP pasilla\n")


fname <- sprintf("r_%s_%s.json",
                 format(Sys.time(), "%Y%m%d"),
                 tolower(Sys.info()["sysname"]))
write_json(results, file.path(RESULTS_DIR, fname),
           pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("\nWrote %s\n", file.path(RESULTS_DIR, fname)))
