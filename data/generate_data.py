"""
Load the four benchmark datasets from the committed CSVs.

Data in ``data/`` is the source of truth - it was extracted
once from the Bioconductor data packages (ALL, pasilla,
tweeDEseqCountData) plus GEO (GSE60450) by the maintainer. See
``_setup_datasets.R`` and ``_setup_gse60450.py`` for the extraction
scripts, and ``DATA_PROVENANCE.md`` for the dataset provenance table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The repo should ship these CSVs - if "
            f"yours has been stripped, re-run _setup_datasets.R and "
            f"_setup_gse60450.py."
        )
    return path


def load_all(*, small: bool = False) -> dict:
    """
    Chiaretti ALL (2004): RMA-normalised HG-U95Av2 expression plus
    clinical phenotype. 12,625 genes x 128 samples.
    """
    prefix = "all_small" if small else "all"
    expr = pd.read_csv(_require(DATA_DIR / f"{prefix}_expr.csv.gz"), index_col=0)
    targets = pd.read_csv(_require(DATA_DIR / f"{prefix}_targets.csv"), index_col=0)
    return {"name": prefix, "expr": expr, "targets": targets}


def load_gse60450() -> dict:
    """Mouse mammary gland RNA-seq (Fu et al. 2015). Raw counts."""
    counts = pd.read_csv(_require(DATA_DIR / "gse60450_counts.csv.gz"), index_col=0)
    targets = pd.read_csv(_require(DATA_DIR / "gse60450_targets.csv"), index_col=0)
    return {"name": "gse60450", "counts": counts, "targets": targets}


def load_pasilla() -> dict:
    """Drosophila RNA-seq gene-level counts (pasilla Bioc package)."""
    counts = pd.read_csv(_require(DATA_DIR / "pasilla_counts.csv.gz"), index_col=0)
    targets = pd.read_csv(_require(DATA_DIR / "pasilla_targets.csv"), index_col=0)
    return {"name": "pasilla", "counts": counts, "targets": targets}


def load_yoruba() -> dict:
    """Yoruba HapMap RNA-seq (Pickrell 2010). Raw counts, 69 individuals."""
    counts = pd.read_csv(_require(DATA_DIR / "yoruba_counts.csv.gz"), index_col=0)
    targets = pd.read_csv(_require(DATA_DIR / "yoruba_targets.csv"), index_col=0)
    return {"name": "yoruba", "counts": counts, "targets": targets}


def build_two_group_design(labels) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-group one-contrast design: columns for each level (sorted),
    contrast = level2 minus level1.
    """
    cats = pd.Categorical(labels)
    X = pd.get_dummies(cats, drop_first=False).astype(float).values
    C = np.zeros((X.shape[1], 1))
    C[0, 0] = -1.0
    C[1, 0] = 1.0
    return X, C


if __name__ == "__main__":
    for loader in (load_all, load_gse60450, load_pasilla, load_yoruba):
        d = loader()
        matrix_key = "expr" if "expr" in d else "counts"
        m = d[matrix_key]
        print(f"{d['name']:12s}  {matrix_key:8s}  shape {m.shape}")
