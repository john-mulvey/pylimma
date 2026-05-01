"""
One-time download of GSE60450 raw counts and sample metadata from GEO.

Downloads (a) the supplementary counts file (raw gene counts) and
(b) the GEO series matrix (sample metadata deposited by the study
authors). Sample labels are parsed from the series matrix, not
hard-coded, so provenance is clear.

Runs in pure Python - no R needed.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COUNTS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60450/suppl/"
    "GSE60450_Lactation-GenewiseCounts.txt.gz"
)
SM_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60450/matrix/GSE60450_series_matrix.txt.gz"
)


def _fetch_gz(url: str) -> bytes:
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _parse_series_matrix(text: str) -> dict[str, list[str]]:
    """
    Return a dict mapping series-matrix tag names (e.g.
    '!Sample_title') to a list of per-sample string values.

    Multiple lines with the same tag are preserved - the key becomes
    '<tag>__<n>' (1-indexed) for occurrences after the first.
    """
    out: dict[str, list[str]] = {}
    tag_counts: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("!Sample_"):
            continue
        parts = line.split("\t")
        tag = parts[0]
        vals = [p.strip().strip('"') for p in parts[1:]]
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        key = tag if tag_counts[tag] == 1 else f"{tag}__{tag_counts[tag]}"
        out[key] = vals
    return out


def _extract_char(char_lines: list[list[str]], prefix: str) -> list[str]:
    """Find the Sample_characteristics_ch1 row starting with `prefix: `."""
    for row in char_lines:
        if all(v.startswith(prefix + ":") for v in row):
            return [v.split(":", 1)[1].strip() for v in row]
    raise KeyError(f"no characteristics row with prefix {prefix!r}")


def main() -> None:
    # (a) Counts.
    counts_raw = _fetch_gz(COUNTS_URL)
    with gzip.open(io.BytesIO(counts_raw), "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    counts = df.set_index("EntrezGeneID").drop(columns=["Length"])
    # Raw column names are like "MCL1-DG_BC2CTUACXX_ACTTGA_L002_R1".
    # Use only the "MCL1-DG" prefix so we can join with series-matrix
    # metadata keyed on the same identifier.
    counts.columns = counts.columns.str.extract(r"(MCL1-[A-Z]+)", expand=False)
    if counts.columns.hasnans:
        raise RuntimeError(
            "Unable to parse GSE60450 column names - GEO may have "
            "changed the supplementary file format."
        )

    # (b) Series-matrix metadata.
    sm_raw = _fetch_gz(SM_URL)
    sm_text = gzip.decompress(sm_raw).decode("utf-8", errors="replace")
    sm = _parse_series_matrix(sm_text)

    # All Sample_characteristics_ch1 occurrences (as a list of rows).
    char_rows = [v for k, v in sm.items() if k.startswith("!Sample_characteristics_ch1")]
    stage = _extract_char(char_rows, "developmental stage")
    immuno = _extract_char(char_rows, "immunophenotype")

    # Sample IDs are in the first !Sample_description row as
    # "Sample name: MCL1-XX".
    desc_rows = [v for k, v in sm.items() if k.startswith("!Sample_description")]
    id_row = next(row for row in desc_rows if all(v.startswith("Sample name:") for v in row))
    ids = [v.split(":", 1)[1].strip() for v in id_row]

    stage_tidy = {
        "virgin": "virgin",
        "18.5 day pregnancy": "pregnant",
        "2 day lactation": "lactate",
    }
    immuno_tidy = {
        "basal cell population": "basal",
        "luminal cell population": "luminal",
    }

    targets = pd.DataFrame(
        {
            "sample_id": ids,
            "immunophenotype": [immuno_tidy.get(v, v) for v in immuno],
            "developmental_stage": [stage_tidy.get(v, v) for v in stage],
        }
    ).set_index("sample_id")
    targets["group"] = targets["immunophenotype"] + "." + targets["developmental_stage"]

    # Reindex targets to the column order of counts so downstream
    # consumers can zip them together.
    targets = targets.loc[counts.columns]

    # Write.
    counts_path = DATA_DIR / "gse60450_counts.csv.gz"
    counts.to_csv(counts_path, compression="gzip")
    (DATA_DIR / "gse60450_targets.csv").write_text(targets.to_csv())
    print(f"[write] {counts_path}  shape={counts.shape}")


if __name__ == "__main__":
    main()
