"""Build the curated GBIF Backbone extract from the public 2023-08-28 release.

The taxa preparation (described-species benchmark), the taxonomic-hierarchy
figure, and the name-lookup cache read a CSV extract of the GBIF Backbone
Taxonomy. This module builds that extract from the archived public release:

    GBIF Secretariat (2023). GBIF Backbone Taxonomy, release 2023-08-28.
    https://doi.org/10.15468/39omei
    https://hosted-datasets.gbif.org/datasets/backbone/2023-08-28/backbone.zip

Usage (from the repository root, after downloading the archive to
``data/gbif/raw/backbone-2023-08-28.zip``):

    python -m data_helpers.sources.build_gbif_extract

The archive is verified against its SHA-256. ``Taxon.tsv`` is unquoted
tab-separated text and is streamed unchanged in row order (no filtering;
downstream code selects accepted species-rank rows itself), keeping the columns
the repository reads; ``eml.xml`` is copied beside it. The result is checked
against recorded row counts and the extract's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE = "2023-08-28"
ARCHIVE_URL = (
    "https://hosted-datasets.gbif.org/datasets/backbone/2023-08-28/backbone.zip"
)
ARCHIVE_SHA256 = "2897887cc7fb6fe27188c5b0fa5ce2b14762b43c61ceecd1e3dfad25083f5770"
ARCHIVE_SIZE = 971_465_842
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "gbif" / "raw" / "backbone-2023-08-28.zip"
DEFAULT_EXTRACT = REPO_ROOT / "data" / "gbif" / "curated" / "gbif_curated.csv"
DEFAULT_EML = REPO_ROOT / "data" / "gbif" / "extracted" / "eml.xml"

# Columns read by benchmark.py, hierarchy.py, and build_gbif.py, in Taxon.tsv order.
EXTRACT_COLUMNS = (
    "taxonID",
    "scientificName",
    "canonicalName",
    "taxonRank",
    "taxonomicStatus",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
)

# Complete extract, as written by build_extract with the pinned pyarrow.
EXTRACT_SHA256 = "62616875cf44666bd8a53a76c873b7df29a8a2270617a56678a85542d8ee0e4d"
EXPECTED_SOURCE_AUDIT = {
    "GBIF backbone rows scanned": 7_746_724,
    "Species-rank rows": 4_994_322,
    "Accepted species-rank rows": 2_614_593,
    "Accepted species rows missing taxonID": 0,
    "Accepted species complete through kingdom": 2_614_593,
    "Unique kingdom paths": 9,
    "Accepted species complete through phylum": 2_601_849,
    "Unique phylum paths": 271,
    "Accepted species complete through class": 2_498_498,
    "Unique class paths": 706,
    "Accepted species complete through order": 2_418_924,
    "Unique order paths": 2_548,
}

class GbifExtractError(RuntimeError):
    """Raised when the archive or the rebuilt extract fails verification."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(archive: Path) -> None:
    """Check the archive's size and SHA-256 against the recorded release."""
    if not archive.exists():
        raise GbifExtractError(
            f"GBIF archive not found at {archive}. Download {ARCHIVE_URL}."
        )
    size = archive.stat().st_size
    if size != ARCHIVE_SIZE:
        raise GbifExtractError(f"Archive size {size:,} != expected {ARCHIVE_SIZE:,}.")
    observed = sha256_file(archive)
    if observed != ARCHIVE_SHA256:
        raise GbifExtractError(f"Archive SHA-256 {observed} != expected {ARCHIVE_SHA256}.")


def _copy_eml(zf: zipfile.ZipFile, eml_path: Path) -> None:
    eml_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open("eml.xml") as source, eml_path.open("wb") as target:
        shutil.copyfileobj(source, target)


def build_extract(
    archive: Path = DEFAULT_ARCHIVE,
    extract_path: Path = DEFAULT_EXTRACT,
    eml_path: Path = DEFAULT_EML,
    *,
    verify: bool = True,
    block_size: int = 64 << 20,
) -> dict[str, int]:
    """Stream every ``Taxon.tsv`` row into the CSV extract and copy ``eml.xml``."""
    if verify:
        verify_archive(archive)
    extract_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = extract_path.with_name(f".{extract_path.name}.part")
    rows = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            _copy_eml(zf, eml_path)
            with zf.open("Taxon.tsv") as source:
                # Taxon.tsv is unquoted; quotes inside names are literal text and
                # must not start quoted fields.
                reader = csv.open_csv(
                    source,
                    read_options=csv.ReadOptions(block_size=block_size, use_threads=False),
                    parse_options=csv.ParseOptions(delimiter="\t", quote_char=False),
                    convert_options=csv.ConvertOptions(
                        include_columns=list(EXTRACT_COLUMNS),
                        column_types={column: pa.string() for column in EXTRACT_COLUMNS},
                        strings_can_be_null=True,
                    ),
                )
                schema = pa.schema([(column, pa.string()) for column in EXTRACT_COLUMNS])
                # Own the file handle: a path-opened sink can stay locked on Windows.
                with temporary.open("wb") as sink:
                    with csv.CSVWriter(sink, schema) as writer:
                        for batch in reader:
                            writer.write_batch(batch.select(list(EXTRACT_COLUMNS)))
                            rows += batch.num_rows
        temporary.replace(extract_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": rows}


def _audit_problems(observed, expected: dict[str, int]) -> list[str]:
    values = dict(zip(observed["metric"], observed["value"]))
    problems = [
        f"{metric}: observed={int(values[metric]):,} expected={value:,}"
        for metric, value in expected.items()
        if metric in values and int(values[metric]) != value
    ]
    problems.extend(
        f"{metric}: not reported" for metric in expected if metric not in values
    )
    return problems


def validate_gbif_extract(extract_path: Path = DEFAULT_EXTRACT) -> list[str]:
    """Check the extract against its recorded counts and SHA-256; return mismatches."""
    from data_helpers.analysis.taxa.hierarchy import stream_gbif_rank_counts

    _, audit = stream_gbif_rank_counts(extract_path)
    problems = _audit_problems(audit, EXPECTED_SOURCE_AUDIT)
    observed_sha = sha256_file(extract_path)
    if observed_sha != EXTRACT_SHA256:
        problems.append(
            f"extract SHA-256 {observed_sha} != recorded {EXTRACT_SHA256} "
            "(a different pyarrow version can change CSV formatting; "
            "the row counts above are the substantive check)"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--eml-output", type=Path, default=DEFAULT_EML)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not check the extract against its recorded counts.",
    )
    args = parser.parse_args(argv)
    output = args.output or DEFAULT_EXTRACT
    try:
        summary = build_extract(args.archive, output, args.eml_output)
        print(f"Wrote {summary['rows']:,} rows to {output}")
        if not args.skip_validation:
            problems = validate_gbif_extract(output)
            if problems:
                print("Validation FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
                return 1
            print("Validation passed: every recorded count matches.")
    except GbifExtractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
