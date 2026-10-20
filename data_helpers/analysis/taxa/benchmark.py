"""Shared broad-group rules and the GBIF described-diversity benchmark.

Broad groups are consumed from the per-taxon hierarchy assignments created by
``taxa-with-api``. The benchmark side counts accepted species-rank GBIF rows
into the same broad groups so publication attention can be compared against
described diversity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from labelling.taxa_groups import (
    TaxaGroupConfigError,
    assign_taxon_group,
    load_group_config,
)

CHORDATES = "Chordates"
NON_CHORDATE_ANIMALS = "Non-chordate animals"
OTHER_GROUP = "Other"
UNRESOLVED_GROUP = "Unresolved"


class TaxaDriverError(ValueError):
    """Raised when an input violates the taxa-driver analysis contract."""


@dataclass(frozen=True)
class DescribedDiversity:
    """Accepted species-rank GBIF benchmark and its construction audit."""

    counts: pd.DataFrame
    audit: pd.DataFrame


def load_broad_group_mapping(path: str | Path) -> dict[str, Any]:
    """Load the same hierarchy-rule config used by ``taxa-with-api``."""
    try:
        mapping = load_group_config(path)
    except (OSError, TaxaGroupConfigError) as exc:
        raise TaxaDriverError(str(exc)) from exc
    required = {CHORDATES, NON_CHORDATE_ANIMALS, OTHER_GROUP, UNRESOLVED_GROUP}
    missing = required.difference(mapping["group_order"])
    if missing:
        raise TaxaDriverError(
            f"Broad-group mapping is missing required groups: {sorted(missing)}"
        )
    return mapping


def _lineage_group(
    kingdom: Any,
    phylum: Any,
    taxon_class: Any,
    mapping: dict[str, Any],
) -> str:
    """Map one GBIF species lineage using the same configured hierarchy."""
    classification = [
        {"rank": rank, "name": str(value)}
        for rank, value in (
            ("KINGDOM", kingdom),
            ("PHYLUM", phylum),
            ("CLASS", taxon_class),
        )
        if not pd.isna(value) and str(value).strip()
    ]
    assignment = assign_taxon_group(
        raw_rank="species",
        raw_name="GBIF benchmark species",
        match_status="exact",
        classification=classification,
        config=mapping,
    )
    return assignment.group or mapping["unresolved_group"]


def build_described_diversity_benchmark(
    gbif_path: str | Path,
    mapping: dict[str, Any],
    *,
    block_size: int = 16 << 20,
) -> DescribedDiversity:
    """Count accepted species-rank GBIF backbone rows by broad group.

    The large CSV is streamed with PyArrow and only the rank, status, and
    class/phylum lineage columns are materialized.
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as csv
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise TaxaDriverError(
            "PyArrow is required to stream the GBIF benchmark."
        ) from exc

    columns = [
        "taxonID",
        "taxonRank",
        "taxonomicStatus",
        "kingdom",
        "phylum",
        "class",
    ]
    reader = csv.open_csv(
        gbif_path,
        read_options=csv.ReadOptions(block_size=block_size, use_threads=False),
        parse_options=csv.ParseOptions(newlines_in_values=True),
        convert_options=csv.ConvertOptions(
            include_columns=columns,
            column_types={column: pa.string() for column in columns},
            strings_can_be_null=True,
        ),
    )

    total_rows = 0
    species_rank_rows = 0
    accepted_species_rows = 0
    accepted_missing_taxon_id = 0
    lineage_counts: Counter[tuple[Any, Any, Any]] = Counter()
    for batch in reader:
        total_rows += batch.num_rows
        ranks = pc.utf8_lower(batch["taxonRank"])
        statuses = pc.utf8_lower(batch["taxonomicStatus"])
        species_mask = pc.equal(ranks, "species")
        species_rank_rows += int(
            pc.sum(pc.cast(pc.fill_null(species_mask, False), pa.int64())).as_py()
        )
        accepted_mask = pc.and_(species_mask, pc.equal(statuses, "accepted"))
        accepted = batch.filter(accepted_mask)
        accepted_species_rows += accepted.num_rows
        if accepted.num_rows == 0:
            continue
        accepted_missing_taxon_id += int(
            pc.sum(pc.cast(pc.is_null(accepted["taxonID"]), pa.int64())).as_py()
        )
        lineage = accepted.select(["kingdom", "phylum", "class"]).to_pandas()
        for key, count in lineage.value_counts(dropna=False).items():
            normalized = tuple(None if pd.isna(value) else value for value in key)
            lineage_counts[normalized] += int(count)

    group_counts: Counter[str] = Counter()
    for (kingdom, phylum, taxon_class), count in lineage_counts.items():
        group_counts[_lineage_group(kingdom, phylum, taxon_class, mapping)] += count

    if sum(group_counts.values()) != accepted_species_rows:
        raise TaxaDriverError(
            "GBIF broad-group counts do not sum to accepted species rows."
        )
    counts = pd.DataFrame(
        {
            "broad_group": mapping["group_order"],
            "described_species_count": [
                int(group_counts[group]) for group in mapping["group_order"]
            ],
        }
    )
    counts["described_species_share"] = (
        counts["described_species_count"] / accepted_species_rows
    )
    counts["described_species_share_pct"] = counts["described_species_share"] * 100
    audit = pd.DataFrame(
        {
            "metric": [
                "GBIF backbone rows scanned",
                "Species-rank rows",
                "Accepted species-rank rows",
                "Accepted species rows missing taxonID",
                "Broad-group benchmark total",
            ],
            "value": [
                total_rows,
                species_rank_rows,
                accepted_species_rows,
                accepted_missing_taxon_id,
                int(counts["described_species_count"].sum()),
            ],
        }
    )
    return DescribedDiversity(counts=counts, audit=audit)
