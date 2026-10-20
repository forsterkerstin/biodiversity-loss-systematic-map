# World Bank collection notes

Generated: 2026-07-16T15:55:46+00:00

## Scope collected

- World Bank API source registry: 71 sources (70 reporting data availability).
- World Bank country registry: 295 records, comprising 217 economies and 78 aggregates.
- World Development Indicators metadata: 1,498 source-2 indicators.
- Raw WDI indicator API records: 1,510; exact duplicate metadata rows removed: 12.
- WDI source last-updated value reported by the API: 2026-07-13.
- The WDI bulk file covered 1960–2025 and produced 9,011,494 non-null observations across 398,468 entity-indicator rows.

## IPBES matching

- IPBES records inspected: 257.
- Unique nonblank IPBES ISO3 codes: 249.
- Blank IPBES ISO3 records: 8.
- Records matched to a World Bank economy: 216.
- Records without a World Bank economy match: 41.
- Match statuses: `{"exact_iso3": 215, "needs_review": 7, "no_wb_economy": 34, "wb_specific_code": 1}`.

Unmatched ISO3 territories:

```text
AIA ALA ATA ATF BES BLM BVT CCK COK CXR ESH FLK GGY GLP GUF HMD IOT JEY MSR MTQ MYT NFK NIU PCN REU SGS SHN SJM SPM TKL TWN UMI VAT WLF
```

Special geographies still requiring an explicit analytical decision:

```text
United States Minor Outlying Islands (Navassa), Clipperton Island, Hawaii (USA), Paracel Islands, Spratly Islands, Akrotiri and Dhekelia, Northern Cyprus
```

Kosovo is mapped to the World Bank-specific code `XKX`. No parent-economy
inheritance was inferred for territories, dependencies, or disputed areas.

## Missingness policy

- Blank WDI cells are not imputed or gap-filled.
- Only non-null values are stored in the observation fact table.
- `coverage.parquet` records all year cells inspected and counts both non-null
  and missing cells for every entity-indicator row.
- World Bank aggregates are retained and marked explicitly; they are not
  mistaken for ISO3 countries based on code length.
- Current and historical income/lending classifications are stored separately.
- Historical income groups are not reconstructed from revised GNI observations.

## Reproducibility

`manifest.json` records the exact source URLs, SHA-256 hashes, upstream update
dates, row counts, and generated artifacts. The raw versioned downloads are
retained under the snapshot's `raw/` directory.
