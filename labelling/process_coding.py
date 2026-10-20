from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from src.config import DATASETS_DIR
from evals_local.loaders import load_predictions


_DEFAULT_RUN_LIST = _REPO_ROOT / "checklists" / "mappings" / "run_names.json"
_TAXA_API_TASK = "taxa-with-api"

_CORE_FIELDS = ["UT", "title", "authors", "abstract",
                "source", "publication_year", "wos_categories", "doi"]

# Evals sub-tasks that are expected to cover every record in the dataset.
# Downstream levels (threats_l1/l2, ecosystems_biome/efg) run on a subset only.
_PRIMARY_EVALS_TASKS = {
    "driver", "geography", "study", "taxa",
    "threats_l0", "ecosystems_realm",
}

# Evals sub-tasks whose loaders filter batch outputs by a "{task}-*.jsonl" file
# prefix (multiple levels can share one run folder). Everything else is loaded
# via load_jsonl_folder, which reads *any* *.jsonl in the folder — so those runs
# only need at least one .jsonl, regardless of filename prefix.
_PREFIX_FILTERED_EVALS_TASKS = {
    "threats_l0", "threats_l1", "threats_l2",
    "ecosystems_realm", "ecosystems_biome", "ecosystems_efg",
}


@dataclass(frozen=True)
class LabelSpec:
    """One evals sub-task feeding one output partition file.

    evals_task:     task name understood by evals_local.loaders.load_predictions.
    colmap:         {prediction_column_from_evals: output_column_name}.
    required_files: if the sub-task's *.jsonl are absent, raise (True) or skip (False).
    """
    evals_task: str
    colmap: dict[str, str]
    required_files: bool = True

    @property
    def full_coverage(self) -> bool:
        return self.evals_task in _PRIMARY_EVALS_TASKS

    @property
    def prefix_filtered(self) -> bool:
        return self.evals_task in _PREFIX_FILTERED_EVALS_TASKS


# ── Task registry ─────────────────────────────────────────────────────────────
# Maps a --task value to the ordered list of evals sub-tasks that contribute
# label columns. Composite tasks (threats, ecosystems) fold multiple levels into
# one output file; optional downstream levels are skipped if their files are
# missing. Extend this dict as new task data lands.
TASK_SPECS: dict[str, list[LabelSpec]] = {
    "driver": [LabelSpec("driver", {"pred": "driver"})],

    # Composite: all three threat levels in one file (l2 is optional).
    "threats": [
        LabelSpec("threats_l0", {"pred_threat_l0": "pred_threat_l0"}),
        LabelSpec("threats_l1", {"pred_threat_l1": "pred_threat_l1"}),
        LabelSpec("threats_l2", {
                  "pred_threat_l2": "pred_threat_l2"}, required_files=False),
    ],
    "threats_l0": [LabelSpec("threats_l0", {"pred_threat_l0": "pred_threat_l0"})],
    "threats_l1": [LabelSpec("threats_l1", {"pred_threat_l1": "pred_threat_l1"})],
    "threats_l2": [LabelSpec("threats_l2", {"pred_threat_l2": "pred_threat_l2"})],

    "geography": [LabelSpec("geography", {
        "pred_regions": "pred_regions",
        "pred_subregions": "pred_subregions",
        "pred_countries": "pred_countries",
        "pred_locales": "locales",
        "pred_locale_coordinates": "locale_coordinates",
    })],

    # Composite: realm/biome/efg in one file (biome, efg optional).
    "ecosystems": [
        LabelSpec("ecosystems_realm", {"pred_realm": "realm"}),
        LabelSpec("ecosystems_biome", {
                  "pred_biome": "biome"}, required_files=False),
        LabelSpec("ecosystems_efg", {"pred_efg": "efg"}, required_files=False),
    ],
    "ecosystems_realm": [LabelSpec("ecosystems_realm", {"pred_realm": "realm"})],
    "ecosystems_biome": [LabelSpec("ecosystems_biome", {"pred_biome": "biome"})],
    "ecosystems_efg": [LabelSpec("ecosystems_efg", {"pred_efg": "efg"})],

    "study": [LabelSpec("study", {
        "pred_study_design": "pred_study_design",
        "pred_methods_data_collection": "pred_methods_data_collection",
        "pred_methods_analysis": "pred_methods_analysis",
        "pred_has_comparison": "pred_has_comparison",
        "pred_comparison_types": "pred_comparison_types",
    })],

    "taxa": [LabelSpec("taxa", {
        "pred_kingdom": "kingdom",
        "pred_phylum": "phylum",
        "pred_class": "class",
        "pred_order": "order",
        "pred_genus": "genus",
        "pred_species": "species",
    })],

    # Implemented by labelling/taxa_api.py rather than evals_local.loaders.
    _TAXA_API_TASK: [],
}

# List-valued core columns that need flattening for Excel/CSV output.
_LIST_CORE_FIELDS = ("authors", "wos_categories")
_DISTRIBUTION_CAP = 100


def _load_dataset(run_name: str) -> dict[str, dict]:
    """Return {UT: document} from the dataset JSON saved by the orchestrator."""
    path = DATASETS_DIR / f"{run_name}-dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found for run '{run_name}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))["documents"]


def _cell(val, *, preserve_lists: bool = False):
    """Serialize a value for Excel/CSV.

    Core metadata lists stay human-readable; label lists can be preserved as
    JSON arrays so downstream code can recover list[str] values losslessly.
    """
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if isinstance(val, list):
        if preserve_lists:
            return json.dumps(val, ensure_ascii=False)
        if all(v is None or isinstance(v, (str, int, float, bool)) for v in val):
            return "; ".join("" if v is None else str(v) for v in val)
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return val


def _summarise_label(series: pd.Series) -> dict:
    """Coverage + value distribution for one label column."""
    non_null = 0
    multi = 0
    counter: Counter = Counter()
    for val in series:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if isinstance(val, list):
            if len(val) == 0:
                continue
            non_null += 1
            if len(val) > 1:
                multi += 1
            for item in val:
                if isinstance(item, (dict, list)):
                    continue  # complex payloads not tallied
                counter[str(item)] += 1
        else:
            non_null += 1
            counter[str(val)] += 1

    return {
        "non_null": non_null,
        "null": int(len(series)) - non_null,
        "multi_value": multi,
        "distribution": dict(counter.most_common(_DISTRIBUTION_CAP)),
    }


def _load_run(task: str, run_name: str) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Load core fields + inferred label column(s) for one run.

    Returns:
        df           — one row per dataset record, core fields + label columns.
        label_cols   — output label column names, in registry order.
        run_summary  — coverage counts and per-label summaries for metadata.
    """
    docs = _load_dataset(run_name)
    dataset_uts = list(docs.keys())
    base = (
        pd.DataFrame([{f: docs[ut].get(f, "") for f in _CORE_FIELDS}
                      for ut in dataset_uts])
        if dataset_uts else pd.DataFrame(columns=_CORE_FIELDS)
    )
    base["UT"] = base["UT"].astype(str)
    dataset_ut_set = set(base["UT"])

    label_cols: list[str] = []
    label_summaries: dict[str, dict] = {}
    skipped_levels: list[str] = []

    for spec in TASK_SPECS[task]:
        try:
            pred = load_predictions(spec.evals_task, run_name)
        except FileNotFoundError:
            if spec.required_files:
                raise
            skipped_levels.append(spec.evals_task)
            print(
                f"  [skip] no '{spec.evals_task}' outputs in run '{run_name}'")
            continue

        if "custom_id" not in pred.columns:
            raise AssertionError(
                f"[{run_name}] '{spec.evals_task}' predictions have no custom_id column.")

        for pred_col in spec.colmap:
            if pred_col not in pred.columns:
                pred[pred_col] = None

        sub = pred[["custom_id", *spec.colmap.keys()]].copy()
        sub = sub.rename(columns={"custom_id": "UT", **spec.colmap})
        sub["UT"] = sub["UT"].astype(str)
        sub = sub.drop_duplicates("UT", keep="first")

        pred_ut_set = set(sub["UT"])
        unknown = pred_ut_set - dataset_ut_set
        if unknown:
            raise AssertionError(
                f"[{run_name}] '{spec.evals_task}': {len(unknown)} prediction UT(s) "
                f"absent from dataset — run_name/dataset mismatch. "
                f"Sample: {sorted(unknown)[:5]}"
            )
        if spec.full_coverage and pred_ut_set != dataset_ut_set:
            missing = dataset_ut_set - pred_ut_set
            print(
                f"  [warn] '{spec.evals_task}' covers {len(pred_ut_set):,}/"
                f"{len(dataset_ut_set):,} records; {len(missing):,} without output "
                "(possible incomplete batch)."
            )

        base = base.merge(sub, on="UT", how="left")
        out_cols = list(spec.colmap.values())
        label_cols.extend(out_cols)
        for out_col in out_cols:
            label_summaries[out_col] = _summarise_label(base[out_col])

    run_summary = {
        "total": len(base),
        "skipped_levels": skipped_levels,
        "labels": label_summaries,
    }
    return base, label_cols, run_summary


def _year_range(chunk: pd.DataFrame) -> tuple[int, int]:
    years = pd.to_numeric(chunk.get("publication_year"),
                          errors="coerce").dropna()
    if not len(years):
        return 0, 0
    return int(years.max()), int(years.min())


# ── Run-name resolution and missing-run detection ─────────────────────────────

def _load_run_names_from_list(path: Path, key: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Run list JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"Run list JSON must be an object at top level: {path}")
    if key not in data:
        available = ", ".join(sorted(str(k) for k in data.keys())) or "<none>"
        raise KeyError(
            f"Run key '{key}' not found in {path}. Available keys: {available}")
    run_names = data[key]
    if not isinstance(run_names, list) or not all(isinstance(v, str) for v in run_names):
        raise ValueError(
            f"Run list key '{key}' must contain a list of strings.")
    return run_names


def _resolve_run_names(args: argparse.Namespace) -> list[str]:
    run_names = list(args.run_names)
    if not run_names:
        run_list = args.run_list if args.run_list is not None else _DEFAULT_RUN_LIST
        default_run_key = "taxa" if args.task == _TAXA_API_TASK else args.task
        run_key = args.run_key if args.run_key is not None else default_run_key
        run_names = _load_run_names_from_list(run_list, run_key)

    if not run_names:
        raise ValueError(
            "No run names resolved. Provide them positionally or via --run-list.")

    seen: set[str] = set()
    duplicates = [r for r in run_names if r in seen or seen.add(r)]
    if duplicates:
        raise ValueError(
            f"Duplicate run names provided. Sample: {duplicates[:5]}")
    return run_names


def _missing_run_reasons(task: str, run_name: str) -> list[str]:
    reasons: list[str] = []

    dataset_path = DATASETS_DIR / f"{run_name}-dataset.json"
    if not dataset_path.exists():
        reasons.append(f"missing dataset: {dataset_path}")

    output_dir = _REPO_ROOT / "data" / "artifacts" / "batch_outputs" / run_name
    if not output_dir.exists():
        reasons.append(f"missing batch output dir: {output_dir}")
        return reasons

    if task == _TAXA_API_TASK:
        if not any(output_dir.glob("*.jsonl")):
            reasons.append(f"missing *.jsonl in: {output_dir}")
        return reasons

    for spec in TASK_SPECS[task]:
        if not spec.required_files:
            continue
        if spec.prefix_filtered:
            # Level-specific loaders require the "{task}-*.jsonl" naming.
            if not any(output_dir.glob(f"{spec.evals_task}-*.jsonl")):
                reasons.append(
                    f"missing '{spec.evals_task}-*.jsonl' in: {output_dir}")
        else:
            # Folder loaders read any *.jsonl regardless of filename prefix.
            if not any(output_dir.glob("*.jsonl")):
                reasons.append(f"missing *.jsonl in: {output_dir}")
    return reasons


def _collect_missing_runs(task: str, run_names: list[str]) -> dict[str, list[str]]:
    return {r: reasons for r in run_names
            if (reasons := _missing_run_reasons(task, r))}


def _print_missing_runs(task: str, run_names: list[str]) -> None:
    missing = _collect_missing_runs(task, run_names)
    present = len(run_names) - len(missing)
    print(f"Checked {len(run_names):,} run(s): {present:,} present, "
          f"{len(missing):,} missing/incomplete.")
    if not missing:
        return
    print("\nMissing or incomplete runs:")
    for run_name, reasons in missing.items():
        print(f"- {run_name}")
        for reason in reasons:
            print(f"  - {reason}")


def parse_args() -> argparse.Namespace:
    tasks = ", ".join(sorted(TASK_SPECS))
    parser = argparse.ArgumentParser(
        description="Infer task labels from coding batch outputs and write per-partition files."
    )
    parser.add_argument(
        "--task", required=True, choices=sorted(TASK_SPECS),
        help=f"Labelling task (any except screening). One of: {tasks}.",
    )
    parser.add_argument(
        "run_names", nargs="*",
        help="Run names (partitions). If omitted, read from --run-list under --run-key.",
    )
    parser.add_argument(
        "--run-list", type=Path, default=None,
        help=f"Run list JSON (default: {_DEFAULT_RUN_LIST.relative_to(_REPO_ROOT)}).",
    )
    parser.add_argument(
        "--run-key", default=None,
        help="Key to read from --run-list (default: the --task value).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Output folder name under data/ (default: coding-taxa for "
            "taxa-with-api; otherwise coding-<task>)."
        ),
    )
    parser.add_argument(
        "--print-missing", action="store_true",
        help="Print runs missing datasets or required batch outputs, then exit.",
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Skip runs missing datasets or required batch outputs and process the rest.",
    )
    api_group = parser.add_argument_group(
        "taxa-with-api options",
        "These options apply only when --task taxa-with-api.",
    )
    api_group.add_argument(
        "--api-workers", type=int, default=4,
        help="Concurrent GBIF request workers (default: 4).",
    )
    api_group.add_argument(
        "--api-rate-limit", type=float, default=8.0,
        help="Maximum request starts per second across workers (default: 8).",
    )
    api_group.add_argument(
        "--api-timeout", type=float, default=5.0,
        help="GBIF connect and read timeout in seconds (default: 5).",
    )
    api_group.add_argument(
        "--api-max-attempts", type=int, default=4,
        help="Maximum attempts for each transiently failing request (default: 4).",
    )
    api_group.add_argument(
        "--api-retry-delay", type=float, default=5.0,
        help="Delay between transient request attempts in seconds (default: 5).",
    )
    api_group.add_argument(
        "--api-user-agent",
        default="biodiversity-evidence-synthesis/taxa-with-api",
        help="User-Agent sent to GBIF.",
    )
    api_group.add_argument(
        "--force", action="store_true",
        help="Rebuild completed partition files (cached API matches are still reused).",
    )
    api_group.add_argument(
        "--refresh-api-cache", action="store_true",
        help="Ignore cached matches and request every taxon again.",
    )
    api_group.add_argument(
        "--cached-only",
        action="store_true",
        help=(
            "Disable all GBIF requests and fail before lookup if a required "
            "query is absent from the existing release-aware cache."
        ),
    )
    api_group.add_argument(
        "--taxa-group-config",
        default=str(
            _REPO_ROOT
            / "checklists"
            / "mappings"
            / "taxa_broad_groups.json"
        ),
        help="Broad-taxon hierarchy-rule JSON (default: in-place mapping).",
    )
    api_group.add_argument(
        "--retry-final-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry cached terminal API failures on a later invocation (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task
    run_names = _resolve_run_names(args)

    if args.print_missing:
        _print_missing_runs(task, run_names)
        return

    skipped_missing: dict[str, list[str]] = {}
    if args.skip_missing:
        skipped_missing = _collect_missing_runs(task, run_names)
        run_names = [r for r in run_names if r not in skipped_missing]
        print(f"Skipping {len(skipped_missing):,} missing/incomplete run(s); "
              f"processing {len(run_names):,} present run(s).")
        if not run_names:
            print("No present runs to process.")
            return

    output_dir_name = args.output_dir or (
        "coding-taxa" if task == _TAXA_API_TASK else f"coding-{task}"
    )
    output_base = _REPO_ROOT / "data" / output_dir_name
    all_dir = output_base / "all"

    if task == _TAXA_API_TASK:
        if args.api_workers <= 0:
            raise ValueError("--api-workers must be positive.")
        if args.api_rate_limit <= 0:
            raise ValueError("--api-rate-limit must be positive.")
        if args.api_timeout <= 0:
            raise ValueError("--api-timeout must be positive.")
        if args.api_max_attempts <= 0:
            raise ValueError("--api-max-attempts must be positive.")
        if args.api_retry_delay < 0:
            raise ValueError("--api-retry-delay must be non-negative.")
        if args.cached_only and args.refresh_api_cache:
            raise ValueError(
                "--cached-only cannot be combined with --refresh-api-cache."
            )

        from labelling.taxa_api import process_taxa_with_api

        process_taxa_with_api(
            run_names=run_names,
            datasets_directory=DATASETS_DIR,
            batch_outputs_directory=(
                _REPO_ROOT / "data" / "artifacts" / "batch_outputs"
            ),
            output_base=output_base,
            workers=args.api_workers,
            requests_per_second=args.api_rate_limit,
            timeout_seconds=args.api_timeout,
            max_attempts=args.api_max_attempts,
            retry_delay_seconds=args.api_retry_delay,
            user_agent=args.api_user_agent,
            force=args.force,
            refresh_cache=args.refresh_api_cache,
            retry_final_failures=args.retry_final_failures,
            skipped_missing=skipped_missing,
            group_config_path=Path(args.taxa_group_config),
            cached_only=args.cached_only,
        )
        return

    per_run_summaries: dict[str, dict] = {}
    partition_details: list[dict] = []
    all_frames: list[pd.DataFrame] = []
    seen_uts: set[str] = set()
    label_cols: list[str] = []

    output_base.mkdir(parents=True, exist_ok=True)

    for idx, run_name in enumerate(run_names, start=1):
        print(f"[{idx}/{len(run_names)}] Loading run '{run_name}' (task={task}) ...")
        df, run_label_cols, summary = _load_run(task, run_name)
        label_cols = run_label_cols or label_cols

        run_uts = set(df["UT"].astype(str))
        overlap = run_uts & seen_uts
        if overlap:
            raise AssertionError(
                f"Duplicate UTs across runs: {len(overlap)} record(s) from '{run_name}' "
                f"already seen. Sample: {sorted(overlap)[:5]}"
            )
        seen_uts |= run_uts

        df["_source_run"] = run_name
        df["_partition"] = idx
        per_run_summaries[run_name] = summary

        # Write this partition's Excel file.
        col_order = _CORE_FIELDS + run_label_cols + ["_source_run"]
        excel_df = df[[c for c in col_order if c in df.columns]].copy()
        for col in excel_df.columns:
            excel_df[col] = excel_df[col].map(
                lambda val, c=col: _cell(val, preserve_lists=c in run_label_cols)
            )

        max_year, min_year = _year_range(df)
        filename = f"{task}_{idx:02d}_{max_year}-{min_year}.xlsx"
        part_dir = output_base / str(idx)
        part_dir.mkdir(parents=True, exist_ok=True)
        dest = part_dir / filename
        excel_df.to_excel(dest, index=False)

        all_frames.append(df)
        partition_details.append({
            "partition": idx,
            "source_run": run_name,
            "file": str(dest.relative_to(_REPO_ROOT)),
            "records": len(df),
            "skipped_levels": summary["skipped_levels"],
            "year_range": {"min": min_year, "max": max_year},
        })
        print(f"    {len(df):,} records → {dest.relative_to(_REPO_ROOT)}")

    if not all_frames:
        print("No records found. Nothing to write.")
        return

    # Combined all-records CSV.
    combined = pd.concat(all_frames, ignore_index=True)
    combined["UT"] = combined["UT"].astype(str)
    all_dir.mkdir(parents=True, exist_ok=True)
    all_path = all_dir / f"{task}_all.csv"
    csv_order = _CORE_FIELDS + label_cols + ["_partition", "_source_run"]
    csv_df = combined[[c for c in csv_order if c in combined.columns]].copy()
    for col in csv_df.columns:
        csv_df[col] = csv_df[col].map(
            lambda val, c=col: _cell(val, preserve_lists=c in label_cols)
        )
    csv_df.to_csv(all_path, index=False)
    print(f"\nCombined table written to {all_path.relative_to(_REPO_ROOT)}")

    # Aggregate label distributions across all runs.
    label_totals = {col: _summarise_label(combined[col])
                    for col in label_cols if col in combined.columns}

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "output_dir": str(output_base.relative_to(_REPO_ROOT)),
        "run_names": run_names,
        "run_list": str(args.run_list) if args.run_list else str(_DEFAULT_RUN_LIST),
        "run_key": args.run_key or task,
        "skip_missing": args.skip_missing,
        "skipped_missing": skipped_missing,
        "label_columns": label_cols,
        "outputs": {
            "all": str(all_path.relative_to(_REPO_ROOT)),
            "partitions_dir": str(output_base.relative_to(_REPO_ROOT)),
        },
        "totals": {
            "records": len(combined),
            "partitions": len(partition_details),
            "labels": label_totals,
        },
        "per_run": per_run_summaries,
        "partitions": partition_details,
    }
    meta_path = output_base / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Metadata written to {meta_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
