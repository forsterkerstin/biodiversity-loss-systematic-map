from __future__ import annotations
from src.config import DATASETS_DIR
from evals_local.loaders import load_screening_predictions

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))


_CORE_FIELDS = ["UT", "title", "authors", "abstract",
                "source", "publication_year", "wos_categories", "doi"]
_STAGE_NUMERIC_COLS = ["s1_r", "s2_r", "s3_r", "s4_r"]
_STAGE_LABEL_COLS = ["s1_bio", "s2_dir", "s3_drivers", "s4_link"]


def _load_dataset(run_name: str) -> dict[str, dict]:
    """Return {UT: document} from the dataset JSON saved by orchestrator."""
    path = DATASETS_DIR / f"{run_name}-dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found for run '{run_name}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))["documents"]


def _load_run(run_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Load predictions and dataset for one run.

    Returns:
        all_df       — DataFrame with _CORE_FIELDS + prediction/stage cols for all records.
        run_summary  — counts for metadata.

    Raises AssertionError if prediction UTs and dataset UTs don't match exactly.
    """
    pred_df = load_screening_predictions(run_name)
    docs = _load_dataset(run_name)

    pred_uts = set(pred_df["UT"].astype(str))
    dataset_uts = set(docs.keys())
    missing_in_dataset = pred_uts - dataset_uts
    missing_in_preds = dataset_uts - pred_uts

    if missing_in_dataset or missing_in_preds:
        parts = []
        if missing_in_dataset:
            parts.append(
                f"{len(missing_in_dataset)} prediction UTs absent from dataset")
        if missing_in_preds:
            parts.append(
                f"{len(missing_in_preds)} dataset UTs absent from predictions")
        raise AssertionError(
            f"[{run_name}] Join mismatch — {'; '.join(parts)}. "
            "Ensure run_name matches the dataset that was used for this run."
        )

    # Stage-failure breakdown (first failing stage per record)
    s1, s2, s3, s4 = (pred_df[c] for c in ["s1_r", "s2_r", "s3_r", "s4_r"])
    no_output = int(pred_df["custom_id"].isna().sum())
    stage_failures = {
        "stage_1": int((s1 == 0).sum()),
        "stage_2": int(((s1 == 1) & (s2 == 0)).sum()),
        "stage_3": int(((s1 == 1) & (s2 == 1) & (s3 == 0)).sum()),
        "stage_4": int(((s1 == 1) & (s2 == 1) & (s3 == 1) & (s4 == 0)).sum()),
    }

    pred_df["UT"] = pred_df["UT"].astype(str)
    eligible_mask = pred_df["pred"] == '["ELIGIBLE"]'

    # Core doc fields from dataset, ordered exactly like the prediction rows.
    core_docs = [
        {field: docs[ut].get(field, "") for field in _CORE_FIELDS}
        for ut in pred_df["UT"]
    ]
    core_df = pd.DataFrame(core_docs) if core_docs else pd.DataFrame(
        columns=_CORE_FIELDS)

    # Prediction and stage cols to carry for auditability.
    stage_label_cols = [c for c in _STAGE_LABEL_COLS if c in pred_df.columns]
    prediction_cols = [
        c
        for c in ["custom_id", "model", "created_at", "raw_output", "pred"]
        if c in pred_df.columns
    ]
    pred_cols = prediction_cols + _STAGE_NUMERIC_COLS + stage_label_cols
    pred_details_df = pred_df[pred_cols].reset_index(drop=True)
    all_df = pd.concat([core_df, pred_details_df], axis=1)
    all_df["_source_run"] = run_name
    all_df["is_eligible"] = eligible_mask.reset_index(drop=True)
    all_df["eligibility"] = all_df["is_eligible"].map(
        {True: "ELIGIBLE", False: "NOT_ELIGIBLE"})

    run_summary = {
        "total": len(pred_df),
        "eligible": int(eligible_mask.sum()),
        "not_eligible": int((~eligible_mask).sum()),
        "no_output": no_output,
        "stage_failures": stage_failures,
    }
    return all_df, run_summary


def _list_to_str(val) -> str:
    if isinstance(val, list):
        return "; ".join(str(v) for v in val)
    return str(val) if val is not None else ""


def _csv_cell(val) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract eligible records from screening runs and write partitioned Excel files."
    )
    parser.add_argument(
        "run_names",
        nargs="*",
        help="One or more screening run names (e.g. train-screening-l0-v1).",
    )
    parser.add_argument(
        "--run-list",
        type=Path,
        default=None,
        help="Path to a JSON file containing run name lists, e.g. {'screening': ['run1', 'run2']}.",
    )
    parser.add_argument(
        "--run-key",
        default="screening",
        help="Key to read from --run-list (default: screening).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder name under data/ (e.g. 'screening_eligible').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400_000,
        help="Maximum records per output partition (default: 400000).",
    )
    parser.add_argument(
        "--print-missing",
        action="store_true",
        help="Print runs that are missing datasets or batch output JSONL files, then exit without processing.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip runs missing datasets or batch output JSONL files and process the runs that are present.",
    )
    return parser.parse_args()


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

    if args.run_list is not None:
        run_names.extend(_load_run_names_from_list(
            args.run_list, args.run_key))

    if not run_names:
        raise ValueError("Provide run names positionally or via --run-list.")

    seen = set()
    duplicates = []
    for run_name in run_names:
        if run_name in seen:
            duplicates.append(run_name)
        seen.add(run_name)
    if duplicates:
        raise ValueError(
            f"Duplicate run names provided. Sample: {duplicates[:5]}")

    return run_names


def _missing_run_reasons(run_name: str) -> list[str]:
    reasons = []

    dataset_path = DATASETS_DIR / f"{run_name}-dataset.json"
    if not dataset_path.exists():
        reasons.append(f"missing dataset: {dataset_path}")

    output_dir = _REPO_ROOT / "data" / "artifacts" / "batch_outputs" / run_name
    if not output_dir.exists():
        reasons.append(f"missing batch output dir: {output_dir}")
    elif not any(output_dir.glob("*.jsonl")):
        reasons.append(f"missing batch output JSONL files: {output_dir}")

    return reasons


def _collect_missing_runs(run_names: list[str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for run_name in run_names:
        reasons = _missing_run_reasons(run_name)
        if reasons:
            missing[run_name] = reasons
    return missing


def _print_missing_runs(run_names: list[str]) -> None:
    missing = _collect_missing_runs(run_names)

    present = len(run_names) - len(missing)
    print(
        f"Checked {len(run_names):,} run(s): {present:,} present, {len(missing):,} missing/incomplete.")
    if not missing:
        return

    print("\nMissing or incomplete runs:")
    for run_name, reasons in missing.items():
        print(f"- {run_name}")
        for reason in reasons:
            print(f"  - {reason}")


def _filter_missing_runs(run_names: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    missing = _collect_missing_runs(run_names)
    runnable = [run_name for run_name in run_names if run_name not in missing]
    return runnable, missing


def main() -> None:
    args = parse_args()
    run_names = _resolve_run_names(args)

    if args.print_missing:
        _print_missing_runs(run_names)
        return

    skipped_missing: dict[str, list[str]] = {}
    if args.skip_missing:
        run_names, skipped_missing = _filter_missing_runs(run_names)
        print(
            f"Skipping {len(skipped_missing):,} missing/incomplete run(s); "
            f"processing {len(run_names):,} present run(s)."
        )
        if not run_names:
            print("No present runs to process.")
            return

    output_base = _REPO_ROOT / "data" / args.output_dir

    per_run_summaries: dict[str, dict] = {}
    all_frames: list[pd.DataFrame] = []
    seen_uts: set[str] = set()

    for run_name in run_names:
        print(f"Loading run '{run_name}' ...")
        all_df, summary = _load_run(run_name)

        run_uts = set(all_df["UT"].astype(str))
        overlap = run_uts & seen_uts
        if overlap:
            raise AssertionError(
                f"Duplicate UTs across runs: {len(overlap)} record(s) from '{run_name}' "
                f"already seen in a previous run. Sample: {sorted(overlap)[:5]}"
            )

        seen_uts |= run_uts
        all_frames.append(all_df)
        per_run_summaries[run_name] = summary

        print(
            f"  total={summary['total']:,}  eligible={summary['eligible']:,}  "
            f"not_eligible={summary['not_eligible']:,}  no_output={summary['no_output']:,}"
        )

    if not all_frames or all(len(f) == 0 for f in all_frames):
        print("No records found. Nothing to write.")
        return

    combined_all = pd.concat(all_frames, ignore_index=True)
    combined_all["UT"] = combined_all["UT"].astype(str)

    # Final sanity check — assert no duplicates survived (should be caught above, but verify)
    dupes = combined_all["UT"][combined_all["UT"].duplicated(
        keep=False)].unique().tolist()
    if dupes:
        raise AssertionError(
            f"Duplicate UTs in combined output ({len(dupes)} unique IDs). "
            f"Sample: {dupes[:5]}"
        )

    output_base.mkdir(parents=True, exist_ok=True)
    all_dir = output_base / "all"
    eligible_dir = output_base / "eligible"
    all_dir.mkdir(parents=True, exist_ok=True)
    eligible_dir.mkdir(parents=True, exist_ok=True)

    all_path = all_dir / "screening_all.csv"
    csv_df = combined_all.copy()
    for col in csv_df.columns:
        csv_df[col] = csv_df[col].map(_csv_cell)
    csv_df.to_csv(all_path, index=False)
    print(
        f"\nAll screening records written to {all_path.relative_to(_REPO_ROOT)}")

    combined_eligible = combined_all[combined_all["is_eligible"]].copy()
    total_eligible = len(combined_eligible)
    n_partitions = -(-total_eligible //
                     args.batch_size) if total_eligible else 0
    print(f"\nTotal eligible records: {total_eligible:,}")
    print(
        f"Writing {n_partitions} partition(s) of up to {args.batch_size:,} records ...")

    partition_details: list[dict] = []

    for i in range(n_partitions):
        chunk = combined_eligible.iloc[i *
                                       args.batch_size: (i + 1) * args.batch_size].copy()
        part_num = i + 1

        years = pd.to_numeric(
            chunk["publication_year"], errors="coerce").dropna()
        max_year = int(years.max()) if len(years) else 0
        min_year = int(years.min()) if len(years) else 0

        filename = f"wos_{part_num:02d}_{max_year}-{min_year}.xlsx"
        part_dir = eligible_dir / str(part_num)
        part_dir.mkdir(parents=True, exist_ok=True)
        dest = part_dir / filename

        excel_df = chunk.copy()
        excel_df["authors"] = excel_df["authors"].apply(_list_to_str)
        excel_df["wos_categories"] = excel_df["wos_categories"].apply(
            _list_to_str)

        # Column order: core fields → eligibility → stage numerics → stage labels
        stage_label_cols = [
            c for c in _STAGE_LABEL_COLS if c in excel_df.columns]
        col_order = _CORE_FIELDS + ["eligibility"] + \
            _STAGE_NUMERIC_COLS + stage_label_cols
        excel_df = excel_df[[c for c in col_order if c in excel_df.columns]]

        excel_df.to_excel(dest, index=False)
        print(
            f"  Partition {part_num}: {len(chunk):,} records → {dest.relative_to(_REPO_ROOT)}")

        partition_details.append({
            "partition": part_num,
            "file": str(dest.relative_to(_REPO_ROOT)),
            "records": len(chunk),
            "year_range": {"min": min_year, "max": max_year},
        })

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_base.relative_to(_REPO_ROOT)),
        "batch_size": args.batch_size,
        "run_names": run_names,
        "run_list": str(args.run_list) if args.run_list else None,
        "run_key": args.run_key if args.run_list else None,
        "skip_missing": args.skip_missing,
        "skipped_missing": skipped_missing,
        "outputs": {
            "all": str(all_path.relative_to(_REPO_ROOT)),
            "eligible_dir": str(eligible_dir.relative_to(_REPO_ROOT)),
        },
        "per_run": per_run_summaries,
        "totals": {
            "input_records": sum(s["total"] for s in per_run_summaries.values()),
            "eligible": total_eligible,
            "not_eligible": sum(s["not_eligible"] for s in per_run_summaries.values()),
            "no_output": sum(s["no_output"] for s in per_run_summaries.values()),
        },
        "partitions": partition_details,
    }

    meta_path = output_base / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nMetadata written to {meta_path.relative_to(_REPO_ROOT)}")

    eligible_meta_path = eligible_dir / "partition_metadata.json"
    eligible_meta = {
        "generated_at": metadata["generated_at"],
        "output_dir": str(eligible_dir.relative_to(_REPO_ROOT)),
        "batch_size": args.batch_size,
        "run_names": run_names,
        "skip_missing": args.skip_missing,
        "skipped_missing": skipped_missing,
        "totals": {
            "eligible": total_eligible,
        },
        "partitions": partition_details,
    }
    eligible_meta_path.write_text(json.dumps(
        eligible_meta, indent=2, ensure_ascii=False))
    print(
        f"Eligible partition metadata written to {eligible_meta_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
