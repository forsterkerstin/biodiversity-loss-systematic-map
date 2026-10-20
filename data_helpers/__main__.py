from __future__ import annotations

import argparse

from data_helpers.datasets.splitter import SplitError, run_from_config


def _parse_labels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split labels datasets into train/dev/test with stratification.")
    parser.add_argument("--labels", default="", help="Comma-separated list of labels (e.g., l0,l1,l2).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    args = parser.parse_args()

    labels = _parse_labels(args.labels)

    try:
        summaries = run_from_config(labels=labels, seed=args.seed)
    except SplitError as exc:
        raise SystemExit(f"data_helpers error: {exc}") from exc

    for summary in summaries:
        label = summary["label"]
        counts = summary["split_counts"]
        no_abstract = summary.get("no_abstract_rows", 0)
        if no_abstract:
            print(f"{label}: dropped {no_abstract} rows with empty abstract")
        print(f"{label}: train={counts['train']} dev={counts['dev']} test={counts['test']}")


if __name__ == "__main__":
    main()
