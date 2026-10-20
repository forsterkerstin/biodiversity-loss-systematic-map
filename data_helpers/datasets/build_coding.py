import os

import pandas as pd

from data_helpers._config import BASE_COLS, CODING_CFG as _CODING_CFG

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOURCE = os.path.join(ROOT, _CODING_CFG["source"])
SHEETS = _CODING_CFG["sheets"]
LABELS = [
    {
        "name": item["name"],
        "cols": BASE_COLS + item["extra_cols"],
        "out": os.path.join(ROOT, item["output"]),
    }
    for item in _CODING_CFG["labels"]
]


def load_sheets(needed_cols):
    frames = []
    for sheet in SHEETS:
        df = pd.read_excel(SOURCE, sheet_name=sheet)
        present = [c for c in needed_cols if c in df.columns]
        label_cols = [c for c in present if c not in BASE_COLS]
        mask = df[label_cols].notna().any(axis=1)
        sub = df[present][mask].copy()
        print(f"  {sheet}: {len(sub)} records with label data")
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="UT (Unique WOS ID)", keep="first")
    print(f"  → {before} rows before dedup, {len(combined)} after dedup on UT")
    no_abstract = combined["Abstract"].isna() | combined["Abstract"].astype(str).str.strip().eq("")
    if no_abstract.any():
        print(f"  → dropped {no_abstract.sum()} rows with empty abstract, {(~no_abstract).sum()} remaining")
    combined = combined[~no_abstract]
    return combined


def main():
    import sys
    filter_arg = sys.argv[1] if len(sys.argv) > 1 else None
    labels = LABELS
    if filter_arg:
        labels = [l for l in LABELS if filter_arg.lower() in l["name"].lower()]
        if not labels:
            print(f"No label matching '{filter_arg}'. Available: {[l['name'] for l in LABELS]}")
            sys.exit(1)
    for label in labels:
        print(f"\n=== {label['name']} ===")
        df = load_sheets(label["cols"])
        os.makedirs(os.path.dirname(label["out"]), exist_ok=True)
        df.to_csv(label["out"], index=False)
        print(f"Saved {len(df)} rows → {label['out']}")


if __name__ == "__main__":
    main()
