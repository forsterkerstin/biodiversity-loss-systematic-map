import os

import pandas as pd

from data_helpers._config import BASE_COLS, SCREENING_CFG as _SCREENING_CFG

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOURCE = os.path.join(ROOT, _SCREENING_CFG["source"])
SHEETS = _SCREENING_CFG["sheets"]
LABEL_COL = _SCREENING_CFG["label_column"]
TARGET_COLS = BASE_COLS + [LABEL_COL]
OUT = os.path.join(ROOT, _SCREENING_CFG["output"])


def load_sheets():
    frames = []
    for sheet in SHEETS:
        df = pd.read_excel(SOURCE, sheet_name=sheet)
        # Jaureguiberry uses correct spelling — normalise to match other sheets
        if "eligibility" in df.columns and LABEL_COL not in df.columns:
            df = df.rename(columns={"eligibility": LABEL_COL})
        present = [c for c in TARGET_COLS if c in df.columns]
        mask = df[LABEL_COL].notna()
        sub = df[present][mask].copy()
        print(f"  {sheet}: {len(sub)} records with eligbility data")
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
    print("=== L0) Screening ===")
    df = load_sheets()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows → {OUT}")


if __name__ == "__main__":
    main()
