from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("healthai/data/raw")
PROC_DIR = Path("healthai/data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
	# Drop exact duplicates
	df = df.drop_duplicates()
	# Strip column names
	df.columns = [c.strip() for c in df.columns]
	# Simple missing handling: fill numeric with median, categorical with mode
	numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
	categorical_cols = [c for c in df.columns if c not in numeric_cols]
	for c in numeric_cols:
		if df[c].isna().any():
			df[c] = df[c].fillna(df[c].median())
	for c in categorical_cols:
		if df[c].isna().any():
			mode = df[c].mode(dropna=True)
			df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "missing")
	return df


def eda_report(df: pd.DataFrame) -> str:
	lines = []
	lines.append("# Quick EDA Summary")
	lines.append("\n## Shape")
	lines.append(str(df.shape))
	lines.append("\n## Columns")
	lines.append(", ".join(df.columns))
	lines.append("\n## Missing per column")
	lines.append(str(df.isna().sum().to_dict()))
	lines.append("\n## Numeric describe")
	lines.append(str(df.describe(include=["number"]).T))
	return "\n".join(lines)


def process_file(path: Path) -> None:
	df = pd.read_csv(path)
	report = eda_report(df)
	clean = basic_clean(df)
	# write outputs
	stem = path.stem
	rep_path = PROC_DIR / f"{stem}_eda.txt"
	clean_path = PROC_DIR / f"{stem}_clean.csv"
	rep_path.write_text(report)
	clean.to_csv(clean_path, index=False)
	print(f"[eda] {rep_path}")
	print(f"[clean] {clean_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Basic EDA and cleaning for raw CSVs")
	parser.add_argument("--input", default=str(RAW_DIR), help="Folder containing raw CSVs")
	args = parser.parse_args()
	folder = Path(args.input)
	csvs = sorted(folder.glob("*.csv"))
	if not csvs:
		print(f"No CSVs found in {folder}")
		return
	for csv in csvs:
		print(f"[process] {csv}")
		process_file(csv)
	print("[done]")


if __name__ == "__main__":
	main() 