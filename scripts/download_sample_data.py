from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn import datasets

RAW_DIR = Path("healthai/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, name: str) -> None:
	out = RAW_DIR / f"{name}.csv"
	df.to_csv(out, index=False)
	print(f"[saved] {out}")


def breast_cancer() -> None:
	data = datasets.load_breast_cancer(as_frame=True)
	df = data.frame
	save_df(df, "classification_breast_cancer")


def diabetes() -> None:
	data = datasets.load_diabetes(as_frame=True)
	df = data.frame
	save_df(df, "regression_diabetes")


def iris() -> None:
	data = datasets.load_iris(as_frame=True)
	df = data.frame
	save_df(df, "clustering_iris")


def main() -> None:
	print("[data] writing sample datasets to", RAW_DIR)
	breast_cancer()
	diabetes()
	iris()
	print("[done]")


if __name__ == "__main__":
	main() 