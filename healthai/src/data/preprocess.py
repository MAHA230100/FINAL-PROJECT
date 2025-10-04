from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class SplitPaths:
	train: Path
	val: Path
	test: Path


class TabularPreprocessor:
	def __init__(self, categorical_cols: list[str], numeric_cols: list[str]):
		self.categorical_cols = categorical_cols
		self.numeric_cols = numeric_cols
		self.pipeline: Optional[Pipeline] = None

	def build_pipeline(self) -> Pipeline:
		categorical = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])
		numeric = Pipeline(steps=[("scaler", StandardScaler())])
		pre = ColumnTransformer(
			transformers=[
				("cat", categorical, self.categorical_cols),
				("num", numeric, self.numeric_cols),
			]
		)
		self.pipeline = Pipeline(steps=[("pre", pre)])
		return self.pipeline

	def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
		if self.pipeline is None:
			self.build_pipeline()
		X = df[self.categorical_cols + self.numeric_cols]
		return self.pipeline.fit_transform(X)

	def transform(self, df: pd.DataFrame) -> np.ndarray:
		if self.pipeline is None:
			raise ValueError("Pipeline not fitted")
		X = df[self.categorical_cols + self.numeric_cols]
		return self.pipeline.transform(X)


def split_dataframe(df: pd.DataFrame, target_col: Optional[str] = None, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if target_col and target_col in df.columns:
		train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=seed, stratify=df[target_col] if df[target_col].nunique() > 1 else None)
		relative_val = val_size / (test_size + val_size)
		val_df, test_df = train_test_split(temp_df, test_size=1 - relative_val, random_state=seed, stratify=temp_df[target_col] if temp_df[target_col].nunique() > 1 else None)
	else:
		train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=seed)
		relative_val = val_size / (test_size + val_size)
		val_df, test_df = train_test_split(temp_df, test_size=1 - relative_val, random_state=seed)
	return train_df, val_df, test_df


# Text, imaging, and time-series preprocessing hooks to be implemented later. 