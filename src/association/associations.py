from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_association_rules(df_transactions: pd.DataFrame, min_support: float = 0.05, metric: str = "lift", min_threshold: float = 1.0) -> pd.DataFrame:
	"""
	Expects one-hot encoded transaction DataFrame (columns are items, rows are patient records with 0/1).
	Returns association rules with support, confidence, lift.
	"""
	freq = apriori(df_transactions, min_support=min_support, use_colnames=True)
	rules = association_rules(freq, metric=metric, min_threshold=min_threshold)
	return rules.sort_values(by=["lift", "confidence"], ascending=False)
