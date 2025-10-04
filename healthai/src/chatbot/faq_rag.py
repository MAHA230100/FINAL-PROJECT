from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_DIR = Path("healthai/docs")
INDEX_DIR = Path("healthai/models")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class SimpleRAG:
	def __init__(self, docs: List[str], ids: List[str]):
		self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
		X = self.vectorizer.fit_transform(docs).astype(np.float32).toarray()
		self.index = faiss.IndexFlatIP(X.shape[1])
		# Normalize for cosine similarity via inner product
		norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
		Xn = X / norms
		self.index.add(Xn)
		self.docs = docs
		self.ids = ids

	def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
		q = self.vectorizer.transform([query]).astype(np.float32).toarray()
		q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
		dists, idx = self.index.search(q, k)
		results = []
		for i, score in zip(idx[0], dists[0]):
			if i == -1:
				continue
			results.append((self.docs[i], float(score)))
		return results

	def answer(self, query: str) -> str:
		chunks = self.search(query, k=3)
		context = "\n---\n".join([c for c, _ in chunks])
		return f"Context:\n{context}\n\nAnswer (heuristic): Based on the retrieved context, {query}"


def load_docs_from_folder(folder: Path) -> Tuple[List[str], List[str]]:
	docs = []
	ids = []
	for path in sorted(folder.glob("**/*.txt")):
		text = path.read_text(encoding="utf-8", errors="ignore")
		if text.strip():
			docs.append(text)
			ids.append(str(path))
	return docs, ids


def build_default_rag() -> SimpleRAG:
	faq_dir = DOCS_DIR / "faq"
	faq_dir.mkdir(parents=True, exist_ok=True)
	# Ensure a starter FAQ exists
	starter = faq_dir / "starter_faq.txt"
	if not starter.exists():
		starter.write_text("""
What is HealthAI?
HealthAI is a demo project that showcases end-to-end AI in healthcare.

How do I run the API?
Use `uvicorn healthai.src.api.main:app --reload` or docker-compose.

How do I run the UI?
Use `streamlit run healthai/src/ui/dashboard.py` or docker-compose.
""".strip())
	docs, ids = load_docs_from_folder(faq_dir)
	return SimpleRAG(docs, ids) 