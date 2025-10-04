# Architecture

```
users ──▶ Streamlit UI (healthai/src/ui)
            │           ▲
            │ HTTP      │ JSON
            ▼           │
         FastAPI (healthai/src/api)
            │ loads models (.pkl)
            ▼
        Models & Artifacts (healthai/models)
            ▲
            │ training/eval
            ▼
  Training Scripts (healthai/src/models/tabular/*)
            ▲
            │ CSV
            ▼
        Data (healthai/data/{raw,processed})

FAQ Chatbot (RAG-lite):
  Streamlit ▶ SimpleRAG ▶ FAISS over `docs/faq/*.txt`
```

- Deployment: docker-compose builds two services (`api`, `ui`).
- Config: `src/config/config.yaml` for paths and settings.
- Explainability: SHAP scripts generate plots saved to `healthai/models/`. 