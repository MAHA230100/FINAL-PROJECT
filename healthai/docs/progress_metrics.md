# Final Progress Metrics

- Repo structure set up: yes (mono-repo `healthai/`)
- Data: 3 sample datasets downloaded; EDA and cleaning outputs generated
- Baseline models: classification and regression trained with RF; models saved
- Metrics:
  - Classification: accuracy/F1 reported via CLI and notebook
  - Regression: RMSE/MAE/R2 reported via CLI and notebook
- Explainability: SHAP bar and beeswarm plots generated and saved
- API: inference endpoints wired to saved models with graceful fallback
- UI: Streamlit dashboard with API integration for predictions
- Chatbot: RAG-lite FAQ integrated into dashboard
- Docker: two services (api, ui) with compose; smoke test script provided

Next steps (optional):
- Replace TF-IDF RAG with embedding model, persist vector store
- Add unit tests under `tests/`
- Add authentication for API/UI 