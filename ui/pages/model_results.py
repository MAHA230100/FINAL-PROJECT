import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show_model_results():
    """Model results and metrics page"""
    st.header("🤖 Model Results")
    
    _default_api = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_BASE = st.sidebar.text_input("API base URL", _default_api, key="api_base_url_models")
    
    # Model Training Section
    st.subheader("Train New Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["classification", "regression", "clustering"],
            key="model_type"
        )
        dataset_name = st.text_input("Dataset Name", value="mimic_demo", key="train_dataset")
    
    with col2:
        parameters = {
            "max_depth": st.slider("Max Depth", 1, 10, 3),
            "n_estimators": st.slider("N Estimators", 10, 200, 100),
            "learning_rate": st.slider("Learning Rate", 0.01, 1.0, 0.1)
        }
    
    if st.button("Train Model", key="train_model"):
        try:
            response = requests.post(
                f"{API_BASE}/model/train/{model_type}",
                json={"model_type": model_type, "dataset_name": dataset_name, "parameters": parameters},
                timeout=60
            )
            result = response.json()
            st.success("Model training initiated!")
            st.json(result)
        except Exception as e:
            st.error(f"Model training failed: {e}")
    
    # Model Results Section
    st.subheader("Model Results")
    
    selected_model_type = st.selectbox(
        "Select Model Type to View Results",
        ["classification", "regression", "clustering"],
        key="view_model_type"
    )
    
    if st.button("Load Model Results", key="load_results"):
        try:
            response = requests.get(f"{API_BASE}/model/results/{selected_model_type}", timeout=10)
            results = response.json()
            
            if selected_model_type == "classification":
                st.write("**Classification Metrics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{results.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precision", f"{results.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{results.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1 Score", f"{results.get('f1_score', 0):.3f}")
                
                # Confusion Matrix
                if "confusion_matrix" in results:
                    st.write("**Confusion Matrix**")
                    cm = results["confusion_matrix"]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
            
            elif selected_model_type == "regression":
                st.write("**Regression Metrics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{results.get('rmse', 0):.3f}")
                with col2:
                    st.metric("MAE", f"{results.get('mae', 0):.3f}")
                with col3:
                    st.metric("R² Score", f"{results.get('r2_score', 0):.3f}")
                with col4:
                    st.metric("Residuals", results.get('residuals', 'N/A'))
            
            elif selected_model_type == "clustering":
                st.write("**Clustering Metrics**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("N Clusters", results.get('n_clusters', 0))
                with col2:
                    st.metric("Silhouette Score", f"{results.get('silhouette_score', 0):.3f}")
                with col3:
                    st.metric("Total Samples", sum(results.get('cluster_sizes', [0, 0, 0])))
                
                # Cluster sizes visualization
                if "cluster_sizes" in results:
                    st.write("**Cluster Sizes**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    cluster_sizes = results["cluster_sizes"]
                    ax.bar(range(len(cluster_sizes)), cluster_sizes)
                    ax.set_title('Cluster Sizes')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Number of Samples')
                    st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Failed to load model results: {e}")
    
    # List available models
    st.subheader("Available Models")
    if st.button("List Models", key="list_models"):
        try:
            response = requests.get(f"{API_BASE}/model/list", timeout=10)
            models = response.json()
            
            if "models" in models:
                df = pd.DataFrame(models["models"])
                st.dataframe(df)
        except Exception as e:
            st.error(f"Failed to list models: {e}")
