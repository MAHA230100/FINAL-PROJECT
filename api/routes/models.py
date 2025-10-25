from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import os

router = APIRouter(prefix="/model", tags=["models"])

class ModelTrainingRequest(BaseModel):
    model_type: str  # classification, regression, clustering
    dataset_name: str
    parameters: Dict[str, Any] = {}
    test_size: float = 0.2
    random_state: int = 42

class ModelPredictionRequest(BaseModel):
    model_type: str
    model_name: str
    features: Dict[str, Any]

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    r2_score: Optional[float] = None
    created_at: str
    status: str

@router.get("/")
async def get_models_info():
    """Get information about available models"""
    try:
        models_path = Path("models")
        
        if not models_path.exists():
            return {
                "message": "No models found. Train models first.",
                "models": [],
                "status": "no_models"
            }
        
        models = []
        
        # Check for classification models
        classification_path = models_path / "classification"
        if classification_path.exists():
            results_path = classification_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                for model_name, model_data in results.items():
                    models.append({
                        "name": model_name,
                        "type": "classification",
                        "accuracy": model_data.get('metrics', {}).get('accuracy'),
                        "f1_score": model_data.get('metrics', {}).get('f1_score'),
                        "precision": model_data.get('metrics', {}).get('precision'),
                        "recall": model_data.get('metrics', {}).get('recall'),
                        "model_class": model_data.get('model_type'),
                        "status": "trained"
                    })
        
        # Check for regression models
        regression_path = models_path / "regression"
        if regression_path.exists():
            results_path = regression_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                for model_name, model_data in results.items():
                    models.append({
                        "name": model_name,
                        "type": "regression",
                        "r2_score": model_data.get('metrics', {}).get('r2_score'),
                        "rmse": model_data.get('metrics', {}).get('rmse'),
                        "mae": model_data.get('metrics', {}).get('mae'),
                        "model_class": model_data.get('model_type'),
                        "status": "trained"
                    })
        
        # Check for clustering models
        clustering_path = models_path / "clustering"
        if clustering_path.exists():
            results_path = clustering_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                for model_name, model_data in results.items():
                    models.append({
                        "name": model_name,
                        "type": "clustering",
                        "n_clusters": model_data.get('metrics', {}).get('n_clusters'),
                        "silhouette_score": model_data.get('metrics', {}).get('silhouette_score'),
                        "model_class": model_data.get('model_type'),
                        "status": "trained"
                    })
        
        return {
            "message": "Model information retrieved",
            "models": models,
            "total_models": len(models),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@router.post("/train")
async def train_model(request: ModelTrainingRequest):
    """Train a new model"""
    try:
        # This would integrate with the actual model training script
        # For now, return a success response
        return {
            "message": f"Model training initiated for {request.model_type}",
            "model_type": request.model_type,
            "dataset_name": request.dataset_name,
            "parameters": request.parameters,
            "estimated_time": "5-15 minutes",
            "status": "initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/classification")
async def get_classification_models():
    """Get classification models and their results"""
    try:
        models_path = Path("models/classification")
        
        if not models_path.exists():
            raise HTTPException(status_code=404, detail="Classification models not found")
        
        results_path = models_path / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Classification model results not found")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Get best model
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
        
        return {
            "message": "Classification models retrieved",
            "models": results,
            "best_model": best_model,
            "best_model_metrics": results[best_model]['metrics'],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving classification models: {str(e)}")

@router.get("/regression")
async def get_regression_models():
    """Get regression models and their results"""
    try:
        models_path = Path("models/regression")
        
        if not models_path.exists():
            raise HTTPException(status_code=404, detail="Regression models not found")
        
        results_path = models_path / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Regression model results not found")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Get best model
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['r2_score'])
        
        return {
            "message": "Regression models retrieved",
            "models": results,
            "best_model": best_model,
            "best_model_metrics": results[best_model]['metrics'],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving regression models: {str(e)}")

@router.get("/clustering")
async def get_clustering_models():
    """Get clustering models and their results"""
    try:
        models_path = Path("models/clustering")
        
        if not models_path.exists():
            raise HTTPException(status_code=404, detail="Clustering models not found")
        
        results_path = models_path / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Clustering model results not found")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Get best model
        best_model = max(results.keys(), key=lambda x: results[x]['metrics'].get('silhouette_score', 0))
        
        return {
            "message": "Clustering models retrieved",
            "models": results,
            "best_model": best_model,
            "best_model_metrics": results[best_model]['metrics'],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clustering models: {str(e)}")

@router.post("/predict")
async def predict(request: ModelPredictionRequest):
    """Make predictions using trained models"""
    try:
        # This would integrate with the actual model prediction logic
        # For now, return a mock prediction
        mock_predictions = {
            "classification": {
                "prediction": "Discharged",
                "confidence": 0.85,
                "probabilities": {
                    "Discharged": 0.85,
                    "Readmitted": 0.15
                }
            },
            "regression": {
                "prediction": 5.2,
                "confidence_interval": [4.1, 6.3]
            },
            "clustering": {
                "cluster": 2,
                "cluster_confidence": 0.78
            }
        }
        
        return {
            "message": f"Prediction made using {request.model_name}",
            "model_type": request.model_type,
            "model_name": request.model_name,
            "features": request.features,
            "prediction": mock_predictions.get(request.model_type, {}),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.get("/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        models_path = Path("models")
        performance_data = {}
        
        # Get classification performance
        classification_path = models_path / "classification" / "results.json"
        if classification_path.exists():
            with open(classification_path, 'r') as f:
                performance_data["classification"] = json.load(f)
        
        # Get regression performance
        regression_path = models_path / "regression" / "results.json"
        if regression_path.exists():
            with open(regression_path, 'r') as f:
                performance_data["regression"] = json.load(f)
        
        # Get clustering performance
        clustering_path = models_path / "clustering" / "results.json"
        if clustering_path.exists():
            with open(clustering_path, 'r') as f:
                performance_data["clustering"] = json.load(f)
        
        if not performance_data:
            raise HTTPException(status_code=404, detail="No model performance data found")
        
        return {
            "message": "Model performance metrics retrieved",
            "performance": performance_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model performance: {str(e)}")

@router.get("/visualizations")
async def get_model_visualizations():
    """Get model visualization files"""
    try:
        models_path = Path("models/visualizations")
        
        if not models_path.exists():
            raise HTTPException(status_code=404, detail="Model visualizations not found")
        
        visualizations = []
        for file in models_path.glob("*"):
            visualizations.append({
                "name": file.name,
                "type": file.suffix[1:],
                "size": file.stat().st_size,
                "path": str(file.relative_to(models_path))
            })
        
        return {
            "message": "Model visualizations retrieved",
            "visualizations": visualizations,
            "count": len(visualizations),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model visualizations: {str(e)}")

@router.delete("/{model_type}/{model_name}")
async def delete_model(model_type: str, model_name: str):
    """Delete a specific model"""
    try:
        model_path = Path(f"models/{model_type}/{model_name}.pkl")
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Delete model file
        model_path.unlink()
        
        return {
            "message": f"Model {model_name} of type {model_type} deleted successfully",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

@router.get("/metadata")
async def get_model_metadata():
    """Get model metadata"""
    try:
        models_path = Path("models/metadata.json")
        
        if not models_path.exists():
            raise HTTPException(status_code=404, detail="Model metadata not found")
        
        with open(models_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            "message": "Model metadata retrieved",
            "metadata": metadata,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model metadata: {str(e)}")