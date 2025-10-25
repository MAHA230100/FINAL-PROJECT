from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path
import os

router = APIRouter(prefix="/data", tags=["data"])

class DataCleanRequest(BaseModel):
    dataset_name: str
    cleaning_options: Dict[str, Any] = {}

class DataRequest(BaseModel):
    dataset_name: str
    columns: List[str] = []
    filters: Dict[str, Any] = {}
    cleaning_options: Dict[str, bool] = {
        "remove_duplicates": True,
        "handle_missing": True,
        "normalize": False
    }

class DataSummary(BaseModel):
    total_rows: int
    total_columns: int
    missing_values: int
    data_types: Dict[str, str]
    sample_data: List[Dict[str, Any]]

@router.get("/raw")
async def get_raw_data():
    """Get raw data information"""
    try:
        raw_data_path = Path("data/raw")
        datasets = []
        
        if raw_data_path.exists():
            for file in raw_data_path.glob("*.csv"):
                file_info = {
                    "name": file.name,
                    "size": file.stat().st_size,
                    "path": str(file),
                    "last_modified": file.stat().st_mtime
                }
                datasets.append(file_info)
        
        return {
            "message": "Raw data information retrieved",
            "datasets": datasets,
            "total_datasets": len(datasets),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving raw data: {str(e)}")

@router.get("/raw/{dataset_name}")
async def get_raw_dataset(dataset_name: str, limit: int = 100):
    """Get specific raw dataset"""
    try:
        dataset_path = Path(f"data/raw/{dataset_name}")
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Get summary
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "columns": df.columns.tolist()
        }
        
        # Get sample data
        sample_data = df.head(limit).to_dict('records')
        
        return {
            "message": f"Dataset {dataset_name} retrieved successfully",
            "summary": summary,
            "sample_data": sample_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset: {str(e)}")

@router.post("/clean")
async def clean_data(request: DataCleanRequest):
    """Clean and preprocess data"""
    try:
        # This would integrate with the actual data cleaning script
        # For now, return a success response
        return {
            "message": f"Data cleaning initiated for {request.dataset_name}",
            "cleaning_options": request.cleaning_options,
            "status": "success",
            "estimated_time": "2-5 minutes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning data: {str(e)}")

@router.get("/cleaned")
async def get_cleaned_data():
    """Get cleaned data information"""
    try:
        cleaned_data_path = Path("data/cleaned")
        datasets = []
        
        if cleaned_data_path.exists():
            for file in cleaned_data_path.glob("*.csv"):
                file_info = {
                    "name": file.name,
                    "size": file.stat().st_size,
                    "path": str(file),
                    "last_modified": file.stat().st_mtime
                }
                datasets.append(file_info)
        
        return {
            "message": "Cleaned data information retrieved",
            "datasets": datasets,
            "total_datasets": len(datasets),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cleaned data: {str(e)}")

@router.get("/cleaned/{dataset_name}")
async def get_cleaned_dataset(dataset_name: str, limit: int = 100):
    """Get specific cleaned dataset"""
    try:
        dataset_path = Path(f"data/cleaned/{dataset_name}")
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Cleaned dataset not found")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Get summary
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "columns": df.columns.tolist(),
            "data_quality_score": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
        }
        
        # Get sample data
        sample_data = df.head(limit).to_dict('records')
        
        return {
            "message": f"Cleaned dataset {dataset_name} retrieved successfully",
            "summary": summary,
            "sample_data": sample_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cleaned dataset: {str(e)}")

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload new dataset"""
    try:
        # Save uploaded file
        upload_path = Path("data/raw") / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "message": f"File {file.filename} uploaded successfully",
            "path": str(upload_path),
            "size": len(content),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/summary/{dataset_name}")
async def get_data_summary(dataset_name: str):
    """Get comprehensive data summary"""
    try:
        # Try cleaned data first, then raw data
        dataset_path = Path(f"data/cleaned/{dataset_name}")
        if not dataset_path.exists():
            dataset_path = Path(f"data/raw/{dataset_name}")
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Generate comprehensive summary
        summary = {
            "basic_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "file_size": dataset_path.stat().st_size
            },
            "data_quality": {
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum(),
                "data_completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            },
            "column_info": {
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict()
            },
            "statistics": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
        }
        
        return {
            "message": f"Data summary for {dataset_name}",
            "summary": summary,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating data summary: {str(e)}")