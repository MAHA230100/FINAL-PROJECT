from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path
import os

router = APIRouter(prefix="/eda", tags=["eda"])

class EDAAnalysisRequest(BaseModel):
    dataset_name: str
    analysis_type: str = "comprehensive"  # basic, advanced, comprehensive
    columns: List[str] = []
    visualization_types: List[str] = ["histogram", "correlation", "boxplot"]
    output_format: str = "json"  # json, html, png

class EDAResults(BaseModel):
    analysis_id: str
    dataset_name: str
    analysis_type: str
    results: Dict[str, Any]
    visualizations: List[str]
    status: str

@router.get("/")
async def get_eda_info():
    """Get EDA analysis information"""
    try:
        eda_results_path = Path("eda_results")
        
        if not eda_results_path.exists():
            return {
                "message": "No EDA results found. Run EDA analysis first.",
                "available_analyses": [],
                "status": "no_data"
            }
        
        # Get available analyses
        analyses = []
        if (eda_results_path / "summary_statistics.json").exists():
            analyses.append({
                "type": "summary_statistics",
                "file": "summary_statistics.json",
                "description": "Dataset summary and basic statistics"
            })
        
        if (eda_results_path / "clinical_insights.json").exists():
            analyses.append({
                "type": "clinical_insights",
                "file": "clinical_insights.json",
                "description": "Clinical insights and patterns"
            })
        
        # Get available visualizations
        visualizations = []
        for viz_dir in ["distributions", "correlations", "outcomes", "interactive"]:
            viz_path = eda_results_path / viz_dir
            if viz_path.exists():
                viz_files = list(viz_path.glob("*"))
                visualizations.append({
                    "category": viz_dir,
                    "files": [f.name for f in viz_files],
                    "count": len(viz_files)
                })
        
        return {
            "message": "EDA analysis information retrieved",
            "available_analyses": analyses,
            "visualizations": visualizations,
            "total_analyses": len(analyses),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving EDA info: {str(e)}")

@router.post("/analyze")
async def run_eda_analysis(request: EDAAnalysisRequest):
    """Run EDA analysis on dataset"""
    try:
        # This would integrate with the actual EDA script
        # For now, return a success response with analysis ID
        analysis_id = f"eda_{request.dataset_name}_{request.analysis_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "message": f"EDA analysis initiated for {request.dataset_name}",
            "analysis_id": analysis_id,
            "analysis_type": request.analysis_type,
            "columns_to_analyze": request.columns,
            "visualization_types": request.visualization_types,
            "estimated_time": "3-10 minutes",
            "status": "initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running EDA analysis: {str(e)}")

@router.get("/summary")
async def get_eda_summary(dataset_name: Optional[str] = None):
    """Get EDA summary statistics"""
    try:
        eda_results_path = Path("eda_results")
        summary_path = eda_results_path / "summary_statistics.json"
        
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="EDA summary not found. Run EDA analysis first.")
        
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        return {
            "message": "EDA summary statistics retrieved",
            "summary": summary_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving EDA summary: {str(e)}")

@router.get("/insights")
async def get_clinical_insights():
    """Get clinical insights from EDA"""
    try:
        eda_results_path = Path("eda_results")
        insights_path = eda_results_path / "clinical_insights.json"
        
        if not insights_path.exists():
            raise HTTPException(status_code=404, detail="Clinical insights not found. Run EDA analysis first.")
        
        with open(insights_path, 'r') as f:
            insights_data = json.load(f)
        
        return {
            "message": "Clinical insights retrieved",
            "insights": insights_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clinical insights: {str(e)}")

@router.get("/visualizations")
async def get_eda_visualizations():
    """Get available EDA visualizations"""
    try:
        eda_results_path = Path("eda_results")
        visualizations = []
        
        if not eda_results_path.exists():
            return {
                "message": "No EDA visualizations found",
                "visualizations": [],
                "status": "no_data"
            }
        
        # Get visualization categories
        viz_categories = ["distributions", "correlations", "outcomes", "interactive"]
        
        for category in viz_categories:
            category_path = eda_results_path / category
            if category_path.exists():
                viz_files = []
                for file in category_path.glob("*"):
                    viz_files.append({
                        "name": file.name,
                        "type": file.suffix[1:],  # Remove the dot
                        "size": file.stat().st_size,
                        "path": str(file.relative_to(eda_results_path))
                    })
                
                if viz_files:
                    visualizations.append({
                        "category": category,
                        "files": viz_files,
                        "count": len(viz_files)
                    })
        
        return {
            "message": "EDA visualizations retrieved",
            "visualizations": visualizations,
            "total_categories": len(visualizations),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving visualizations: {str(e)}")

@router.get("/visualizations/{category}")
async def get_visualization_category(category: str):
    """Get visualizations for specific category"""
    try:
        eda_results_path = Path("eda_results")
        category_path = eda_results_path / category
        
        if not category_path.exists():
            raise HTTPException(status_code=404, detail=f"Visualization category '{category}' not found")
        
        viz_files = []
        for file in category_path.glob("*"):
            viz_files.append({
                "name": file.name,
                "type": file.suffix[1:],
                "size": file.stat().st_size,
                "path": str(file.relative_to(eda_results_path)),
                "last_modified": file.stat().st_mtime
            })
        
        return {
            "message": f"Visualizations for category '{category}' retrieved",
            "category": category,
            "files": viz_files,
            "count": len(viz_files),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving category visualizations: {str(e)}")

@router.get("/correlations")
async def get_correlation_analysis():
    """Get correlation analysis results"""
    try:
        eda_results_path = Path("eda_results")
        corr_path = eda_results_path / "correlations"
        
        if not corr_path.exists():
            raise HTTPException(status_code=404, detail="Correlation analysis not found")
        
        # Get correlation files
        corr_files = []
        for file in corr_path.glob("*"):
            corr_files.append({
                "name": file.name,
                "type": file.suffix[1:],
                "size": file.stat().st_size,
                "path": str(file.relative_to(eda_results_path))
            })
        
        return {
            "message": "Correlation analysis retrieved",
            "files": corr_files,
            "count": len(corr_files),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving correlation analysis: {str(e)}")

@router.get("/distributions")
async def get_distribution_analysis():
    """Get distribution analysis results"""
    try:
        eda_results_path = Path("eda_results")
        dist_path = eda_results_path / "distributions"
        
        if not dist_path.exists():
            raise HTTPException(status_code=404, detail="Distribution analysis not found")
        
        # Get distribution files
        dist_files = []
        for file in dist_path.glob("*"):
            dist_files.append({
                "name": file.name,
                "type": file.suffix[1:],
                "size": file.stat().st_size,
                "path": str(file.relative_to(eda_results_path))
            })
        
        return {
            "message": "Distribution analysis retrieved",
            "files": dist_files,
            "count": len(dist_files),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving distribution analysis: {str(e)}")

@router.get("/outcomes")
async def get_outcome_analysis():
    """Get outcome analysis results"""
    try:
        eda_results_path = Path("eda_results")
        outcomes_path = eda_results_path / "outcomes"
        
        if not outcomes_path.exists():
            raise HTTPException(status_code=404, detail="Outcome analysis not found")
        
        # Get outcome files
        outcome_files = []
        for file in outcomes_path.glob("*"):
            outcome_files.append({
                "name": file.name,
                "type": file.suffix[1:],
                "size": file.stat().st_size,
                "path": str(file.relative_to(eda_results_path))
            })
        
        return {
            "message": "Outcome analysis retrieved",
            "files": outcome_files,
            "count": len(outcome_files),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving outcome analysis: {str(e)}")

@router.get("/interactive")
async def get_interactive_visualizations():
    """Get interactive visualizations"""
    try:
        eda_results_path = Path("eda_results")
        interactive_path = eda_results_path / "interactive"
        
        if not interactive_path.exists():
            raise HTTPException(status_code=404, detail="Interactive visualizations not found")
        
        # Get interactive files
        interactive_files = []
        for file in interactive_path.glob("*.html"):
            interactive_files.append({
                "name": file.name,
                "type": "html",
                "size": file.stat().st_size,
                "path": str(file.relative_to(eda_results_path)),
                "description": f"Interactive {file.stem.replace('_', ' ').title()}"
            })
        
        return {
            "message": "Interactive visualizations retrieved",
            "files": interactive_files,
            "count": len(interactive_files),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving interactive visualizations: {str(e)}")