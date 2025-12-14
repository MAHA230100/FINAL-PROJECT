# Clustering Models

This directory contains trained clustering models for patient segmentation and pattern discovery.

## Model Types

- **Patient Segmentation**: Group patients by similar characteristics
- **Risk Clustering**: Identify patient risk patterns
- **Treatment Clustering**: Group similar treatment approaches

## Model Files

Each model includes:
- `{model_name}_model.pkl`: Trained model
- `{model_name}_scaler.pkl`: Feature scaler
- `{model_name}_features.pkl`: Feature names
- `{model_name}_metrics.json`: Performance metrics

## Usage

Models are automatically loaded by the API endpoints. Use the `/model/results/clustering` endpoint to view clustering results.