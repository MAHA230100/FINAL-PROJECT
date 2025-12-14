# Regression Models

This directory contains trained regression models for continuous value prediction.

## Model Types

- **Length of Stay (LOS)**: Predict patient hospital stay duration
- **Cost Prediction**: Predict healthcare costs
- **Resource Utilization**: Predict resource requirements

## Model Files

Each model includes:
- `{model_name}_model.pkl`: Trained model
- `{model_name}_scaler.pkl`: Feature scaler
- `{model_name}_features.pkl`: Feature names
- `{model_name}_metrics.json`: Performance metrics

## Usage

Models are automatically loaded by the API endpoints. Use the `/predict/regress` endpoint to make predictions.