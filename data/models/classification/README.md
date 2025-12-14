# Classification Models

This directory contains trained classification models for disease prediction and risk assessment.

## Model Types

- **Mortality Prediction**: Binary classification for patient mortality risk
- **Disease Classification**: Multi-class classification for disease types
- **Risk Assessment**: Risk level classification (low, medium, high)

## Model Files

Each model includes:
- `{model_name}_model.pkl`: Trained model
- `{model_name}_scaler.pkl`: Feature scaler
- `{model_name}_features.pkl`: Feature names
- `{model_name}_metrics.json`: Performance metrics

## Usage

Models are automatically loaded by the API endpoints. Use the `/predict/classify` endpoint to make predictions.