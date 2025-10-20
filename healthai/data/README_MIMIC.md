# MIMIC-IV Demo Data Processing

This guide walks through extracting, cleaning, and processing the MIMIC-IV clinical database demo for machine learning tasks.

## Dataset Overview

The MIMIC-IV demo contains de-identified health records for 100 patients from Beth Israel Deaconess Medical Center. It includes:

- **Hospital data**: admissions, patients, lab events, diagnoses, procedures
- **ICU data**: chart events, ICU stays, vital signs, medications

## Step-by-Step Processing

### 1. Extract Raw Data

```bash
cd healthai/data
python extract_mimic_data.py
```

**What it does:**
- Decompresses all `.csv.gz` files from `../../scripts/data/mimic_demo/`
- Saves to `healthai/data/raw/mimic/`
- Shows data summary with row/column counts

**Output:**
- `hosp_*.csv` files (hospital data)
- `icu_*.csv` files (ICU data)
- `demo_subject_ids.csv` (patient IDs)

### 2. Clean and Preprocess Data

```bash
python clean_mimic_data.py
```

**What it does:**
- Cleans patients, admissions, lab events, chart events
- Handles missing values and outliers
- Creates ML-ready datasets:
  - **Mortality prediction**: age, gender, admission info → death risk
  - **Length of stay prediction**: demographics, admission type → LOS days
  - **Lab summary**: aggregated lab values per patient

**Output files in `healthai/data/processed/mimic/`:**
- `mortality_prediction.csv`
- `los_prediction.csv`
- `lab_summary.csv`

### 3. Train Models

```bash
python process_mimic_data.py
```

**What it does:**
- Trains Random Forest models for both tasks
- Evaluates performance with metrics
- Saves trained models and scalers
- Creates feature importance reports

**Output models in `healthai/models/`:**
- `mimic_mortality_model.pkl` + scaler + features
- `mimic_los_model.pkl` + scaler + features

## Using the Models

### API Integration

The models are automatically integrated into the FastAPI:

```python
# Classification endpoint
POST /predict/classify
{
  "features": [65.0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # age, gender, admission features
}

# Regression endpoint  
POST /predict/regress
{
  "features": [65.0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # same features
}
```

### Feature Descriptions

**Mortality Prediction Features:**
- `anchor_age`: Patient age
- `gender`: 1=male, 0=female
- `los_days`: Length of stay
- `adm_type_*`: Admission type (dummy encoded)
- `adm_loc_*`: Admission location (dummy encoded)

**Length of Stay Features:**
- Same as above, but `los_days` is the target variable

## Expected Performance

**Mortality Prediction:**
- Accuracy: ~85-90% (imbalanced dataset)
- Top features: age, admission type, length of stay

**Length of Stay Prediction:**
- RMSE: ~3-5 days
- R²: ~0.3-0.5
- Top features: age, admission type, admission location

## Data Quality Notes

- **Small dataset**: Only 100 patients (demo limitation)
- **Missing values**: Handled with median imputation
- **Outliers**: Removed beyond 3 standard deviations
- **Categorical encoding**: One-hot encoding for admission types/locations

## Next Steps

1. **Scale up**: Use full MIMIC-IV dataset (requires access approval)
2. **Feature engineering**: Add more clinical features, time-series patterns
3. **Advanced models**: Try XGBoost, neural networks, ensemble methods
4. **Validation**: Cross-validation, temporal validation for time-series

## Troubleshooting

**Common issues:**
- **Memory errors**: Process smaller chunks for large datasets
- **Missing dependencies**: Install `pandas`, `scikit-learn`, `numpy`
- **File paths**: Ensure scripts run from `healthai/data/` directory

**File structure:**
```
healthai/
├── data/
│   ├── extract_mimic_data.py      # Step 1: Extract
│   ├── clean_mimic_data.py        # Step 2: Clean  
│   ├── process_mimic_data.py      # Step 3: Train
│   ├── README_MIMIC.md           # This guide
│   ├── raw/mimic/                # Extracted CSV files
│   └── processed/mimic/           # Cleaned ML datasets
└── models/                        # Trained models
```
