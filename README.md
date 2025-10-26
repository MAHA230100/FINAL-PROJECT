# HealthAI - Healthcare Data Science Platform

A comprehensive full-stack data science platform for healthcare analytics, machine learning, and AI tools integration.

## 🏗️ Clean Architecture

```
project/
├── docker-compose.yml          # Main Docker configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── api/                        # FastAPI backend
│   ├── main.py                 # API entry point
│   ├── routes/                 # API route modules
│   │   ├── predictions.py      # ML prediction endpoints
│   │   ├── data.py            # Data management endpoints
│   │   ├── eda.py             # EDA endpoints
│   │   ├── models.py          # Model management endpoints
│   │   └── ai_tools.py        # AI utilities endpoints
│   └── ai_tools/              # AI utilities
│       ├── summarizer.py      # Text summarization
│       └── classifier.py     # Text classification
├── ui/                        # Streamlit frontend
│   ├── dashboard.py           # Main UI entry point
│   ├── pages/                 # UI page modules
│   │   ├── data_display.py    # Data visualization
│   │   ├── eda_visualization.py # EDA pages
│   │   ├── model_results.py   # Model results
│   │   └── ai_tools_demo.py   # AI tools demo
│   └── components/            # Reusable UI components
│       └── sidebar.py             # Navigation sidebar
├── data/                      # Data storage
│   ├── raw/                   # Raw datasets
│   └── cleaned/               # Processed datasets
├── notebooks/                 # Jupyter notebooks
│   ├── data_cleaning.ipynb    # Data preprocessing
│   └── eda.ipynb              # Exploratory data analysis
├── scripts/                   # Automation scripts
│   ├── clean_data.py          # Data cleaning automation
│   ├── run_eda.py            # EDA automation
│   └── train_model.py        # Model training automation
├── models/                    # Trained models
│   ├── classification/        # Classification models
│   ├── regression/            # Regression models
│   └── clustering/            # Clustering models
└── docker/                    # Docker configurations
    ├── api/Dockerfile         # API container
    └── ui/Dockerfile          # UI container
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)

### Running the Application

#### Development Mode (with hot-reload)

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/FINAL-PROJECT
   ```

2. **Start the development environment:**
   ```bash
   # Uses docker-compose.yml + docker-compose.override.yml
   docker-compose up
   ```

3. **Access the application:**
   - **Frontend (Streamlit)**: http://localhost:8501
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

#### Production Deployment

```bash
# Build and start production containers
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

#### Local Testing (without development overrides)

```bash
# Uses only docker-compose.yml (no hot-reload, no development tools)
docker-compose -f docker-compose.yml up
```

### Docker Compose Files Overview

- **`docker-compose.yml`**: Base configuration with service definitions
- **`docker-compose.override.yml`**: Development-specific settings (auto-loaded in dev)
- **`docker-compose.prod.yml`**: Production-specific settings (must be explicitly included)

## 🎯 Features

### Backend API (FastAPI)
- **Predictions**: `/predict/classify`, `/predict/regress`
- **Data Management**: `/data/clean`, `/data/raw`, `/data/cleaned`
- **EDA**: `/eda/analyze`, `/eda/visualizations`
- **Models**: `/model/train/{type}`, `/model/results/{type}`
- **AI Tools**: `/ai-tools/summarize`, `/ai-tools/classify`
- **Health Check**: `/health`

### Frontend UI (Streamlit)
- **🏠 Home**: Dashboard with quick access
- **🔬 Disease Prediction**: Risk classification
- **📈 LOS Prediction**: Length of stay prediction
- **👥 Patient Cohorts**: Patient clustering
- **📊 Data Display**: Raw and cleaned data views
- **📈 EDA Visualization**: Exploratory data analysis
- **🤖 Model Results**: Model performance metrics
- **🛠️ AI Tools Demo**: AI utilities demonstration
- **🔍 Health Check**: System status and monitoring

### Data Science Pipeline
- **Data Cleaning**: Automated data preprocessing
- **EDA**: Exploratory data analysis with visualizations
- **Model Training**: Classification, regression, clustering
- **Model Serving**: Real-time predictions via API
- **Monitoring**: Health checks and performance metrics

### AI Tools Integration
- **Text Summarization**: Extract key information from documents
- **Text Classification**: Categorize and analyze text
- **Future Tools**: Translation, sentiment analysis, and more

## 📊 Data Management

### Raw Data
- Place datasets in `data/raw/`
- Supports CSV, JSON, and other formats
- Includes MIMIC demo data and sample datasets

### Cleaned Data
- Processed data stored in `data/cleaned/`
- Automated cleaning pipeline
- Quality validation and metrics

### Notebooks
- `data_cleaning.ipynb`: Data preprocessing workflow
- `eda.ipynb`: Exploratory data analysis
- Interactive analysis and visualization

## 🔧 Development

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run API server:**
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Run UI server:**
   ```bash
   streamlit run ui/dashboard.py --server.port 8501
   ```

### Scripts

- `scripts/clean_data.py`: Data cleaning automation
- `scripts/run_eda.py`: EDA automation
- `scripts/train_model.py`: Model training automation

### Model Management

- Models stored in `models/`
- Automatic model loading in API
- Performance metrics tracking
- Model versioning support

## 🐳 Docker Configuration

### Services

- **API Service**: FastAPI backend on port 8000
- **UI Service**: Streamlit frontend on port 8501
- **Networking**: Services communicate via Docker network

### Environment Variables

- `API_BASE_URL`: Backend API URL (default: http://api:8000)

## 📈 Usage Examples

### API Usage

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict/classify",
    json={"features": [65, 1, 0, 1, 0.5]}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
```

### Data Cleaning

```bash
python scripts/clean_data.py --input data/raw/sample.csv --output data/cleaned/cleaned.csv
```

### Model Training

```bash
python scripts/train_model.py --model-type classification --target mortality
```

## 🔍 API Endpoints

### Predictions
- `POST /predict/classify` - Disease risk classification
- `POST /predict/regress` - Length of stay prediction

### Data Management
- `POST /data/clean` - Clean datasets
- `GET /data/raw` - List raw datasets
- `GET /data/cleaned` - List cleaned datasets

### EDA
- `POST /eda/analyze` - Run EDA analysis
- `GET /eda/visualizations` - List available visualizations

### Models
- `POST /model/train/{type}` - Train models
- `GET /model/results/{type}` - Get model results
- `GET /model/list` - List available models

### AI Tools
- `POST /ai-tools/summarize` - Text summarization
- `POST /ai-tools/classify` - Text classification
- `GET /ai-tools/available` - List available tools

## 🛠️ Customization

### Adding New Pages
1. Create page module in `ui/pages/`
2. Import and add to `dashboard.py`
3. Add navigation in `components/sidebar.py`

### Adding New API Routes
1. Create route module in `api/routes/`
2. Import and include in `main.py`
3. Add corresponding UI components

### Adding AI Tools
1. Create tool module in `api/ai_tools/`
2. Add route in `routes/ai_tools.py`
3. Update UI in `pages/ai_tools_demo.py`

## 📝 Notes

- The application maintains backward compatibility with existing functionality
- All existing endpoints and UI features are preserved
- Docker setup is optimized for development and production
- Modular structure allows easy extension and maintenance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `docker-compose up`
5. Submit a pull request

## 📄 License

This project is part of a final project for educational purposes.