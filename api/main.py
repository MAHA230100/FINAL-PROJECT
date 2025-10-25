from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import route modules
from .routes import predictions, data, eda, models, ai_tools

app = FastAPI(title="HealthAI API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router)
app.include_router(data.router)
app.include_router(eda.router)
app.include_router(models.router)
app.include_router(ai_tools.router)


@app.get("/health")
def health():
    return {"status": "ok", "message": "HealthAI API is running"}


@app.get("/")
def root():
    return {
        "message": "HealthAI API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "predictions": "/predict",
            "data": "/data", 
            "eda": "/eda",
            "models": "/model",
            "ai_tools": "/ai-tools"
        }
    }