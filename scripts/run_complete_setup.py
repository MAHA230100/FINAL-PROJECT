# scripts/run_complete_setup.py
import os
from pathlib import Path

def setup_directories():
    """Create all necessary directories for the application."""
    # Create data directories
    data_dirs = [
        "data/raw",
        "data/processed",
        "data/eda_results",
        "data/models",
        "data/visualizations"
    ]
    
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    print("🚀 Setting up project directories...")
    setup_directories()
    print("✅ Setup complete!")