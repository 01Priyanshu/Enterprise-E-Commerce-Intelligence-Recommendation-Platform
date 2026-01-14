"""
Configuration settings for the E-Commerce Intelligence Platform
Loads environment variables and provides centralized configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": int(os.getenv("DATABASE_PORT", 5432)),
    "database": os.getenv("DATABASE_NAME", "ecommerce_db"),
    "user": os.getenv("DATABASE_USER", "user"),
    "password": os.getenv("DATABASE_PASSWORD", "password"),
}

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@"
    f"{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# API Configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "secret_key": os.getenv("API_SECRET_KEY", "your-secret-key-change-this"),
    "algorithm": os.getenv("API_ALGORITHM", "HS256"),
    "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30)),
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "ecommerce-ml-platform"),
}

# Model Configuration
MODEL_CONFIG = {
    "registry_path": MODELS_DIR,
    "segmentation_model": "segmentation/kmeans_model.pkl",
    "purchase_prediction_model": "purchase_prediction/xgboost_model.pkl",
    "recommendation_model": "recommendation/collaborative_filter.pkl",
    "churn_prediction_model": "churn_prediction/random_forest_model.pkl",
    "price_optimization_model": "price_optimization/regression_model.pkl",
}

# Data Configuration
DATA_CONFIG = {
    "raw_data_path": DATA_DIR / "raw",
    "processed_data_path": DATA_DIR / "processed",
    "features_path": DATA_DIR / "features",
}

# Logging Configuration
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": os.getenv("LOG_FILE", "./logs/app.log"),
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
    "enable_monitoring": os.getenv("ENABLE_MONITORING", "true").lower() == "true",
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "server_port": int(os.getenv("STREAMLIT_SERVER_PORT", 8501)),
    "server_address": os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost"),
}

# Model Training Configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
}

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, BASE_DIR / "logs"]:
    directory.mkdir(parents=True, exist_ok=True)