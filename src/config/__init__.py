"""Configuration module for the E-Commerce Intelligence Platform"""
from .settings import (
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    DATABASE_CONFIG,
    DATABASE_URL,
    API_CONFIG,
    MLFLOW_CONFIG,
    MODEL_CONFIG,
    DATA_CONFIG,
    LOG_CONFIG,
    FEATURE_FLAGS,
    STREAMLIT_CONFIG,
    TRAINING_CONFIG,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "DATABASE_CONFIG",
    "DATABASE_URL",
    "API_CONFIG",
    "MLFLOW_CONFIG",
    "MODEL_CONFIG",
    "DATA_CONFIG",
    "LOG_CONFIG",
    "FEATURE_FLAGS",
    "STREAMLIT_CONFIG",
    "TRAINING_CONFIG",
]