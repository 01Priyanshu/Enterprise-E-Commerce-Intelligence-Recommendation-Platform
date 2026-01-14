"""
Main FastAPI application
E-Commerce Intelligence Platform API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.endpoints import predictions_router
from src.api.models.schemas import HealthResponse, MetricsResponse
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="E-Commerce Intelligence Platform API",
    description="Production-ready ML API serving customer segmentation, predictions, and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track metrics
start_time = time.time()
request_count = 0

# Include routers
app.include_router(predictions_router, prefix="/api/v1", tags=["predictions"])

@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to E-Commerce Intelligence Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded={
            "segmentation": True,
            "purchase_prediction": True,
            "churn_prediction": True,
            "recommendation": True,
            "price_optimization": True
        }
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["metrics"])
async def get_metrics():
    """Get API metrics"""
    uptime = time.time() - start_time
    return MetricsResponse(
        total_requests=request_count,
        active_models=5,
        uptime_seconds=uptime
    )

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("="*60)
    logger.info("STARTING E-COMMERCE INTELLIGENCE PLATFORM API")
    logger.info("="*60)
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down API...")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")