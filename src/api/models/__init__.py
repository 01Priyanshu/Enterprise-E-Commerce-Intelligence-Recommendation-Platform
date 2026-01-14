"""API models"""
from .schemas import (
    CustomerSegmentRequest, CustomerSegmentResponse,
    PurchasePredictRequest, PurchasePredictResponse,
    ChurnPredictRequest, ChurnPredictResponse,
    RecommendationRequest, RecommendationResponse,
    PriceOptimizationRequest, PriceOptimizationResponse,
    HealthResponse, MetricsResponse
)

__all__ = [
    "CustomerSegmentRequest", "CustomerSegmentResponse",
    "PurchasePredictRequest", "PurchasePredictResponse",
    "ChurnPredictRequest", "ChurnPredictResponse",
    "RecommendationRequest", "RecommendationResponse",
    "PriceOptimizationRequest", "PriceOptimizationResponse",
    "HealthResponse", "MetricsResponse"
]