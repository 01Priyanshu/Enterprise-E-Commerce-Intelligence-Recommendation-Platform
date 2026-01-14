"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Request Models
class CustomerSegmentRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    
class PurchasePredictRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    
class ChurnPredictRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    
class RecommendationRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    n_recommendations: Optional[int] = Field(5, description="Number of recommendations")
    
class PriceOptimizationRequest(BaseModel):
    product_id: str = Field(..., description="Product ID")

# Response Models
class CustomerSegmentResponse(BaseModel):
    customer_id: str
    segment: int
    segment_name: str
    confidence: float
    
class PurchasePredictResponse(BaseModel):
    customer_id: str
    purchase_probability: float
    will_purchase: bool
    risk_level: str
    
class ChurnPredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_at_risk: bool
    risk_level: str
    recommended_actions: List[str]
    
class ProductRecommendation(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    score: float
    
class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[ProductRecommendation]
    
class PriceOptimizationResponse(BaseModel):
    product_id: str
    product_name: Optional[str] = None
    current_price: float
    optimal_price: float
    price_adjustment: float
    price_adjustment_pct: float
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: dict
    
class MetricsResponse(BaseModel):
    total_requests: int
    active_models: int
    uptime_seconds: float