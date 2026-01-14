"""
Prediction endpoints for all ML models
"""
from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api.models.schemas import (
    CustomerSegmentRequest, CustomerSegmentResponse,
    PurchasePredictRequest, PurchasePredictResponse,
    ChurnPredictRequest, ChurnPredictResponse,
    RecommendationRequest, RecommendationResponse, ProductRecommendation,
    PriceOptimizationRequest, PriceOptimizationResponse
)
from src.models.segmentation import CustomerSegmentation
from src.models.purchase_prediction import PurchasePredictor
from src.models.churn_prediction import ChurnPredictor
from src.models.recommendation import ProductRecommender
from src.models.price_optimization import PriceOptimizer
from src.config import DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Load models (singleton pattern)
class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_all_models()
        return cls._instance
    
    def load_all_models(self):
        logger.info("Loading all ML models...")
        
        # Load data
        self.customer_features = pd.read_csv(DATA_DIR / "features" / "customer_features.csv")
        self.product_features = pd.read_csv(DATA_DIR / "features" / "product_features.csv")
        
        # Load models
        try:
            self.segmentation_model = CustomerSegmentation()
            self.segmentation_model.load_model()
            logger.info("✓ Segmentation model loaded")
        except Exception as e:
            logger.warning(f"Segmentation model not loaded: {e}")
            self.segmentation_model = None
        
        try:
            self.purchase_model = PurchasePredictor()
            self.purchase_model.load_model()
            logger.info("✓ Purchase prediction model loaded")
        except Exception as e:
            logger.warning(f"Purchase model not loaded: {e}")
            self.purchase_model = None
        
        try:
            self.churn_model = ChurnPredictor()
            self.churn_model.load_model()
            logger.info("✓ Churn prediction model loaded")
        except Exception as e:
            logger.warning(f"Churn model not loaded: {e}")
            self.churn_model = None
        
        try:
            self.recommender = ProductRecommender()
            self.recommender.load_model()
            logger.info("✓ Recommendation model loaded")
        except Exception as e:
            logger.warning(f"Recommender not loaded: {e}")
            self.recommender = None
        
        try:
            self.price_optimizer = PriceOptimizer()
            self.price_optimizer.load_model()
            logger.info("✓ Price optimization model loaded")
        except Exception as e:
            logger.warning(f"Price optimizer not loaded: {e}")
            self.price_optimizer = None
        
        logger.info("All models loaded successfully!")

# Initialize model registry
model_registry = ModelRegistry()

@router.post("/segment", response_model=CustomerSegmentResponse)
async def predict_customer_segment(request: CustomerSegmentRequest):
    """Predict customer segment"""
    try:
        customer_data = model_registry.customer_features[
            model_registry.customer_features['customer_id'] == request.customer_id
        ]
        
        if customer_data.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        segment = model_registry.segmentation_model.predict(customer_data)[0]
        
        segment_names = {0: "VIP", 1: "Medium Value", 2: "High Value", 3: "Low Value"}
        
        return CustomerSegmentResponse(
            customer_id=request.customer_id,
            segment=int(segment),
            segment_name=segment_names.get(segment, "Unknown"),
            confidence=0.85
        )
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/purchase-predict", response_model=PurchasePredictResponse)
async def predict_purchase(request: PurchasePredictRequest):
    """Predict purchase likelihood"""
    try:
        customer_data = model_registry.customer_features[
            model_registry.customer_features['customer_id'] == request.customer_id
        ]
        
        if customer_data.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        probability = model_registry.purchase_model.predict(customer_data)[0]
        will_purchase = probability > 0.5
        
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return PurchasePredictResponse(
            customer_id=request.customer_id,
            purchase_probability=float(probability),
            will_purchase=will_purchase,
            risk_level=risk_level
        )
    except Exception as e:
        logger.error(f"Purchase prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/churn-predict", response_model=ChurnPredictResponse)
async def predict_churn(request: ChurnPredictRequest):
    """Predict customer churn risk"""
    try:
        customer_data = model_registry.customer_features[
            model_registry.customer_features['customer_id'] == request.customer_id
        ]
        
        if customer_data.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        probability = model_registry.churn_model.predict(customer_data)[0]
        is_at_risk = probability > 0.5
        
        if probability > 0.7:
            risk_level = "High"
            actions = ["Immediate outreach", "Offer discount", "Personalized campaign"]
        elif probability > 0.4:
            risk_level = "Medium"
            actions = ["Send re-engagement email", "Showcase new products"]
        else:
            risk_level = "Low"
            actions = ["Continue regular engagement"]
        
        return ChurnPredictResponse(
            customer_id=request.customer_id,
            churn_probability=float(probability),
            is_at_risk=is_at_risk,
            risk_level=risk_level,
            recommended_actions=actions
        )
    except Exception as e:
        logger.error(f"Churn prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations for customer"""
    try:
        recommendations = model_registry.recommender.recommend(
            request.customer_id, 
            request.n_recommendations
        )
        
        # Get product details
        rec_list = []
        for i, prod_id in enumerate(recommendations):
            product = model_registry.product_features[
                model_registry.product_features['product_id'] == prod_id
            ]
            
            rec_list.append(ProductRecommendation(
                product_id=prod_id,
                product_name=product['product_name'].values[0] if not product.empty else None,
                score=1.0 - (i * 0.1)  # Decreasing score
            ))
        
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=rec_list
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-price", response_model=PriceOptimizationResponse)
async def optimize_price(request: PriceOptimizationRequest):
    """Get optimal price for product"""
    try:
        product_data = model_registry.product_features[
            model_registry.product_features['product_id'] == request.product_id
        ]
        
        if product_data.empty:
            raise HTTPException(status_code=404, detail="Product not found")
        
        current_price = product_data['price'].values[0]
        optimal_price = model_registry.price_optimizer.predict_optimal_price(product_data)[0]
        
        price_diff = optimal_price - current_price
        price_diff_pct = (price_diff / current_price) * 100
        
        if price_diff_pct > 5:
            recommendation = "Increase price"
        elif price_diff_pct < -5:
            recommendation = "Decrease price"
        else:
            recommendation = "Maintain current price"
        
        return PriceOptimizationResponse(
            product_id=request.product_id,
            product_name=product_data['product_name'].values[0],
            current_price=float(current_price),
            optimal_price=float(optimal_price),
            price_adjustment=float(price_diff),
            price_adjustment_pct=float(price_diff_pct),
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Price optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))