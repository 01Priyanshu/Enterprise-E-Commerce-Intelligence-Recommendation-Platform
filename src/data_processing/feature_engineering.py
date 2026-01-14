"""
Feature engineering for machine learning models
Creates derived features from raw data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Engineer features for ML models"""
    
    def __init__(self):
        self.feature_stats = {}
    
    def create_customer_features(self, customers_df: pd.DataFrame, 
                                 orders_df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level features"""
        logger.info("Creating customer features...")
        
        # Customer purchase history
        customer_stats = orders_df.groupby('customer_id').agg({
            'order_id': 'count',
            'final_amount': ['sum', 'mean', 'std'],
            'num_items': ['sum', 'mean'],
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_stats.columns = [
            'customer_id', 'total_orders', 'total_spent', 'avg_order_value',
            'std_order_value', 'total_items', 'avg_items_per_order',
            'first_order_date', 'last_order_date'
        ]
        
        # Merge with customer data
        features = customers_df.merge(customer_stats, on='customer_id', how='left')
        
        # Fill missing values for customers with no orders
        features['total_orders'] = features['total_orders'].fillna(0)
        features['total_spent'] = features['total_spent'].fillna(0)
        features['avg_order_value'] = features['avg_order_value'].fillna(0)
        features['total_items'] = features['total_items'].fillna(0)
        
        # Customer tenure (days since registration)
        features['customer_tenure_days'] = (
            datetime.now() - pd.to_datetime(features['registration_date'])
        ).dt.days
        
        # Recency (days since last order)
        features['days_since_last_order'] = (
            datetime.now() - pd.to_datetime(features['last_order_date'])
        ).dt.days
        features['days_since_last_order'] = features['days_since_last_order'].fillna(9999)
        
        # Purchase frequency (orders per month)
        features['purchase_frequency'] = (
            features['total_orders'] / 
            (features['customer_tenure_days'] / 30 + 1)
        ).round(2)
        
        # Customer value segments
        features['customer_value_segment'] = pd.qcut(
            features['total_spent'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'VIP'],
            duplicates='drop'
        )
        
        # RFM scores (Recency, Frequency, Monetary)
        features['recency_score'] = pd.qcut(
            features['days_since_last_order'], 
            q=5, 
            labels=[5, 4, 3, 2, 1],
            duplicates='drop'
        ).astype(float)
        
        features['frequency_score'] = pd.qcut(
            features['total_orders'].rank(method='first'), 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(float)
        
        features['monetary_score'] = pd.qcut(
            features['total_spent'].rank(method='first'), 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(float)
        
        # Overall RFM score
        features['rfm_score'] = (
            features['recency_score'] + 
            features['frequency_score'] + 
            features['monetary_score']
        ) / 3
        
        logger.info(f"Created customer features: {len(features)} rows, {len(features.columns)} columns")
        return features
    
    def create_product_features(self, products_df: pd.DataFrame, 
                               orders_df: pd.DataFrame,
                               reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Create product-level features"""
        logger.info("Creating product features...")
        
        features = products_df.copy()
        
        # Product age (days since launch)
        features['product_age_days'] = (
            datetime.now() - pd.to_datetime(features['launch_date'])
        ).dt.days
        
        # Price segments
        features['price_segment'] = pd.qcut(
            features['price'], 
            q=4, 
            labels=['Budget', 'Mid-range', 'Premium', 'Luxury'],
            duplicates='drop'
        )
        
        # Stock status
        features['stock_status'] = features['stock_quantity'].apply(
            lambda x: 'Out of Stock' if x == 0 else (
                'Low Stock' if x < 50 else 'In Stock'
            )
        )
        
        # Review aggregations
        if not reviews_df.empty:
            review_stats = reviews_df.groupby('product_id').agg({
                'rating': ['mean', 'count', 'std'],
                'helpful_count': 'sum'
            }).reset_index()
            
            review_stats.columns = [
                'product_id', 'avg_rating', 'review_count', 
                'rating_std', 'total_helpful'
            ]
            
            features = features.merge(review_stats, on='product_id', how='left')
            features['avg_rating'] = features['avg_rating'].fillna(features['rating'])
            features['review_count'] = features['review_count'].fillna(0)
        
        logger.info(f"Created product features: {len(features)} rows, {len(features.columns)} columns")
        return features
    
    def create_order_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Create order-level features"""
        logger.info("Creating order features...")
        
        features = orders_df.copy()
        
        # Average item price
        features['avg_item_price'] = (
            features['total_amount'] / features['num_items']
        ).round(2)
        
        # Discount percentage
        features['discount_percentage'] = (
            (features['discount'] / features['total_amount']) * 100
        ).round(2)
        
        # Is high value order
        features['is_high_value'] = (
            features['final_amount'] > features['final_amount'].quantile(0.75)
        ).astype(int)
        
        # Is weekend order
        features['is_weekend'] = features['order_dayofweek'].isin([5, 6]).astype(int)
        
        # Season
        features['season'] = features['order_month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else (
                'Spring' if x in [3, 4, 5] else (
                    'Summer' if x in [6, 7, 8] else 'Fall'
                )
            )
        )
        
        logger.info(f"Created order features: {len(features)} rows, {len(features.columns)} columns")
        return features
    
    def create_all_features(self, customers_df: pd.DataFrame,
                          products_df: pd.DataFrame,
                          orders_df: pd.DataFrame,
                          reviews_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create all feature sets"""
        logger.info("Creating all feature sets...")
        
        customer_features = self.create_customer_features(customers_df, orders_df)
        product_features = self.create_product_features(products_df, orders_df, reviews_df)
        order_features = self.create_order_features(orders_df)
        
        return {
            'customer_features': customer_features,
            'product_features': product_features,
            'order_features': order_features
        }