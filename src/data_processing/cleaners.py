"""
Data cleaning and preprocessing functions
Handles missing values, outliers, and data quality issues
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class DataCleaner:
    """Clean and preprocess raw data"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean customer data"""
        logger.info(f"Cleaning customer data: {len(df)} rows")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['customer_id'])
        
        # Handle missing values
        df['email'] = df['email'].fillna('unknown@example.com')
        df['phone'] = df['phone'].fillna('000-000-0000')
        
        # Convert date columns
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        
        # Data validation
        df = df[df['age'].between(18, 120)]
        
        # Standardize categorical values
        df['gender'] = df['gender'].str.strip().str.title()
        df['loyalty_tier'] = df['loyalty_tier'].str.strip().str.title()
        
        self.cleaning_stats['customers'] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'removed_rows': initial_rows - len(df)
        }
        
        logger.info(f"Customer data cleaned: {len(df)} rows remaining")
        return df
    
    def clean_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean product data"""
        logger.info(f"Cleaning product data: {len(df)} rows")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['product_id'])
        
        # Handle missing values
        df['brand'] = df['brand'].fillna('Unknown')
        df['rating'] = df['rating'].fillna(df['rating'].mean())
        df['num_reviews'] = df['num_reviews'].fillna(0)
        
        # Data validation
        df = df[df['price'] > 0]
        df = df[df['stock_quantity'] >= 0]
        
        # Convert date columns
        df['launch_date'] = pd.to_datetime(df['launch_date'])
        
        # Calculate profit margin
        df['profit_margin'] = ((df['price'] - df['cost']) / df['price'] * 100).round(2)
        
        self.cleaning_stats['products'] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'removed_rows': initial_rows - len(df)
        }
        
        logger.info(f"Product data cleaned: {len(df)} rows remaining")
        return df
    
    def clean_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean order data"""
        logger.info(f"Cleaning order data: {len(df)} rows")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['order_id'])
        
        # Convert date columns
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['delivery_date'] = pd.to_datetime(df['delivery_date'])
        
        # Data validation
        df = df[df['final_amount'] >= 0]
        df = df[df['num_items'] > 0]
        
        # Calculate delivery time
        df['delivery_days'] = (df['delivery_date'] - df['order_date']).dt.days
        
        # Handle negative delivery days (data errors)
        df.loc[df['delivery_days'] < 0, 'delivery_days'] = df['delivery_days'].median()
        
        # Add time features
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.month
        df['order_day'] = df['order_date'].dt.day
        df['order_dayofweek'] = df['order_date'].dt.dayofweek
        df['order_quarter'] = df['order_date'].dt.quarter
        
        self.cleaning_stats['orders'] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'removed_rows': initial_rows - len(df)
        }
        
        logger.info(f"Order data cleaned: {len(df)} rows remaining")
        return df
    
    def clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean review data"""
        logger.info(f"Cleaning review data: {len(df)} rows")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['review_id'])
        
        # Handle missing values
        df['review_text'] = df['review_text'].fillna('')
        df['helpful_count'] = df['helpful_count'].fillna(0)
        
        # Data validation
        df = df[df['rating'].between(1, 5)]
        
        # Convert date columns
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Text length feature
        df['review_length'] = df['review_text'].str.len()
        
        # Sentiment proxy (simple rule-based)
        df['sentiment'] = df['rating'].apply(
            lambda x: 'Positive' if x >= 4 else ('Neutral' if x == 3 else 'Negative')
        )
        
        self.cleaning_stats['reviews'] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'removed_rows': initial_rows - len(df)
        }
        
        logger.info(f"Review data cleaned: {len(df)} rows remaining")
        return df
    
    def get_cleaning_summary(self) -> Dict:
        """Get summary of cleaning operations"""
        return self.cleaning_stats