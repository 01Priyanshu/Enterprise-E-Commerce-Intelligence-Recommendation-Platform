"""
Price Optimization Model
Predicts optimal pricing based on product features and demand
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import MODELS_DIR, DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class PriceOptimizer:
    """Optimize product pricing"""
    
    def __init__(self):
        """Initialize price optimization model"""
        self.model = Ridge(alpha=1.0, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = MODELS_DIR / "price_optimization"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, products_df: pd.DataFrame):
        """Prepare features for price prediction"""
        logger.info("Preparing features...")
        
        # Select features that influence price
        features = [
            'product_age_days',
            'stock_quantity',
            'rating',
            'num_reviews',
            'profit_margin',
            'cost'
        ]
        
        self.feature_names = features
        
        # Prepare feature matrix and target
        X = products_df[features].fillna(0)
        y = products_df['price']
        
        logger.info(f"Prepared {len(X)} products with {len(features)} features")
        return X, y
    
    def train(self, products_df: pd.DataFrame, test_size=0.2):
        """Train the price optimization model"""
        logger.info("Training price optimization model...")
        
        # Prepare features
        X, y = self.prepare_features(products_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} products")
        logger.info(f"Test set: {len(X_test)} products")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\nModel Performance:")
        logger.info(f"  R² Score: {r2:.3f}")
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE: ${mae:.2f}")
        
        # Feature coefficients
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        logger.info("\nFeature Impact on Price:")
        for idx, row in feature_importance.iterrows():
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            logger.info(f"  {row['feature']}: {direction} price (coef: {row['coefficient']:.3f})")
        
        # Price range analysis
        logger.info(f"\nPrice Statistics:")
        logger.info(f"  Actual Price Range: ${y_test.min():.2f} - ${y_test.max():.2f}")
        logger.info(f"  Predicted Price Range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
    
    def predict_optimal_price(self, product_features: pd.DataFrame) -> np.ndarray:
        """Predict optimal price for products"""
        X = product_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def suggest_price_adjustment(self, products_df: pd.DataFrame):
        """Suggest price adjustments for products"""
        logger.info("Generating price adjustment suggestions...")
        
        # Predict optimal prices
        optimal_prices = self.predict_optimal_price(products_df)
        
        # Calculate adjustment
        products_df['optimal_price'] = optimal_prices
        products_df['price_diff'] = products_df['optimal_price'] - products_df['price']
        products_df['price_diff_pct'] = (
            products_df['price_diff'] / products_df['price'] * 100
        ).round(2)
        
        # Categorize adjustments
        products_df['adjustment_category'] = products_df['price_diff_pct'].apply(
            lambda x: 'Increase' if x > 5 else ('Decrease' if x < -5 else 'Maintain')
        )
        
        # Summary
        summary = products_df['adjustment_category'].value_counts()
        logger.info("\nPrice Adjustment Summary:")
        for category, count in summary.items():
            logger.info(f"  {category}: {count} products ({count/len(products_df)*100:.1f}%)")
        
        return products_df[['product_id', 'product_name', 'price', 'optimal_price', 
                           'price_diff', 'price_diff_pct', 'adjustment_category']]
    
    def save_model(self):
        """Save the trained model"""
        model_file = self.model_path / "regression_model.pkl"
        scaler_file = self.model_path / "scaler.pkl"
        features_file = self.model_path / "features.pkl"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.feature_names, features_file)
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self):
        """Load a trained model"""
        model_file = self.model_path / "regression_model.pkl"
        scaler_file = self.model_path / "scaler.pkl"
        features_file = self.model_path / "features.pkl"
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_names = joblib.load(features_file)
        
        logger.info(f"Model loaded from {model_file}")

def main():
    """Train and save price optimization model"""
    logger.info("="*60)
    logger.info("PRICE OPTIMIZATION MODEL TRAINING")
    logger.info("="*60)
    
    # Load data
    products_path = DATA_DIR / "features" / "product_features.csv"
    products_df = pd.read_csv(products_path)
    logger.info(f"Loaded {len(products_df)} products")
    
    # Train model
    optimizer = PriceOptimizer()
    results = optimizer.train(products_df)
    
    # Generate price suggestions for sample products
    sample_products = products_df.head(10)
    suggestions = optimizer.suggest_price_adjustment(sample_products)
    
    logger.info("\nSample Price Suggestions:")
    for idx, row in suggestions.head(5).iterrows():
        logger.info(f"\n  {row['product_name']}:")
        logger.info(f"    Current: ${row['price']:.2f}")
        logger.info(f"    Optimal: ${row['optimal_price']:.2f}")
        logger.info(f"    Change: {row['price_diff_pct']:.1f}% ({row['adjustment_category']})")
    
    # Save model
    optimizer.save_model()
    
    logger.info("="*60)
    logger.info("PRICE OPTIMIZATION MODEL COMPLETE!")
    logger.info("="*60)

if __name__ == "__main__":
    main()