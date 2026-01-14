"""
Customer Churn Prediction using Random Forest
Identifies customers at risk of churning
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import MODELS_DIR, DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class ChurnPredictor:
    """Predict customer churn risk"""
    
    def __init__(self):
        """Initialize churn prediction model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = []
        self.model_path = MODELS_DIR / "churn_prediction"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create churn target: no purchase in last 90 days"""
        logger.info("Creating churn target variable...")
        
        # Define churned customers (no purchase in 90+ days, but have purchased before)
        df['is_churned'] = (
            (df['days_since_last_order'] > 90) & 
            (df['total_orders'] > 0)
        ).astype(int)
        
        churned_count = df['is_churned'].sum()
        active_count = len(df) - churned_count
        
        logger.info(f"Churn distribution:")
        logger.info(f"  Churned (1): {churned_count} ({churned_count/len(df)*100:.1f}%)")
        logger.info(f"  Active (0): {active_count} ({active_count/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for churn prediction"""
        logger.info("Preparing features...")
        
        # Select features
        features = [
            'total_orders',
            'total_spent',
            'avg_order_value',
            'customer_tenure_days',
            'days_since_last_order',
            'purchase_frequency',
            'rfm_score',
            'recency_score',
            'frequency_score',
            'monetary_score',
            'age',
            'total_items',
            'avg_items_per_order'
        ]
        
        self.feature_names = features
        
        # Prepare feature matrix and target
        X = df[features].fillna(0)
        y = df['is_churned']
        
        logger.info(f"Prepared {len(X)} samples with {len(features)} features")
        return X, y
    
    def train(self, df: pd.DataFrame, test_size=0.2):
        """Train the churn prediction model"""
        logger.info("Training churn prediction model...")
        
        # Create target
        df = self.create_target(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        logger.info("\nModel Performance:")
        logger.info("\n" + classification_report(y_test, y_pred, 
                                                  target_names=['Active', 'Churned']))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"ROC-AUC Score: {roc_auc:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        logger.info(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 5 Important Features:")
        for idx, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc
        }
    
    def predict(self, df: pd.DataFrame):
        """Predict churn probability for customers"""
        X, _ = self.prepare_features(df)
        predictions = self.model.predict_proba(X)[:, 1]
        return predictions
    
    def save_model(self):
        """Save the trained model"""
        model_file = self.model_path / "random_forest_model.pkl"
        features_file = self.model_path / "features.pkl"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.feature_names, features_file)
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self):
        """Load a trained model"""
        model_file = self.model_path / "random_forest_model.pkl"
        features_file = self.model_path / "features.pkl"
        
        self.model = joblib.load(model_file)
        self.feature_names = joblib.load(features_file)
        
        logger.info(f"Model loaded from {model_file}")

def main():
    """Train and save churn prediction model"""
    logger.info("="*60)
    logger.info("CHURN PREDICTION MODEL TRAINING")
    logger.info("="*60)
    
    # Load data
    features_path = DATA_DIR / "features" / "customer_features.csv"
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} customers")
    
    # Train model
    predictor = ChurnPredictor()
    results = predictor.train(df)
    
    # Save model
    predictor.save_model()
    
    logger.info("="*60)
    logger.info("CHURN PREDICTION MODEL COMPLETE!")
    logger.info("="*60)

if __name__ == "__main__":
    main()