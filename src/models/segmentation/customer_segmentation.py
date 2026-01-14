"""
Customer Segmentation using K-Means Clustering
Groups customers based on purchase behavior
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import MODELS_DIR, DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class CustomerSegmentation:
    """Customer segmentation using K-Means clustering"""
    
    def __init__(self, n_clusters=4):
        """
        Initialize customer segmentation model
        
        Args:
            n_clusters: Number of customer segments
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = MODELS_DIR / "segmentation"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering"""
        logger.info("Preparing features for segmentation...")
        
        # Select relevant features for segmentation
        features = [
            'total_orders',
            'total_spent',
            'avg_order_value',
            'customer_tenure_days',
            'days_since_last_order',
            'purchase_frequency',
            'rfm_score'
        ]
        
        self.feature_names = features
        
        # Extract features and handle missing values
        X = df[features].copy()
        X = X.fillna(0)
        
        logger.info(f"Prepared {len(X)} samples with {len(features)} features")
        return X
    
    def train(self, df: pd.DataFrame):
        """Train the segmentation model"""
        logger.info(f"Training customer segmentation model with {self.n_clusters} clusters...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Get predictions
        df['segment'] = self.model.predict(X_scaled)
        
        # Evaluate
        silhouette = silhouette_score(X_scaled, df['segment'])
        davies_bouldin = davies_bouldin_score(X_scaled, df['segment'])
        
        logger.info(f"Training complete!")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        # Analyze segments
        self.analyze_segments(df)
        
        return df
    
    def analyze_segments(self, df: pd.DataFrame):
        """Analyze characteristics of each segment"""
        logger.info("Analyzing customer segments...")
        
        segment_summary = df.groupby('segment').agg({
            'customer_id': 'count',
            'total_orders': 'mean',
            'total_spent': 'mean',
            'avg_order_value': 'mean',
            'customer_tenure_days': 'mean',
            'rfm_score': 'mean'
        }).round(2)
        
        segment_summary.columns = [
            'Count', 'Avg Orders', 'Avg Spent', 
            'Avg Order Value', 'Avg Tenure', 'Avg RFM'
        ]
        
        # Name segments based on characteristics
        segment_names = {
            0: 'Low Value',
            1: 'Medium Value',
            2: 'High Value',
            3: 'VIP'
        }
        
        # Sort by total spent to assign names
        sorted_segments = segment_summary.sort_values('Avg Spent').index.tolist()
        segment_mapping = {old: segment_names[new] for new, old in enumerate(sorted_segments)}
        
        df['segment_name'] = df['segment'].map(segment_mapping)
        
        logger.info("\nSegment Summary:")
        for idx, row in segment_summary.iterrows():
            name = segment_mapping[idx]
            logger.info(f"\n{name} (Segment {idx}):")
            logger.info(f"  Customers: {int(row['Count'])}")
            logger.info(f"  Avg Orders: {row['Avg Orders']:.1f}")
            logger.info(f"  Avg Spent: ${row['Avg Spent']:.2f}")
            logger.info(f"  Avg Order Value: ${row['Avg Order Value']:.2f}")
        
        return df
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict segments for new customers"""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save_model(self):
        """Save the trained model"""
        model_file = self.model_path / "kmeans_model.pkl"
        scaler_file = self.model_path / "scaler.pkl"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.feature_names, self.model_path / "features.pkl")
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self):
        """Load a trained model"""
        model_file = self.model_path / "kmeans_model.pkl"
        scaler_file = self.model_path / "scaler.pkl"
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_names = joblib.load(self.model_path / "features.pkl")
        
        logger.info(f"Model loaded from {model_file}")

def main():
    """Train and save customer segmentation model"""
    logger.info("="*60)
    logger.info("CUSTOMER SEGMENTATION MODEL TRAINING")
    logger.info("="*60)
    
    # Load data
    features_path = DATA_DIR / "features" / "customer_features.csv"
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} customers")
    
    # Train model
    segmentation = CustomerSegmentation(n_clusters=4)
    df_segmented = segmentation.train(df)
    
    # Save model
    segmentation.save_model()
    
    # Save segmented data
    output_path = DATA_DIR / "processed" / "customers_segmented.csv"
    df_segmented.to_csv(output_path, index=False)
    logger.info(f"Segmented data saved to {output_path}")
    
    logger.info("="*60)
    logger.info("CUSTOMER SEGMENTATION COMPLETE!")
    logger.info("="*60)

if __name__ == "__main__":
    main()