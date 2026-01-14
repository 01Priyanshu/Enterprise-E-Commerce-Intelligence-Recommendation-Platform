"""
Complete data processing pipeline
Orchestrates cleaning and feature engineering
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR
from src.config.logging_config import setup_logger
from .cleaners import DataCleaner
from .feature_engineering import FeatureEngineer

logger = setup_logger(__name__)

class DataPipeline:
    """End-to-end data processing pipeline"""
    
    def __init__(self):
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.raw_data_path = DATA_DIR / "raw"
        self.processed_data_path = DATA_DIR / "processed"
        self.features_path = DATA_DIR / "features"
        
        # Create directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> dict:
        """Load all raw data files"""
        logger.info("Loading raw data...")
        
        data = {
            'customers': pd.read_csv(self.raw_data_path / "customers.csv"),
            'products': pd.read_csv(self.raw_data_path / "products.csv"),
            'orders': pd.read_csv(self.raw_data_path / "orders.csv"),
            'reviews': pd.read_csv(self.raw_data_path / "reviews.csv"),
        }
        
        logger.info(f"Loaded {len(data)} datasets")
        return data
    
    def clean_data(self, raw_data: dict) -> dict:
        """Clean all datasets"""
        logger.info("Cleaning all datasets...")
        
        cleaned_data = {
            'customers': self.cleaner.clean_customers(raw_data['customers']),
            'products': self.cleaner.clean_products(raw_data['products']),
            'orders': self.cleaner.clean_orders(raw_data['orders']),
            'reviews': self.cleaner.clean_reviews(raw_data['reviews']),
        }
        
        # Print cleaning summary
        summary = self.cleaner.get_cleaning_summary()
        logger.info(f"Cleaning complete: {summary}")
        
        return cleaned_data
    
    def engineer_features(self, cleaned_data: dict) -> dict:
        """Create engineered features"""
        logger.info("Engineering features...")
        
        features = self.feature_engineer.create_all_features(
            cleaned_data['customers'],
            cleaned_data['products'],
            cleaned_data['orders'],
            cleaned_data['reviews']
        )
        
        return features
    
    def save_processed_data(self, cleaned_data: dict, features: dict):
        """Save processed data and features"""
        logger.info("Saving processed data...")
        
        # Save cleaned data
        for name, df in cleaned_data.items():
            filepath = self.processed_data_path / f"{name}_cleaned.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name}: {filepath}")
        
        # Save features
        for name, df in features.items():
            filepath = self.features_path / f"{name}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name}: {filepath}")
    
    def run_pipeline(self):
        """Run complete data processing pipeline"""
        logger.info("="*60)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*60)
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Engineer features
        features = self.engineer_features(cleaned_data)
        
        # Save processed data
        self.save_processed_data(cleaned_data, features)
        
        logger.info("="*60)
        logger.info("DATA PROCESSING PIPELINE COMPLETE!")
        logger.info("="*60)
        
        return cleaned_data, features

def main():
    """Run the data processing pipeline"""
    pipeline = DataPipeline()
    cleaned_data, features = pipeline.run_pipeline()
    
    print("\nProcessed Data Summary:")
    print("-" * 60)
    for name, df in features.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} columns")
    print("-" * 60)

if __name__ == "__main__":
    main()