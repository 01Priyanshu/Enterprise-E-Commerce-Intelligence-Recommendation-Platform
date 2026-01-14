"""
Product Recommendation System
Uses collaborative filtering to recommend products
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import MODELS_DIR, DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class ProductRecommender:
    """Collaborative filtering recommendation system"""
    
    def __init__(self, n_recommendations=5):
        """
        Initialize recommender
        
        Args:
            n_recommendations: Number of products to recommend
        """
        self.n_recommendations = n_recommendations
        self.customer_product_matrix = None
        self.similarity_matrix = None
        self.model_path = MODELS_DIR / "recommendation"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def create_interaction_matrix(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-product interaction matrix"""
        logger.info("Creating customer-product interaction matrix...")
        
        # For simplicity, we'll create synthetic product-order relationships
        # In a real scenario, you'd have order_items table
        
        # Extract unique customers and create interactions
        interactions = []
        
        for _, order in orders_df.iterrows():
            customer_id = order['customer_id']
            # Simulate 1-3 products per order
            n_products = np.random.randint(1, 4)
            
            # Sample random product IDs (you'd use actual order items)
            for _ in range(n_products):
                product_id = f"PROD{np.random.randint(1, 501):05d}"
                interactions.append({
                    'customer_id': customer_id,
                    'product_id': product_id,
                    'interaction': 1  # Binary: purchased or not
                })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Aggregate interactions (count purchases)
        interactions_df = interactions_df.groupby(
            ['customer_id', 'product_id']
        ).size().reset_index(name='purchase_count')
        
        logger.info(f"Created {len(interactions_df)} customer-product interactions")
        return interactions_df
    
    def build_matrix(self, interactions_df: pd.DataFrame):
        """Build customer-product matrix"""
        logger.info("Building recommendation matrix...")
        
        # Create pivot table
        matrix = interactions_df.pivot_table(
            index='customer_id',
            columns='product_id',
            values='purchase_count',
            fill_value=0
        )
        
        self.customer_product_matrix = matrix
        
        logger.info(f"Matrix shape: {matrix.shape[0]} customers x {matrix.shape[1]} products")
        return matrix
    
    def calculate_similarity(self):
        """Calculate customer-customer similarity"""
        logger.info("Calculating customer similarity...")
        
        # Calculate cosine similarity between customers
        self.similarity_matrix = cosine_similarity(self.customer_product_matrix)
        
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
    
    def train(self, orders_df: pd.DataFrame):
        """Train the recommendation system"""
        logger.info("Training recommendation system...")
        
        # Create interaction matrix
        interactions_df = self.create_interaction_matrix(orders_df)
        
        # Build customer-product matrix
        self.build_matrix(interactions_df)
        
        # Calculate similarity
        self.calculate_similarity()
        
        logger.info("Training complete!")
        
        # Save interactions for later use
        interactions_file = self.model_path / "interactions.csv"
        interactions_df.to_csv(interactions_file, index=False)
        
        return interactions_df
    
    def recommend(self, customer_id: str, n_recommendations: int = None):
        """
        Recommend products for a customer
        
        Args:
            customer_id: Customer ID
            n_recommendations: Number of recommendations (default: self.n_recommendations)
            
        Returns:
            List of recommended product IDs
        """
        if n_recommendations is None:
            n_recommendations = self.n_recommendations
        
        try:
            # Get customer index
            customer_idx = self.customer_product_matrix.index.get_loc(customer_id)
            
            # Get similar customers
            customer_similarities = self.similarity_matrix[customer_idx]
            similar_customers_idx = np.argsort(customer_similarities)[::-1][1:11]  # Top 10 similar
            
            # Get products these similar customers bought
            similar_customers_purchases = self.customer_product_matrix.iloc[similar_customers_idx]
            
            # Aggregate and score products
            product_scores = similar_customers_purchases.sum(axis=0)
            
            # Remove products already purchased by this customer
            customer_purchases = self.customer_product_matrix.loc[customer_id]
            product_scores = product_scores[customer_purchases == 0]
            
            # Get top recommendations
            recommendations = product_scores.nlargest(n_recommendations).index.tolist()
            
            return recommendations
            
        except KeyError:
            logger.warning(f"Customer {customer_id} not found in matrix")
            # Return popular products as fallback
            return self.get_popular_products(n_recommendations)
    
    def get_popular_products(self, n: int = 5):
        """Get most popular products as fallback"""
        product_popularity = self.customer_product_matrix.sum(axis=0)
        popular_products = product_popularity.nlargest(n).index.tolist()
        return popular_products
    
    def evaluate(self):
        """Evaluate recommendation quality"""
        logger.info("Evaluating recommendation system...")
        
        # Simple evaluation: coverage and diversity
        n_customers = self.customer_product_matrix.shape[0]
        n_products = self.customer_product_matrix.shape[1]
        
        # Test on sample customers
        sample_size = min(100, n_customers)
        sample_customers = self.customer_product_matrix.index[:sample_size]
        
        all_recommendations = set()
        for customer_id in sample_customers:
            recs = self.recommend(customer_id)
            all_recommendations.update(recs)
        
        coverage = len(all_recommendations) / n_products * 100
        
        logger.info(f"Recommendation Coverage: {coverage:.1f}% of products")
        logger.info(f"Average recommendations per customer: {self.n_recommendations}")
        
        return coverage
    
    def save_model(self):
        """Save the trained model"""
        matrix_file = self.model_path / "customer_product_matrix.pkl"
        similarity_file = self.model_path / "similarity_matrix.pkl"
        
        joblib.dump(self.customer_product_matrix, matrix_file)
        joblib.dump(self.similarity_matrix, similarity_file)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        matrix_file = self.model_path / "customer_product_matrix.pkl"
        similarity_file = self.model_path / "similarity_matrix.pkl"
        
        self.customer_product_matrix = joblib.load(matrix_file)
        self.similarity_matrix = joblib.load(similarity_file)
        
        logger.info(f"Model loaded from {self.model_path}")

def main():
    """Train and save recommendation system"""
    logger.info("="*60)
    logger.info("PRODUCT RECOMMENDATION SYSTEM TRAINING")
    logger.info("="*60)
    
    # Load data
    orders_path = DATA_DIR / "processed" / "orders_cleaned.csv"
    orders_df = pd.read_csv(orders_path)
    logger.info(f"Loaded {len(orders_df)} orders")
    
    # Train model
    recommender = ProductRecommender(n_recommendations=5)
    interactions_df = recommender.train(orders_df)
    
    # Evaluate
    coverage = recommender.evaluate()
    
    # Test recommendations
    sample_customer = orders_df['customer_id'].iloc[0]
    recommendations = recommender.recommend(sample_customer)
    logger.info(f"\nSample recommendations for {sample_customer}:")
    for i, prod in enumerate(recommendations, 1):
        logger.info(f"  {i}. {prod}")
    
    # Save model
    recommender.save_model()
    
    logger.info("="*60)
    logger.info("RECOMMENDATION SYSTEM COMPLETE!")
    logger.info("="*60)

if __name__ == "__main__":
    main()