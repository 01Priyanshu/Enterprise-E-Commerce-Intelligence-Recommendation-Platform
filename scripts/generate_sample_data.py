"""
Generate realistic e-commerce sample data
Creates customers, products, orders, and reviews datasets
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class EcommerceDataGenerator:
    """Generate realistic e-commerce datasets"""
    
    def __init__(self, n_customers=5000, n_products=500, n_orders=15000):
        """
        Initialize data generator
        
        Args:
            n_customers: Number of customers to generate
            n_products: Number of products to generate
            n_orders: Number of orders to generate
        """
        self.n_customers = n_customers
        self.n_products = n_products
        self.n_orders = n_orders
        
        logger.info(f"Initializing data generator: {n_customers} customers, "
                   f"{n_products} products, {n_orders} orders")
    
    def generate_customers(self) -> pd.DataFrame:
        """Generate customer data"""
        logger.info("Generating customer data...")
        
        customers = []
        for i in range(self.n_customers):
            customer = {
                'customer_id': f'CUST{i+1:06d}',
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'email': fake.email(),
                'phone': fake.phone_number(),
                'address': fake.street_address(),
                'city': fake.city(),
                'state': fake.state(),
                'zip_code': fake.zipcode(),
                'country': 'USA',
                'registration_date': fake.date_between(start_date='-3y', end_date='today'),
                'age': np.random.randint(18, 75),
                'gender': np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04]),
                'income_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
                'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                                 p=[0.5, 0.3, 0.15, 0.05]),
            }
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        logger.info(f"Generated {len(df)} customers")
        return df
    
    def generate_products(self) -> pd.DataFrame:
        """Generate product catalog"""
        logger.info("Generating product data...")
        
        categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 
                     'Sports', 'Beauty', 'Toys', 'Food & Beverage']
        
        products = []
        for i in range(self.n_products):
            category = np.random.choice(categories)
            base_price = np.random.uniform(10, 1000)
            
            product = {
                'product_id': f'PROD{i+1:05d}',
                'product_name': f"{fake.word().title()} {fake.word().title()}",
                'category': category,
                'sub_category': f"{category} - {fake.word().title()}",
                'brand': fake.company(),
                'price': round(base_price, 2),
                'cost': round(base_price * 0.6, 2),  # 40% margin
                'stock_quantity': np.random.randint(0, 1000),
                'weight_kg': round(np.random.uniform(0.1, 20), 2),
                'rating': round(np.random.uniform(3.0, 5.0), 1),
                'num_reviews': np.random.randint(0, 500),
                'is_active': np.random.choice([True, False], p=[0.9, 0.1]),
                'launch_date': fake.date_between(start_date='-2y', end_date='today'),
            }
            products.append(product)
        
        df = pd.DataFrame(products)
        logger.info(f"Generated {len(df)} products")
        return df
    
    def generate_orders(self, customers_df: pd.DataFrame, 
                       products_df: pd.DataFrame) -> pd.DataFrame:
        """Generate order transactions"""
        logger.info("Generating order data...")
        
        orders = []
        customer_ids = customers_df['customer_id'].tolist()
        product_ids = products_df['product_id'].tolist()
        product_prices = dict(zip(products_df['product_id'], products_df['price']))
        
        for i in range(self.n_orders):
            customer_id = np.random.choice(customer_ids)
            order_date = fake.date_time_between(start_date='-1y', end_date='now')
            
            # Number of items in order
            num_items = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            selected_products = np.random.choice(product_ids, size=num_items, replace=False)
            
            total_amount = sum([product_prices[pid] for pid in selected_products])
            discount = round(total_amount * np.random.uniform(0, 0.2), 2)
            shipping_cost = round(np.random.uniform(0, 15), 2)
            final_amount = round(total_amount - discount + shipping_cost, 2)
            
            order = {
                'order_id': f'ORD{i+1:08d}',
                'customer_id': customer_id,
                'order_date': order_date,
                'num_items': num_items,
                'total_amount': round(total_amount, 2),
                'discount': discount,
                'shipping_cost': shipping_cost,
                'final_amount': final_amount,
                'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'],
                                                   p=[0.5, 0.25, 0.2, 0.05]),
                'shipping_address': fake.address(),
                'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled', 'Returned'],
                                                p=[0.85, 0.05, 0.05, 0.05]),
                'delivery_date': order_date + timedelta(days=np.random.randint(1, 10)),
            }
            orders.append(order)
        
        df = pd.DataFrame(orders)
        logger.info(f"Generated {len(df)} orders")
        return df
    
    def generate_reviews(self, orders_df: pd.DataFrame, 
                        products_df: pd.DataFrame) -> pd.DataFrame:
        """Generate product reviews"""
        logger.info("Generating review data...")
        
        # About 30% of orders have reviews
        n_reviews = int(self.n_orders * 0.3)
        
        reviews = []
        product_ids = products_df['product_id'].tolist()
        
        positive_comments = [
            "Great product! Highly recommend.",
            "Excellent quality and fast shipping.",
            "Love it! Exactly as described.",
            "Perfect! Will buy again.",
            "Amazing value for money.",
        ]
        
        negative_comments = [
            "Not as expected. Disappointed.",
            "Poor quality. Would not recommend.",
            "Arrived damaged.",
            "Overpriced for what you get.",
            "Not worth the money.",
        ]
        
        for i in range(n_reviews):
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.15, 0.35, 0.4])
            
            if rating >= 4:
                comment = np.random.choice(positive_comments)
            else:
                comment = np.random.choice(negative_comments)
            
            review = {
                'review_id': f'REV{i+1:08d}',
                'product_id': np.random.choice(product_ids),
                'customer_id': orders_df.iloc[i % len(orders_df)]['customer_id'],
                'rating': rating,
                'review_text': comment,
                'review_date': fake.date_time_between(start_date='-1y', end_date='now'),
                'helpful_count': np.random.randint(0, 100),
                'verified_purchase': np.random.choice([True, False], p=[0.8, 0.2]),
            }
            reviews.append(review)
        
        df = pd.DataFrame(reviews)
        logger.info(f"Generated {len(df)} reviews")
        return df
    
    def generate_all_data(self, save_to_csv=True):
        """Generate all datasets and optionally save to CSV"""
        logger.info("Starting complete data generation...")
        
        # Generate all datasets
        customers_df = self.generate_customers()
        products_df = self.generate_products()
        orders_df = self.generate_orders(customers_df, products_df)
        reviews_df = self.generate_reviews(orders_df, products_df)
        
        if save_to_csv:
            # Save to CSV files
            raw_data_path = DATA_DIR / "raw"
            raw_data_path.mkdir(parents=True, exist_ok=True)
            
            customers_df.to_csv(raw_data_path / "customers.csv", index=False)
            products_df.to_csv(raw_data_path / "products.csv", index=False)
            orders_df.to_csv(raw_data_path / "orders.csv", index=False)
            reviews_df.to_csv(raw_data_path / "reviews.csv", index=False)
            
            logger.info(f"All data saved to {raw_data_path}")
        
        return {
            'customers': customers_df,
            'products': products_df,
            'orders': orders_df,
            'reviews': reviews_df,
        }

def main():
    """Main function to generate sample data"""
    print("=" * 60)
    print("E-COMMERCE DATA GENERATOR")
    print("=" * 60)
    
    # Create generator
    generator = EcommerceDataGenerator(
        n_customers=5000,
        n_products=500,
        n_orders=15000
    )
    
    # Generate all data
    data = generator.generate_all_data(save_to_csv=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Customers: {len(data['customers']):,}")
    print(f"Products: {len(data['products']):,}")
    print(f"Orders: {len(data['orders']):,}")
    print(f"Reviews: {len(data['reviews']):,}")
    print("\nData saved to: data/raw/")
    print("=" * 60)

if __name__ == "__main__":
    main()