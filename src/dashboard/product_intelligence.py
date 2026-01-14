"""
Product Intelligence Dashboard
Product analytics, recommendations, and pricing insights
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title="Product Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load product and order data"""
    products = pd.read_csv(DATA_DIR / "features" / "product_features.csv")
    orders = pd.read_csv(DATA_DIR / "features" / "order_features.csv")
    
    return products, orders

def main():
    # Title
    st.title("🛍️ Product Intelligence Dashboard")
    st.markdown("### Product Analytics, Recommendations & Pricing Insights")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading product data..."):
        products, orders = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    selected_category = st.sidebar.multiselect(
        "Category",
        options=products['category'].unique().tolist(),
        default=products['category'].unique().tolist()
    )
    
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(products['price'].min()),
        max_value=float(products['price'].max()),
        value=(float(products['price'].min()), float(products['price'].max()))
    )
    
    # Filter products
    filtered_products = products[
        (products['category'].isin(selected_category)) &
        (products['price'].between(price_range[0], price_range[1]))
    ]
    
    # === KEY METRICS ===
    st.header("📊 Product Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Products",
            f"{len(filtered_products):,}",
            delta=f"{len(filtered_products)/len(products)*100:.0f}% of catalog"
        )
    
    with col2:
        avg_price = filtered_products['price'].mean()
        st.metric(
            "Avg Price",
            f"${avg_price:.2f}",
            delta=f"${filtered_products['price'].std():.2f} std"
        )
    
    with col3:
        avg_rating = filtered_products['rating'].mean()
        st.metric(
            "Avg Rating",
            f"{avg_rating:.2f} ⭐",
            delta=f"{(avg_rating/5*100):.0f}% score"
        )
    
    with col4:
        in_stock = (filtered_products['stock_quantity'] > 0).sum()
        st.metric(
            "In Stock",
            f"{in_stock:,}",
            delta=f"{in_stock/len(filtered_products)*100:.0f}%"
        )
    
    st.markdown("---")
    
    # === PRODUCT PERFORMANCE ===
    st.header("🏆 Top Performing Products")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top rated products
        top_rated = filtered_products.nlargest(10, 'rating')[
            ['product_name', 'rating', 'price', 'category']
        ]
        
        fig = px.bar(
            top_rated,
            x='rating',
            y='product_name',
            orientation='h',
            title='Top 10 Highest Rated Products',
            color='category',
            labels={'rating': 'Rating', 'product_name': 'Product'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Most reviewed products
        most_reviewed = filtered_products.nlargest(10, 'num_reviews')[
            ['product_name', 'num_reviews', 'rating']
        ]
        
        fig = px.bar(
            most_reviewed,
            x='num_reviews',
            y='product_name',
            orientation='h',
            title='Top 10 Most Reviewed Products',
            color='rating',
            color_continuous_scale='Viridis',
            labels={'num_reviews': 'Number of Reviews', 'product_name': 'Product'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === PRICING ANALYSIS ===
    st.header("💰 Pricing Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by category
        fig = px.box(
            filtered_products,
            x='category',
            y='price',
            title='Price Distribution by Category',
            labels={'price': 'Price ($)', 'category': 'Category'},
            color='category'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit margin distribution
        fig = px.histogram(
            filtered_products,
            x='profit_margin',
            nbins=30,
            title='Profit Margin Distribution',
            labels={'profit_margin': 'Profit Margin (%)', 'count': 'Products'},
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === INVENTORY MANAGEMENT ===
    st.header("📦 Inventory Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stock status distribution
        stock_status_counts = filtered_products['stock_status'].value_counts()
        
        fig = px.pie(
            values=stock_status_counts.values,
            names=stock_status_counts.index,
            title='Stock Status Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Low stock products
        low_stock = filtered_products[
            filtered_products['stock_quantity'].between(1, 50)
        ].nlargest(10, 'price')
        
        fig = px.bar(
            low_stock,
            x='stock_quantity',
            y='product_name',
            orientation='h',
            title='High-Value Products with Low Stock',
            color='price',
            color_continuous_scale='Reds',
            labels={'stock_quantity': 'Stock Quantity', 'product_name': 'Product'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === CATEGORY ANALYSIS ===
    st.header("📂 Category Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Products by category
        category_counts = filtered_products['category'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title='Number of Products by Category',
            labels={'x': 'Category', 'y': 'Count'},
            color=category_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average rating by category
        avg_rating_by_cat = filtered_products.groupby('category')['rating'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=avg_rating_by_cat.index,
            y=avg_rating_by_cat.values,
            title='Average Rating by Category',
            labels={'x': 'Category', 'y': 'Average Rating'},
            color=avg_rating_by_cat.values,
            color_continuous_scale='Greens'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # === PRICE OPTIMIZATION RECOMMENDATIONS ===
    st.header("💡 Price Optimization Insights")
    
    # Simulate price recommendations
    filtered_products['optimal_price'] = filtered_products['price'] * np.random.uniform(0.95, 1.05, len(filtered_products))
    filtered_products['price_adjustment'] = ((filtered_products['optimal_price'] - filtered_products['price']) / filtered_products['price'] * 100).round(2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        increase_count = (filtered_products['price_adjustment'] > 5).sum()
        st.metric("Increase Price", f"{increase_count}", delta="Opportunity")
    
    with col2:
        maintain_count = (filtered_products['price_adjustment'].between(-5, 5)).sum()
        st.metric("Maintain Price", f"{maintain_count}", delta="Optimal")
    
    with col3:
        decrease_count = (filtered_products['price_adjustment'] < -5).sum()
        st.metric("Decrease Price", f"{decrease_count}", delta="Competitive")
    
    # Price adjustment opportunities
    st.subheader("🎯 Top Price Adjustment Opportunities")
    
    price_opps = filtered_products.nlargest(10, 'price_adjustment', keep='all')[
        ['product_name', 'price', 'optimal_price', 'price_adjustment', 'category']
    ]
    
    fig = px.bar(
        price_opps,
        x='price_adjustment',
        y='product_name',
        orientation='h',
        title='Top 10 Products for Price Increase',
        color='category',
        labels={'price_adjustment': 'Potential Increase (%)', 'product_name': 'Product'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # === PRODUCT SEARCH ===
    st.header("🔍 Product Search & Details")
    
    search_term = st.text_input("Search for a product:", "")
    
    if search_term:
        search_results = filtered_products[
            filtered_products['product_name'].str.contains(search_term, case=False, na=False)
        ]
        
        st.write(f"Found {len(search_results)} products")
        
        st.dataframe(
            search_results[['product_name', 'category', 'price', 'rating', 
                           'stock_quantity', 'profit_margin']],
            use_container_width=True
        )
    
    # === DETAILED PRODUCT TABLE ===
    st.header("📋 Product Catalog")
    
    display_columns = st.multiselect(
        "Select columns to display:",
        options=['product_id', 'product_name', 'category', 'brand', 'price', 
                'rating', 'num_reviews', 'stock_quantity', 'profit_margin'],
        default=['product_name', 'category', 'price', 'rating', 'stock_quantity']
    )
    
    if display_columns:
        st.dataframe(
            filtered_products[display_columns].head(50),
            use_container_width=True
        )
    
    # === DOWNLOAD DATA ===
    st.header("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_products.to_csv(index=False)
        st.download_button(
            label="Download Product Data (CSV)",
            data=csv,
            file_name="product_data.csv",
            mime="text/csv"
        )
    
    with col2:
        summary_stats = filtered_products.describe().to_csv()
        st.download_button(
            label="Download Summary Statistics (CSV)",
            data=summary_stats,
            file_name="product_summary.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*Product Intelligence powered by E-Commerce Intelligence Platform*")

if __name__ == "__main__":
    main()