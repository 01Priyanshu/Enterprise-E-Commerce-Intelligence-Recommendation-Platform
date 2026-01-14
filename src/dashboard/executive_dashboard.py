"""
Executive Dashboard
High-level business metrics and KPIs
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data"""
    customers = pd.read_csv(DATA_DIR / "processed" / "customers_segmented.csv")
    orders = pd.read_csv(DATA_DIR / "features" / "order_features.csv")
    products = pd.read_csv(DATA_DIR / "features" / "product_features.csv")
    
    # Convert dates
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    return customers, orders, products

def main():
    # Title
    st.title("📊 Executive Dashboard")
    st.markdown("### E-Commerce Intelligence Platform")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        customers, orders, products = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = orders['order_date'].min()
    max_date = orders['order_date'].max()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    if len(date_range) == 2:
        mask = (orders['order_date'] >= pd.to_datetime(date_range[0])) & \
               (orders['order_date'] <= pd.to_datetime(date_range[1]))
        filtered_orders = orders[mask]
    else:
        filtered_orders = orders
    
    # === KEY METRICS ===
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_orders['final_amount'].sum()
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{total_revenue/len(filtered_orders):.0f} avg/order"
        )
    
    with col2:
        total_orders = len(filtered_orders)
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            delta=f"{total_orders/filtered_orders['customer_id'].nunique():.1f} per customer"
        )
    
    with col3:
        total_customers = customers['customer_id'].nunique()
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            delta=f"{(customers['total_orders'] > 0).sum()/total_customers*100:.0f}% active"
        )
    
    with col4:
        avg_order_value = filtered_orders['final_amount'].mean()
        st.metric(
            label="Avg Order Value",
            value=f"${avg_order_value:.2f}",
            delta=f"{filtered_orders['num_items'].mean():.1f} items/order"
        )
    
    st.markdown("---")
    
    # === REVENUE TRENDS ===
    st.header("💰 Revenue Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue over time
        daily_revenue = filtered_orders.groupby(filtered_orders['order_date'].dt.date)['final_amount'].sum().reset_index()
        
        fig = px.line(
            daily_revenue,
            x='order_date',
            y='final_amount',
            title='Daily Revenue Trend',
            labels={'order_date': 'Date', 'final_amount': 'Revenue ($)'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Orders over time
        daily_orders = filtered_orders.groupby(filtered_orders['order_date'].dt.date).size().reset_index()
        daily_orders.columns = ['date', 'count']
        
        fig = px.bar(
            daily_orders,
            x='date',
            y='count',
            title='Daily Order Volume',
            labels={'date': 'Date', 'count': 'Number of Orders'}
        )
        fig.update_traces(marker_color='#2ca02c')
        st.plotly_chart(fig, use_container_width=True)
    
    # === CUSTOMER SEGMENTATION ===
    st.header("👥 Customer Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment distribution
        segment_counts = customers['segment_name'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Distribution by Segment',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by segment
        segment_revenue = customers.groupby('segment_name')['total_spent'].sum().sort_values(ascending=True)
        
        fig = px.bar(
            x=segment_revenue.values,
            y=segment_revenue.index,
            orientation='h',
            title='Total Revenue by Customer Segment',
            labels={'x': 'Revenue ($)', 'y': 'Segment'}
        )
        fig.update_traces(marker_color='#ff7f0e')
        st.plotly_chart(fig, use_container_width=True)
    
    # === PRODUCT PERFORMANCE ===
    st.header("🛍️ Product Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products by revenue (simulated)
        top_products = products.nlargest(10, 'price')[['product_name', 'price', 'category']]
        top_products['revenue'] = top_products['price'] * 100  # Simulated sales
        
        fig = px.bar(
            top_products,
            x='revenue',
            y='product_name',
            orientation='h',
            title='Top 10 Products by Revenue',
            color='category',
            labels={'revenue': 'Revenue ($)', 'product_name': 'Product'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category distribution
        category_counts = products['category'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title='Products by Category',
            labels={'x': 'Category', 'y': 'Number of Products'}
        )
        fig.update_traces(marker_color='#9467bd')
        st.plotly_chart(fig, use_container_width=True)
    
    # === ORDER ANALYSIS ===
    st.header("📦 Order Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method distribution
        payment_dist = filtered_orders['payment_method'].value_counts()
        
        fig = px.pie(
            values=payment_dist.values,
            names=payment_dist.index,
            title='Payment Method Distribution',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Order status
        status_dist = filtered_orders['order_status'].value_counts()
        
        fig = px.bar(
            x=status_dist.index,
            y=status_dist.values,
            title='Order Status Distribution',
            labels={'x': 'Status', 'y': 'Count'}
        )
        fig.update_traces(marker_color='#e377c2')
        st.plotly_chart(fig, use_container_width=True)
    
    # === DATA TABLE ===
    st.header("📋 Recent Orders")
    st.dataframe(
        filtered_orders[['order_id', 'customer_id', 'order_date', 'final_amount', 'order_status']]
        .sort_values('order_date', ascending=False)
        .head(100),
        use_container_width=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard generated by E-Commerce Intelligence Platform*")

if __name__ == "__main__":
    main()