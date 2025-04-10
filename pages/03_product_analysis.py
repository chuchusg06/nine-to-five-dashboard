import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe

# Page configuration
st.set_page_config(
    page_title="Product Analysis - Nine To Five Marketing",
    page_icon="👔",
    layout="wide"
)

# Function to connect to Google Sheet - reusing from main app
@st.cache_resource
def connect_to_google_sheet():
    # Create credentials from the service account info
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Load credentials from env, secrets, or file
    if 'GOOGLE_CREDENTIALS' in os.environ:
        credentials_dict = json.loads(os.environ['GOOGLE_CREDENTIALS'])
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scope)
    else:
        # Try to get credentials from streamlit secrets
        try:
            credentials_dict = st.secrets["gcp_service_account"]
            credentials = Credentials.from_service_account_info(credentials_dict, scopes=scope)
        except Exception as e:
            st.sidebar.warning(f"Using local credentials file as fallback")
            # Fall back to file
            credentials = Credentials.from_service_account_file('credentials/service_account.json', scopes=scope)
    
    client = gspread.authorize(credentials)
    return client

# Function to load data from a specific sheet
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(sheet_name):
    try:
        client = connect_to_google_sheet()
        sheet_id = "1NG2ZZCVGNb3pIfnmyGd1T9ppqQ6ObOYbvFAX--m_wdo"
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        df = get_as_dataframe(worksheet, evaluate_formulas=True)
        
        # Clean the dataframe (remove empty rows and columns)
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Convert date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
        return df
    except Exception as e:
        st.error(f"Error loading data from {sheet_name}: {e}")
        return pd.DataFrame()

# Sidebar filters
st.sidebar.title("Product Analysis")

# Time period filter - defaulting to All Time
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
    index=4
)

# Product Category filter
products_data = load_data("Raw Data - Products")
if not products_data.empty and 'Product Category' in products_data.columns:
    categories = ["All Categories"] + sorted(products_data['Product Category'].unique().tolist())
    category_filter = st.sidebar.selectbox("Product Category", categories, index=0)
else:
    category_filter = "All Categories"

# Collection filter (9-to-5 vs 5-to-9)
if not products_data.empty and 'Collection' in products_data.columns:
    collections = ["All Collections"] + sorted(products_data['Collection'].unique().tolist())
    collection_filter = st.sidebar.selectbox("Collection", collections, index=0)
else:
    collection_filter = "All Collections"

# Function to filter data
def filter_products(df):
    if df.empty:
        return df
    
    # Apply category filter
    if category_filter != "All Categories" and 'Product Category' in df.columns:
        df = df[df['Product Category'] == category_filter]
    
    # Apply collection filter
    if collection_filter != "All Collections" and 'Collection' in df.columns:
        df = df[df['Collection'] == collection_filter]
    
    return df

# Apply filters
filtered_products = filter_products(products_data)

# Page Title
st.title("Product Analysis")
st.markdown("### Performance metrics for Nine To Five product portfolio")

# Product Performance Overview
st.markdown("## Product Performance Overview")

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

try:
    # Calculate KPIs
    total_revenue = filtered_products['Revenue'].sum() if not filtered_products.empty and 'Revenue' in filtered_products.columns else 0
    total_units = filtered_products['Unit Sold'].sum() if not filtered_products.empty and 'Unit Sold' in filtered_products.columns else 0
    avg_price = total_revenue / total_units if total_units > 0 else 0
    
    # Calculate margin if the data is available
    if not filtered_products.empty and 'COGS' in filtered_products.columns and 'Revenue' in filtered_products.columns:
        total_cogs = filtered_products['COGS'].sum()
        gross_profit = total_revenue - total_cogs
        margin_percentage = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
    else:
        margin_percentage = filtered_products['% Margin'].mean() if not filtered_products.empty and '% Margin' in filtered_products.columns else 0
    
    # Display KPIs
    with kpi1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with kpi2:
        st.metric("Units Sold", f"{total_units:,}")
    
    with kpi3:
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with kpi4:
        st.metric("Margin %", f"{margin_percentage:.1f}%")
except Exception as e:
    st.error(f"Error calculating KPIs: {e}")

# Product Category Analysis
st.markdown("## Product Category Performance")

col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_products.empty and 'Product Category' in filtered_products.columns and 'Revenue' in filtered_products.columns:
            # Group by category
            category_revenue = filtered_products.groupby('Product Category')['Revenue'].sum().reset_index()
            category_revenue = category_revenue.sort_values('Revenue', ascending=False)
            
            # Create revenue by category chart
            fig = px.bar(
                category_revenue,
                x='Product Category',
                y='Revenue',
                title='Revenue by Product Category',
                color='Product Category',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product category data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Product Category Performance chart: {e}")

with col2:
    try:
        if not filtered_products.empty and 'Product Category' in filtered_products.columns and 'Unit Sold' in filtered_products.columns:
            # Group by category
            category_units = filtered_products.groupby('Product Category')['Unit Sold'].sum().reset_index()
            category_units = category_units.sort_values('Unit Sold', ascending=False)
            
            # Create units sold by category chart
            fig = px.bar(
                category_units,
                x='Product Category',
                y='Unit Sold',
                title='Units Sold by Product Category',
                color='Product Category',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product category data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Units Sold by Category chart: {e}")

# 9-to-5 vs 5-to-9 Collection Comparison
st.markdown("## 9-to-5 vs 5-to-9 Collection Comparison")

try:
    if not filtered_products.empty and 'Collection' in filtered_products.columns:
        # Group by collection
        collection_metrics = filtered_products.groupby('Collection').agg({
            'Revenue': 'sum',
            'Unit Sold': 'sum'
        }).reset_index()
        
        # Calculate average price per collection
        collection_metrics['Avg Price'] = collection_metrics['Revenue'] / collection_metrics['Unit Sold']
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by collection
            fig = px.pie(
                collection_metrics,
                values='Revenue',
                names='Collection',
                title='Revenue Split by Collection',
                color='Collection',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD166']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Units by collection
            fig = px.pie(
                collection_metrics,
                values='Unit Sold',
                names='Collection',
                title='Units Sold Split by Collection',
                color='Collection',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD166']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Collection metrics table
        display_collection = collection_metrics.copy()
        display_collection['Revenue'] = display_collection['Revenue'].apply(lambda x: f"${x:,.2f}")
        display_collection['Avg Price'] = display_collection['Avg Price'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            display_collection,
            column_config={
                "Collection": st.column_config.TextColumn("Collection"),
                "Revenue": st.column_config.TextColumn("Revenue"),
                "Unit Sold": st.column_config.NumberColumn("Units Sold", format="%d"),
                "Avg Price": st.column_config.TextColumn("Avg Price")
            },
            use_container_width=True
        )
    else:
        st.info("No collection data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Collection Comparison: {e}")

# Top Products Analysis
st.markdown("## Top Performing Products")

try:
    if not filtered_products.empty and 'Product Name' in filtered_products.columns and 'Revenue' in filtered_products.columns:
        # Group by product
        product_metrics = filtered_products.groupby(['Product Name', 'Product Category', 'Collection']).agg({
            'Revenue': 'sum',
            'Unit Sold': 'sum',
            'Avg Price': 'mean',
            '% Margin': 'mean'
        }).reset_index()
        
        # Sort by revenue
        top_products = product_metrics.sort_values('Revenue', ascending=False).head(10)
        
        # Format for display
        display_products = top_products.copy()
        display_products['Revenue'] = display_products['Revenue'].apply(lambda x: f"${x:,.2f}")
        display_products['Avg Price'] = display_products['Avg Price'].apply(lambda x: f"${x:.2f}")
        display_products['% Margin'] = display_products['% Margin'].apply(lambda x: f"{x:.1f}%")
        
        # Display table
        st.dataframe(
            display_products,
            column_config={
                "Product Name": st.column_config.TextColumn("Product"),
                "Product Category": st.column_config.TextColumn("Category"),
                "Collection": st.column_config.TextColumn("Collection"),
                "Revenue": st.column_config.TextColumn("Revenue"),
                "Unit Sold": st.column_config.NumberColumn("Units Sold", format="%d"),
                "Avg Price": st.column_config.TextColumn("Avg Price"),
                "% Margin": st.column_config.TextColumn("Margin %")
            },
            use_container_width=True
        )
        
        # Top products chart
        fig = px.bar(
            top_products.head(10),
            x='Revenue',
            y='Product Name',
            orientation='h',
            title='Top 10 Products by Revenue',
            color='Product Category',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No product data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Top Products section: {e}")

# Margin Analysis
st.markdown("## Margin Analysis")

try:
    if not filtered_products.empty and 'Product Category' in filtered_products.columns and '% Margin' in filtered_products.columns:
        # Group by category
        margin_by_category = filtered_products.groupby('Product Category')['% Margin'].mean().reset_index()
        margin_by_category = margin_by_category.sort_values('% Margin', ascending=False)
        
        # Create margin chart
        fig = px.bar(
            margin_by_category,
            x='Product Category',
            y='% Margin',
            title='Average Margin % by Product Category',
            color='Product Category',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # If collection data is available, show margin by collection
        if 'Collection' in filtered_products.columns:
            margin_by_collection = filtered_products.groupby('Collection')['% Margin'].mean().reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Margin by collection
                fig = px.bar(
                    margin_by_collection,
                    x='Collection',
                    y='% Margin',
                    title='Average Margin % by Collection',
                    color='Collection',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD166']
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue vs. margin scatter plot
                if 'Revenue' in filtered_products.columns:
                    # Group by product
                    product_metrics = filtered_products.groupby(['Product Name']).agg({
                        'Revenue': 'sum',
                        '% Margin': 'mean',
                        'Product Category': 'first'
                    }).reset_index()
                    
                    # Create scatter plot
                    fig = px.scatter(
                        product_metrics,
                        x='Revenue',
                        y='% Margin',
                        color='Product Category',
                        size='Revenue',
                        hover_name='Product Name',
                        title='Revenue vs. Margin % by Product',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No margin data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Margin Analysis section: {e}")

# Footer
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")
