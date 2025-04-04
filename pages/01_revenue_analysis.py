# Save this as pages/01_revenue_analysis.py
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
    page_title="Revenue Analysis - Nine To Five",
    page_icon="💰",
    layout="wide"
)

# Function to connect to Google Sheet (reuse the same function from main dashboard)
@st.cache_resource
def connect_to_google_sheet():
    # Create credentials from the service account info
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Load credentials from env or file
    if 'GOOGLE_CREDENTIALS' in os.environ:
        credentials_dict = json.loads(os.environ['GOOGLE_CREDENTIALS'])
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scope)
    else:
        credentials = Credentials.from_service_account_file('credentials/service_account.json', scopes=scope)
    
    client = gspread.authorize(credentials)
    return client

# Connect to the Google Sheet
try:
    client = connect_to_google_sheet()
    sheet_id = "1NG2ZZCVGNb3pIfnmyGd1T9ppqQ6ObOYbvFAX--m_wdo"
    spreadsheet = client.open_by_key(sheet_id)
    
    st.sidebar.success("Connected to Google Sheet!")
except Exception as e:
    st.sidebar.error(f"Error connecting to Google Sheet: {e}")
    st.stop()

# Function to load data from a specific sheet
@st.cache_data(ttl=300)
def load_data(sheet_name):
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        df = get_as_dataframe(worksheet, evaluate_formulas=True)
        # Clean the dataframe (remove empty rows)
        df = df.dropna(how='all')
        # Convert date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data from {sheet_name}: {e}")
        return pd.DataFrame()

# Sidebar filters - similar to main dashboard
st.sidebar.title("Analysis Controls")

# Time period filter
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
    index=1
)

# Channel filter
channel_filter = st.sidebar.multiselect(
    "Channel",
    ["Social Organic", "Social Paid", "Affiliates", "Email", "All Channels"],
    default=["All Channels"]
)

# Date range selector (more detailed for analysis)
st.sidebar.markdown("### Custom Date Range")
use_custom_dates = st.sidebar.checkbox("Use custom date range instead")

if use_custom_dates:
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    if start_date > end_date:
        st.sidebar.error("End date must be after start date")

# Function to filter data based on time period or custom dates
def filter_by_time(df):
    if 'Date' not in df.columns or df.empty:
        return df
        
    if use_custom_dates:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
        return df[(df['Date'] >= start) & (df['Date'] <= end)]
    
    today = pd.to_datetime('today').normalize()
    
    if time_period == "Last 7 Days":
        start_date = today - timedelta(days=7)
    elif time_period == "Last 30 Days":
        start_date = today - timedelta(days=30)
    elif time_period == "Last 90 Days":
        start_date = today - timedelta(days=90)
    elif time_period == "Year to Date":
        start_date = pd.to_datetime(f"{today.year}-01-01")
    else:  # All Time
        return df
    
    return df[df['Date'] >= start_date]

# Function to filter data based on channel (reused from main dashboard)
def filter_by_channel(df):
    if "All Channels" in channel_filter or not channel_filter or 'Channel' not in df.columns or df.empty:
        return df
    else:
        return df[df['Channel'].isin(channel_filter)]

# Apply all filters to a dataframe
def apply_filters(df):
    if df.empty:
        return df
        
    df = filter_by_time(df)
    df = filter_by_channel(df)
    return df

# Load all required data
with st.spinner("Loading data from Google Sheets..."):
    revenue_data = load_data("Raw Data - Revenue")
    products_data = load_data("Raw Data - Products")
    campaigns_data = load_data("Raw Data - Campaigns")

# Apply filters
filtered_revenue = apply_filters(revenue_data)
filtered_products = apply_filters(products_data)
filtered_campaigns = apply_filters(campaigns_data)

# Page title
st.title("Revenue Analysis")
st.markdown("Detailed analysis of revenue performance across channels, products, and time")

# Top KPIs for Revenue Analysis
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

# Calculate KPIs with error handling
try:
    # Total Revenue
    total_revenue = filtered_revenue["Revenue"].sum() if not filtered_revenue.empty and "Revenue" in filtered_revenue.columns else 0
    
    # Revenue per order (AOV)
    total_orders = filtered_revenue["Orders"].sum() if not filtered_revenue.empty and "Orders" in filtered_revenue.columns else 0
    aov = total_revenue / total_orders if total_orders > 0 else 0
    
    # Revenue per customer
    total_customers = (filtered_revenue["New Customers"].sum() + filtered_revenue["Returning Customers"].sum()) if not filtered_revenue.empty and "New Customers" in filtered_revenue.columns and "Returning Customers" in filtered_revenue.columns else 0
    revenue_per_customer = total_revenue / total_customers if total_customers > 0 else 0
    
    # Revenue per marketing dollar spent (ROAS consolidated)
    total_marketing_spend = filtered_revenue["Marketing Spend"].sum() if not filtered_revenue.empty and "Marketing Spend" in filtered_revenue.columns else 0
    roas = total_revenue / total_marketing_spend if total_marketing_spend > 0 else 0
    
    with row1_col1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with row1_col2:
        st.metric("Average Order Value", f"${aov:.2f}")
    
    with row1_col3:
        st.metric("Revenue per Customer", f"${revenue_per_customer:.2f}")
    
    with row1_col4:
        st.metric("ROAS", f"{roas:.2f}x")
except Exception as e:
    st.error(f"Error calculating revenue KPIs: {e}")

# Revenue Over Time - Detailed trend analysis
st.markdown("## Revenue Trends")
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_revenue.empty and "Date" in filtered_revenue.columns and "Revenue" in filtered_revenue.columns:
            # Group by date
            daily_revenue = filtered_revenue.groupby('Date')['Revenue'].sum().reset_index()
            
            # Create line chart
            fig = px.line(
                daily_revenue,
                x='Date',
                y='Revenue',
                title='Daily Revenue Trend',
                markers=True
            )
            
            # Add 7-day moving average
            daily_revenue['7_Day_Avg'] = daily_revenue['Revenue'].rolling(window=7, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue['Date'],
                    y=daily_revenue['7_Day_Avg'],
                    mode='lines',
                    name='7-Day Moving Avg',
                    line=dict(color='red', dash='dash')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue trend data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating revenue trend chart: {e}")

with col2:
    try:
        if not filtered_revenue.empty and "Channel" in filtered_revenue.columns and "Revenue" in filtered_revenue.columns and "Date" in filtered_revenue.columns:
            # Group by date and channel
            channel_revenue_over_time = filtered_revenue.groupby(['Date', 'Channel'])['Revenue'].sum().reset_index()
            
            # Create stacked area chart
            fig = px.area(
                channel_revenue_over_time,
                x='Date',
                y='Revenue',
                color='Channel',
                title='Revenue by Channel Over Time',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No channel revenue trend data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating channel revenue trend chart: {e}")

# Revenue by Channel - Detailed breakdown
st.markdown("## Channel Analysis")
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_revenue.empty and "Channel" in filtered_revenue.columns and "Revenue" in filtered_revenue.columns:
            # Group by channel
            channel_revenue = filtered_revenue.groupby('Channel').agg({
                'Revenue': 'sum',
                'Orders': 'sum',
                'Marketing Spend': 'sum'
            }).reset_index()
            
            # Add ROAS calculation
            channel_revenue['ROAS'] = channel_revenue['Revenue'] / channel_revenue['Marketing Spend'].replace(0, float('nan'))
            
            # Sort by revenue
            channel_revenue = channel_revenue.sort_values('Revenue', ascending=False)
            
            # Create the horizontal bar chart
            fig = px.bar(
                channel_revenue,
                y='Channel',
                x='Revenue',
                title='Revenue by Channel',
                orientation='h',
                color='Channel',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create table with channel metrics
            channel_revenue['Revenue'] = channel_revenue['Revenue'].apply(lambda x: f"${x:,.2f}")
            channel_revenue['Marketing Spend'] = channel_revenue['Marketing Spend'].apply(lambda x: f"${x:,.2f}")
            channel_revenue['ROAS'] = channel_revenue['ROAS'].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "N/A")
            
            st.dataframe(
                channel_revenue,
                column_config={
                    "Channel": st.column_config.TextColumn("Channel"),
                    "Revenue": st.column_config.TextColumn("Revenue"),
                    "Orders": st.column_config.NumberColumn("Orders"),
                    "Marketing Spend": st.column_config.TextColumn("Marketing Spend"),
                    "ROAS": st.column_config.TextColumn("ROAS"),
                },
                use_container_width=True
            )
        else:
            st.info("No channel revenue data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating channel revenue breakdown: {e}")

with col2:
    try:
        if not filtered_revenue.empty and "Channel" in filtered_revenue.columns and "Revenue" in filtered_revenue.columns:
            # Calculate channel percentage and total
            channel_percentage = filtered_revenue.groupby('Channel')['Revenue'].sum().reset_index()
            total = channel_percentage['Revenue'].sum()
            
            if total > 0:
                channel_percentage['Percentage'] = (channel_percentage['Revenue'] / total) * 100
                
                # Create pie chart
                fig = px.pie(
                    channel_percentage,
                    values='Percentage',
                    names='Channel',
                    title='Revenue Distribution by Channel (%)',
                    color='Channel',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No revenue data to calculate distribution.")
        else:
            st.info("No channel revenue distribution data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating revenue distribution chart: {e}")

# Product Revenue Analysis
st.markdown("## Product Revenue Analysis")
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_products.empty and "Product Category" in filtered_products.columns and "Revenue" in filtered_products.columns:
            # Group by product category
            category_revenue = filtered_products.groupby('Product Category')['Revenue'].sum().reset_index()
            category_revenue = category_revenue.sort_values('Revenue', ascending=False)
            
            # Create bar chart
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
            st.info("No product category revenue data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating product category revenue chart: {e}")

with col2:
    try:
        if not filtered_products.empty and "Collection" in filtered_products.columns and "Revenue" in filtered_products.columns:
            # Group by collection
            collection_revenue = filtered_products.groupby('Collection')['Revenue'].sum().reset_index()
            collection_revenue = collection_revenue.sort_values('Revenue', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                collection_revenue,
                x='Collection',
                y='Revenue',
                title='Revenue by Collection',
                color='Collection',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No collection revenue data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating collection revenue chart: {e}")

# Campaign Revenue Analysis
st.markdown("## Campaign Performance")

try:
    if not filtered_campaigns.empty and "Campaign Name" in filtered_campaigns.columns and "Revenue" in filtered_campaigns.columns:
        # Select relevant columns
        campaign_data = filtered_campaigns[['Campaign Name', 'Channel', 'Campaign Type', 'Revenue', 'Spend', 'ROAS']].copy()
        
        # Sort by revenue
        campaign_data = campaign_data.sort_values('Revenue', ascending=False)
        
        # Convert to proper formats for display
        campaign_data['Revenue'] = campaign_data['Revenue'].apply(lambda x: f"${x:,.2f}")
        campaign_data['Spend'] = campaign_data['Spend'].apply(lambda x: f"${x:,.2f}")
        campaign_data['ROAS'] = campaign_data['ROAS'].apply(lambda x: f"{x:.2f}x")
        
        # Display as table
        st.dataframe(
            campaign_data,
            column_config={
                "Campaign Name": st.column_config.TextColumn("Campaign"),
                "Channel": st.column_config.TextColumn("Channel"),
                "Campaign Type": st.column_config.TextColumn("Type"),
                "Revenue": st.column_config.TextColumn("Revenue"),
                "Spend": st.column_config.TextColumn("Spend"),
                "ROAS": st.column_config.TextColumn("ROAS"),
            },
            use_container_width=True
        )
    else:
        st.info("No campaign revenue data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating campaign revenue table: {e}")

# Revenue Forecast (simple)
st.markdown("## Revenue Forecast")

try:
    if not filtered_revenue.empty and "Date" in filtered_revenue.columns and "Revenue" in filtered_revenue.columns and len(filtered_revenue) > 7:
        # Group by date
        daily_revenue = filtered_revenue.groupby('Date')['Revenue'].sum().reset_index()
        
        # Sort by date to ensure proper trend
        daily_revenue = daily_revenue.sort_values('Date')
        
        # Create basic trend line
        x = range(len(daily_revenue))
        y = daily_revenue['Revenue']
        
        # Add simple linear trend
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Create forecast for next 7 days
        last_date = daily_revenue['Date'].max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(7)]
        forecast_values = [intercept + slope * (len(daily_revenue) + i) for i in range(7)]
        
        # Create combined dataframe
        historical = pd.DataFrame({
            'Date': daily_revenue['Date'],
            'Revenue': daily_revenue['Revenue'],
            'Type': 'Historical'
        })
        
        forecast = pd.DataFrame({
            'Date': forecast_dates,
            'Revenue': forecast_values,
            'Type': 'Forecast'
        })
        
        combined_data = pd.concat([historical, forecast])
        
        # Create forecast chart
        fig = px.line(
            combined_data,
            x='Date',
            y='Revenue',
            color='Type',
            title='Revenue Trend and 7-Day Forecast',
            markers=True,
            color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate forecast metrics
        total_historical = historical['Revenue'].sum()
        total_forecast = forecast['Revenue'].sum()
        forecast_change = ((total_forecast / (total_historical / 7)) - 1) * 100
        
        # Display forecast metrics
        st.metric("7-Day Revenue Forecast", f"${total_forecast:,.2f}", f"{forecast_change:.1f}% vs. previous 7 days")
    else:
        st.info("Insufficient data for revenue forecasting. Need at least 7 days of data.")
except Exception as e:
    st.error(f"Error creating revenue forecast: {e}")

# Footer with update timestamp
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")