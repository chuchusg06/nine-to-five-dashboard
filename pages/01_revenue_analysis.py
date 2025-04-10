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
    page_title="Revenue Analysis - Nine To Five Marketing",
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

# Function to load data from a specific sheet - reusing from main app
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
st.sidebar.title("Revenue Analysis")

# Time period filter
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
    index=4  # Default to All Time
)

# Channel filter
channel_filter = st.sidebar.multiselect(
    "Channel",
    ["Social Organic", "Social Paid", "Affiliates", "Email", "All Channels"],
    default=["All Channels"]
)

# Load revenue data
revenue_data = load_data("Raw Data - Revenue")

# Apply no time filtering to ensure data shows up
def filter_by_channel(df):
    if "All Channels" in channel_filter or not channel_filter or 'Channel' not in df.columns or df.empty:
        return df
    else:
        return df[df['Channel'].isin(channel_filter)]

# Apply channel filter
filtered_revenue = filter_by_channel(revenue_data)

# Page Title
st.title("Revenue Analysis")
st.markdown("### Detailed revenue performance for Nine To Five Marketing")

# Revenue Over Time Chart
st.markdown("## Revenue Trends")

try:
    if not filtered_revenue.empty and 'Date' in filtered_revenue.columns and 'Revenue' in filtered_revenue.columns:
        # Group data by date
        daily_revenue = filtered_revenue.groupby('Date')['Revenue'].sum().reset_index()
        
        # Create the chart
        fig = px.line(
            daily_revenue,
            x='Date',
            y='Revenue',
            title='Daily Revenue',
            markers=True,
            color_discrete_sequence=['#19A7CE']
        )
        
        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=daily_revenue['Date'],
                y=daily_revenue['Revenue'].rolling(3).mean(),
                mode='lines',
                name='3-Day Moving Average',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Revenue Trend chart: {e}")

# Revenue by Channel and Product breakdown
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns:
            # Group data by channel
            channel_revenue = filtered_revenue.groupby('Channel')['Revenue'].sum().reset_index()
            channel_revenue = channel_revenue.sort_values('Revenue', ascending=False)
            
            # Calculate percentage of total
            total_rev = channel_revenue['Revenue'].sum()
            channel_revenue['Percentage'] = (channel_revenue['Revenue'] / total_rev * 100).round(1)
            
            # Add labels with percentages
            channel_revenue['Channel_Label'] = channel_revenue.apply(
                lambda x: f"{x['Channel']} ({x['Percentage']}%)", axis=1
            )
            
            # Create the chart
            fig = px.pie(
                channel_revenue,
                values='Revenue',
                names='Channel_Label',
                title='Revenue by Channel',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No channel data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Revenue by Channel chart: {e}")

# Load products data
products_data = load_data("Raw Data - Products")
filtered_products = filter_by_channel(products_data)

with col2:
    try:
        if not filtered_products.empty and 'Product Category' in filtered_products.columns:
            # Group data by product category
            product_revenue = filtered_products.groupby('Product Category')['Revenue'].sum().reset_index()
            product_revenue = product_revenue.sort_values('Revenue', ascending=False)
            
            # Calculate percentage of total
            total_rev = product_revenue['Revenue'].sum()
            product_revenue['Percentage'] = (product_revenue['Revenue'] / total_rev * 100).round(1)
            
            # Add labels with percentages
            product_revenue['Category_Label'] = product_revenue.apply(
                lambda x: f"{x['Product Category']} ({x['Percentage']}%)", axis=1
            )
            
            # Create the chart
            fig = px.pie(
                product_revenue,
                values='Revenue',
                names='Category_Label',
                title='Revenue by Product Category',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product category data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Revenue by Product Category chart: {e}")

# Revenue vs. Marketing Spend Analysis
st.markdown("## Revenue ROI Analysis")

try:
    if not filtered_revenue.empty and 'Date' in filtered_revenue.columns and 'Revenue' in filtered_revenue.columns and 'Marketing Spend' in filtered_revenue.columns:
        # Group data by date
        daily_metrics = filtered_revenue.groupby('Date').agg({
            'Revenue': 'sum',
            'Marketing Spend': 'sum'
        }).reset_index()
        
        # Calculate ROAS
        daily_metrics['ROAS'] = daily_metrics['Revenue'] / daily_metrics['Marketing Spend']
        
        # Create the chart
        fig = go.Figure()
        
        # Add revenue bars
        fig.add_trace(
            go.Bar(
                x=daily_metrics['Date'],
                y=daily_metrics['Revenue'],
                name='Revenue',
                marker_color='#19A7CE'
            )
        )
        
        # Add marketing spend bars
        fig.add_trace(
            go.Bar(
                x=daily_metrics['Date'],
                y=daily_metrics['Marketing Spend'],
                name='Marketing Spend',
                marker_color='#FF6B6B'
            )
        )
        
        # Add ROAS line
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['Date'],
                y=daily_metrics['ROAS'],
                name='ROAS',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='green', width=2)
            )
        )
        
        # Update layout for dual y-axis
        fig.update_layout(
            title='Revenue vs. Marketing Spend',
            barmode='group',
            yaxis=dict(title='Amount ($)'),
            yaxis2=dict(
                title='ROAS',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0.1, y=1.15, orientation='h')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue and marketing spend data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Revenue ROI Analysis chart: {e}")

# Key Metrics Table
st.markdown("## Revenue Metrics by Channel")

try:
    if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns:
        # Group by channel with multiple metrics
        channel_metrics = filtered_revenue.groupby('Channel').agg({
            'Revenue': 'sum',
            'Orders': 'sum',
            'New Customers': 'sum',
            'Web Visitors': 'sum',
            'Marketing Spend': 'sum'
        }).reset_index()
        
        # Calculate derived metrics
        channel_metrics['AOV'] = channel_metrics['Revenue'] / channel_metrics['Orders']
        channel_metrics['Conversion Rate (%)'] = (channel_metrics['Orders'] / channel_metrics['Web Visitors'] * 100).round(2)
        channel_metrics['ROAS'] = (channel_metrics['Revenue'] / channel_metrics['Marketing Spend']).round(2)
        channel_metrics['CAC'] = channel_metrics['Marketing Spend'] / channel_metrics['New Customers']
        
        # Format columns for display
        display_metrics = channel_metrics.copy()
        display_metrics['Revenue'] = display_metrics['Revenue'].apply(lambda x: f"${x:,.2f}")
        display_metrics['AOV'] = display_metrics['AOV'].apply(lambda x: f"${x:,.2f}")
        display_metrics['CAC'] = display_metrics['CAC'].apply(lambda x: f"${x:,.2f}")
        display_metrics['ROAS'] = display_metrics['ROAS'].apply(lambda x: f"{x:.2f}x")
        
        # Select columns for display
        display_cols = ['Channel', 'Revenue', 'Orders', 'AOV', 'Conversion Rate (%)', 'ROAS', 'CAC']
        
        # Display as table
        st.dataframe(
            display_metrics[display_cols],
            column_config={
                "Channel": st.column_config.TextColumn("Channel"),
                "Revenue": st.column_config.TextColumn("Revenue"),
                "Orders": st.column_config.NumberColumn("Orders", format="%d"),
                "AOV": st.column_config.TextColumn("Avg. Order Value"),
                "Conversion Rate (%)": st.column_config.NumberColumn("Conv. Rate (%)", format="%.2f%%"),
                "ROAS": st.column_config.TextColumn("ROAS"),
                "CAC": st.column_config.TextColumn("Cust. Acq. Cost")
            },
            use_container_width=True
        )
    else:
        st.info("No revenue metrics available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Revenue Metrics table: {e}")

# Footer
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")
