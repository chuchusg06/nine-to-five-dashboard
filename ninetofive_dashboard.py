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
    page_title="Nine To Five Marketing Dashboard",
    page_icon="👔",
    layout="wide"
)

# Function to connect to Google Sheet
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

# Connect to the Google Sheet
try:
    client = connect_to_google_sheet()
    # Replace with your Google Sheet ID
    sheet_id = "1NG2ZZCVGNb3pIfnmyGd1T9ppqQ6ObOYbvFAX--m_wdo"
    spreadsheet = client.open_by_key(sheet_id)
    
    st.sidebar.success("Connected to Google Sheet!")
    
    # Add debugging right after connection
    debug_expander = st.sidebar.expander("🔍 Debug Information", expanded=True)
    
    # Show worksheets in the spreadsheet
    worksheet_list = spreadsheet.worksheets()
    debug_expander.write(f"Available worksheets in the spreadsheet:")
    for i, worksheet in enumerate(worksheet_list):
        debug_expander.write(f"{i+1}. {worksheet.title} - {worksheet.row_count} rows")
    
    # Add more debugging info
    debug_expander.write("---")
    debug_expander.write("### Google Sheet Connection:")
    debug_expander.write(f"Connected to sheet: {spreadsheet.title}")
    debug_expander.write(f"Spreadsheet ID: {sheet_id}")
    
except Exception as e:
    st.sidebar.error(f"Error connecting to Google Sheet: {e}")
    st.stop()

# Function to load data from a specific sheet
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(sheet_name):
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        df = get_as_dataframe(worksheet, evaluate_formulas=True)
        
        # Clean the dataframe (remove empty rows and columns)
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Print raw date values to debug
        debug_expander.write(f"Raw date values in {sheet_name}: {df['Date'].head(3).tolist() if 'Date' in df.columns else 'No Date column'}")
        
        # Convert date column to datetime if it exists
        if 'Date' in df.columns:
            try:
                # Try different date formats
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                debug_expander.write(f"Converted dates in {sheet_name}: {df['Date'].head(3)}")
            except Exception as date_error:
                debug_expander.error(f"Date conversion error in {sheet_name}: {date_error}")
                
        return df
    except Exception as e:
        debug_expander.error(f"Error loading data from {sheet_name}: {e}")
        return pd.DataFrame()

# Sidebar filters
st.sidebar.title("Dashboard Controls")

# Time period filter
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
    index=4  # Changed from 1 to 4 to make "All Time" the default
)

# Channel filter
channel_filter = st.sidebar.multiselect(
    "Channel",
    ["Social Organic", "Social Paid", "Affiliates", "Email", "All Channels"],
    default=["All Channels"]
)

# Platform filter (for social media)
platform_filter = st.sidebar.multiselect(
    "Social Platforms",
    ["Instagram", "TikTok", "Facebook", "All Platforms"],
    default=["All Platforms"]
)

# Function to filter data based on time period
def filter_by_time(df):
    # For debugging - return all data regardless of time period
    return df
    
    # Original code commented out temporarily
    '''
    if 'Date' not in df.columns or df.empty:
        return df
        
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
    '''

# Function to filter data based on channel
def filter_by_channel(df):
    if "All Channels" in channel_filter or not channel_filter or 'Channel' not in df.columns or df.empty:
        return df
    else:
        return df[df['Channel'].isin(channel_filter)]

# Function to filter data based on platform
def filter_by_platform(df):
    if "All Platforms" in platform_filter or not platform_filter or 'Platforms' not in df.columns or df.empty:
        return df
    else:
        return df[df['Platforms'].isin(platform_filter)]

# Apply all filters to a dataframe
def apply_filters(df):
    if df.empty:
        return df
        
    df = filter_by_time(df)
    df = filter_by_channel(df)
    df = filter_by_platform(df)
    return df

# Load data from all sheets
with st.spinner("Loading data from Google Sheets..."):
    # Try to load data from each sheet with debug information
    try:
        revenue_data = load_data("Raw Data - Revenue")
        debug_expander.write(f"Revenue data: {len(revenue_data)} rows, Columns: {revenue_data.columns.tolist() if not revenue_data.empty else 'None'}")
        if not revenue_data.empty and 'Revenue' in revenue_data.columns:
            debug_expander.write(f"Total revenue in raw data: {revenue_data['Revenue'].sum()}")
    except Exception as e:
        debug_expander.error(f"Failed to load Revenue data: {e}")
        revenue_data = pd.DataFrame()
    
    try:
        products_data = load_data("Raw Data - Products")
        debug_expander.write(f"Products data: {len(products_data)} rows, Columns: {products_data.columns.tolist() if not products_data.empty else 'None'}")
    except Exception as e:
        debug_expander.error(f"Failed to load Products data: {e}")
        products_data = pd.DataFrame()
    
    try:
        customers_data = load_data("Raw Data - Customers")
        debug_expander.write(f"Customers data: {len(customers_data)} rows, Columns: {customers_data.columns.tolist() if not customers_data.empty else 'None'}")
    except Exception as e:
        debug_expander.error(f"Failed to load Customers data: {e}")
        customers_data = pd.DataFrame()
    
    try:
        social_data = load_data("Raw Data - Social Media")
        debug_expander.write(f"Social data: {len(social_data)} rows, Columns: {social_data.columns.tolist() if not social_data.empty else 'None'}")
    except Exception as e:
        debug_expander.error(f"Failed to load Social Media data: {e}")
        social_data = pd.DataFrame()
    
    try:
        affiliates_data = load_data("Raw Data - Affiliates")
        debug_expander.write(f"Affiliates data: {len(affiliates_data)} rows, Columns: {affiliates_data.columns.tolist() if not affiliates_data.empty else 'None'}")
    except Exception as e:
        debug_expander.error(f"Failed to load Affiliates data: {e}")
        affiliates_data = pd.DataFrame()
    
    try:
        email_data = load_data("Raw Data - Email")
        debug_expander.write(f"Email data: {len(email_data)} rows, Columns: {email_data.columns.tolist() if not email_data.empty else 'None'}")
    except Exception as e:
        debug_expander.error(f"Failed to load Email data: {e}")
        email_data = pd.DataFrame()
    
    try:
        campaigns_data = load_data("Raw Data - Campaigns")
        debug_expander.write(f"Campaigns data: {len(campaigns_data)} rows, Columns: {campaigns_data.columns.tolist() if not campaigns_data.empty else 'None'}")
    except Exception as e:
        debug_expander.error(f"Failed to load Campaigns data: {e}")
        campaigns_data = pd.DataFrame()

# Apply filters to the data
filtered_revenue = apply_filters(revenue_data)
debug_expander.write(f"Revenue data before filtering: {len(revenue_data)} rows, after filtering: {len(filtered_revenue)} rows")
if not filtered_revenue.empty and 'Revenue' in filtered_revenue.columns:
    debug_expander.write(f"Total revenue in filtered data: {filtered_revenue['Revenue'].sum()}")

filtered_products = apply_filters(products_data)
filtered_customers = apply_filters(customers_data)
filtered_social = apply_filters(social_data)
filtered_affiliates = apply_filters(affiliates_data)
filtered_email = apply_filters(email_data)
filtered_campaigns = apply_filters(campaigns_data)

# Additional debug info for filtered data
debug_expander.write("---")
debug_expander.write("### Filtered Data:")
debug_expander.write(f"Filtered revenue data: {len(filtered_revenue)} rows")
debug_expander.write(f"Filtered products data: {len(filtered_products)} rows")
debug_expander.write(f"Filtered customers data: {len(filtered_customers)} rows")
debug_expander.write(f"Filtered social data: {len(filtered_social)} rows")
debug_expander.write(f"Filtered affiliates data: {len(filtered_affiliates)} rows")
debug_expander.write(f"Filtered email data: {len(filtered_email)} rows")
debug_expander.write(f"Filtered campaigns data: {len(filtered_campaigns)} rows")

# Dashboard Title
st.title("Nine To Five Marketing Dashboard")
st.markdown("### Marketing performance metrics for transitional 9-to-5 and 5-to-9 clothing")

# KPI Row
st.markdown("## Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Calculate KPIs
try:
    # Total Revenue
    total_revenue = filtered_revenue["Revenue"].sum() if not filtered_revenue.empty and "Revenue" in filtered_revenue.columns else 0
    
    # Calculate month-over-month growth
    mom_growth = 0
    if not filtered_revenue.empty and 'Date' in filtered_revenue.columns and 'Revenue' in filtered_revenue.columns:
        current_month = pd.to_datetime('today').month
        current_year = pd.to_datetime('today').year
        
        current_month_data = filtered_revenue[
            (filtered_revenue['Date'].dt.month == current_month) & 
            (filtered_revenue['Date'].dt.year == current_year)
        ]
        
        prev_month = current_month - 1 if current_month > 1 else 12
        prev_year = current_year if current_month > 1 else current_year - 1
        
        prev_month_data = filtered_revenue[
            (filtered_revenue['Date'].dt.month == prev_month) & 
            (filtered_revenue['Date'].dt.year == prev_year)
        ]
        
        current_month_revenue = current_month_data["Revenue"].sum()
        prev_month_revenue = prev_month_data["Revenue"].sum()
        
        if prev_month_revenue > 0:
            mom_growth = ((current_month_revenue - prev_month_revenue) / prev_month_revenue) * 100

    # Average Order Value
    total_orders = filtered_revenue["Orders"].sum() if not filtered_revenue.empty and "Orders" in filtered_revenue.columns else 0
    aov = total_revenue / total_orders if total_orders > 0 else 0
    
    # Conversion Rate
    total_visitors = filtered_revenue["Web Visitors"].sum() if not filtered_revenue.empty and "Web Visitors" in filtered_revenue.columns else 0
    conversion_rate = (total_orders / total_visitors) * 100 if total_visitors > 0 else 0
    
    # New Customers
    new_customers = filtered_revenue["New Customers"].sum() if not filtered_revenue.empty and "New Customers" in filtered_revenue.columns else 0
    
    with kpi1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}", f"{mom_growth:.1f}% MoM")
    
    with kpi2:
        st.metric("Average Order Value", f"${aov:.2f}")
    
    with kpi3:
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    
    with kpi4:
        st.metric("New Customers", f"{new_customers:,}")
except Exception as e:
    st.error(f"Error calculating KPIs: {e}")

# Second row of KPIs
kpi5, kpi6, kpi7, kpi8 = st.columns(4)

try:
    # Customer Retention
    returning_customers = filtered_revenue["Returning Customers"].sum() if not filtered_revenue.empty and "Returning Customers" in filtered_revenue.columns else 0
    total_customers = new_customers + returning_customers
    retention_rate = (returning_customers / total_customers) * 100 if total_customers > 0 else 0
    
    # ROAS
    marketing_spend = filtered_revenue["Marketing Spend"].sum() if not filtered_revenue.empty and "Marketing Spend" in filtered_revenue.columns else 0
    roas = total_revenue / marketing_spend if marketing_spend > 0 else 0
    
    # Social Engagement Rate
    avg_engagement_rate = 0
    if not filtered_social.empty and 'Engagement Rate (%)' in filtered_social.columns:
        avg_engagement_rate = filtered_social['Engagement Rate (%)'].mean()
    
    # Email Performance
    avg_open_rate = 0
    if not filtered_email.empty and 'Open Rate %' in filtered_email.columns:
        avg_open_rate = filtered_email['Open Rate %'].mean()
    
    with kpi5:
        st.metric("Customer Retention", f"{retention_rate:.1f}%")
    
    with kpi6:
        st.metric("ROAS", f"{roas:.2f}x")
    
    with kpi7:
        st.metric("Social Engagement", f"{avg_engagement_rate:.2f}%")
    
    with kpi8:
        st.metric("Email Open Rate", f"{avg_open_rate:.2f}%")
except Exception as e:
    st.error(f"Error calculating second row KPIs: {e}")

# Revenue by Channel chart
st.markdown("## Channel Performance")
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns and 'Revenue' in filtered_revenue.columns:
            # Group data by channel
            channel_revenue = filtered_revenue.groupby('Channel')['Revenue'].sum().reset_index()
            
            # Create the chart
            fig = px.bar(
                channel_revenue,
                x='Channel',
                y='Revenue',
                title='Revenue by Channel',
                color='Channel',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No channel revenue data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Revenue by Channel chart: {e}")

# Conversion Rate by Channel
with col2:
    try:
        if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns and 'Orders' in filtered_revenue.columns and 'Web Visitors' in filtered_revenue.columns:
            # Calculate conversion rates
            conv_data = filtered_revenue.groupby('Channel').agg({
                'Orders': 'sum',
                'Web Visitors': 'sum'
            }).reset_index()
            
            conv_data['Conversion Rate'] = (conv_data['Orders'] / conv_data['Web Visitors']) * 100
            
            # Create chart
            fig = px.line(
                conv_data,
                x='Channel',
                y='Conversion Rate',
                title='Conversion Rate by Channel (%)',
                markers=True,
                color_discrete_sequence=['#19A7CE']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No conversion data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Conversion Rate chart: {e}")

# Customer Demographics and Product Categories
st.markdown("## Customer and Product Analysis")
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_customers.empty and 'Age Group' in filtered_customers.columns:
            # Group by age
            age_distribution = filtered_customers.groupby('Age Group').size().reset_index(name='Count')
            
            # Create chart
            fig = px.pie(
                age_distribution,
                values='Count',
                names='Age Group',
                title='Customer Demographics by Age Group',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No customer demographic data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Customer Demographics chart: {e}")

with col2:
    try:
        if not filtered_products.empty and 'Product Category' in filtered_products.columns and 'Revenue' in filtered_products.columns:
            # Group by product category
            product_revenue = filtered_products.groupby('Product Category')['Revenue'].sum().reset_index()
            product_revenue = product_revenue.sort_values('Revenue', ascending=False)
            
            # Create chart
            fig = px.bar(
                product_revenue,
                x='Revenue',
                y='Product Category',
                title='Revenue by Product Category',
                orientation='h',
                color='Product Category',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product category data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Product Category chart: {e}")

# Social Media and Email Performance
st.markdown("## Marketing Channel Metrics")
col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_social.empty and 'Platforms' in filtered_social.columns:
            # Group by platform
            platform_engagement = filtered_social.groupby('Platforms').agg({
                'Engagement': 'sum',
                'Impressions': 'sum'
            }).reset_index()
            
            if 'Engagement' in platform_engagement.columns and 'Impressions' in platform_engagement.columns:
                platform_engagement['Engagement Rate'] = (platform_engagement['Engagement'] / platform_engagement['Impressions']) * 100
                
                # Create chart
                fig = px.bar(
                    platform_engagement,
                    x='Platforms',
                    y='Engagement Rate',
                    title='Social Media Engagement Rate by Platform (%)',
                    color='Platforms',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Incomplete social media data for calculating engagement rates.")
        else:
            st.info("No social media data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Social Media chart: {e}")

with col2:
    try:
        if not filtered_email.empty and 'Email Type' in filtered_email.columns:
            # Check if required columns exist
            required_cols = ['Open Rate %', 'Click Rate %', 'Conversion Rate %']
            available_cols = [col for col in required_cols if col in filtered_email.columns]
            
            if available_cols:
                # Group by email type
                email_performance = filtered_email.groupby('Email Type')[available_cols].mean().reset_index()
                
                # Reshape for chart
                email_perf_long = pd.melt(
                    email_performance,
                    id_vars=['Email Type'],
                    value_vars=available_cols,
                    var_name='Metric',
                    value_name='Rate'
                )
                
                # Create chart
                fig = px.bar(
                    email_perf_long,
                    x='Email Type',
                    y='Rate',
                    color='Metric',
                    barmode='group',
                    title='Email Performance by Type (%)',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Email performance metrics are not available in the data.")
        else:
            st.info("No email performance data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Email Performance chart: {e}")

# Style Type Performance (9-to-5 vs 5-to-9)
st.markdown("## 9-to-5 vs 5-to-9 Performance")

try:
    if not filtered_products.empty and 'Style Type' in filtered_products.columns and 'Collection' in filtered_products.columns and 'Revenue' in filtered_products.columns:
        # Group by style type
        style_revenue = filtered_products.groupby(['Style Type', 'Collection'])['Revenue'].sum().reset_index()
        
        # Create chart
        fig = px.bar(
            style_revenue,
            x='Collection',
            y='Revenue',
            color='Style Type',
            barmode='group',
            title='Revenue by Style Type and Collection',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#FFD166']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No style type data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Style Type Performance chart: {e}")

# Campaign Performance
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
        st.info("No campaign data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Campaign Performance table: {e}")

# PR & Influencer Performance
st.markdown("## PR & Influencer Metrics")

try:
    if not filtered_affiliates.empty and 'Type' in filtered_affiliates.columns:
        # Group by type
        pr_performance = filtered_affiliates.groupby('Type').agg({
            'Estimated Value': 'sum',
            'Cost': 'sum',
            'ROI': 'mean',
            'Engagement': 'sum'
        }).reset_index()
        
        # Calculate ROI if it doesn't exist
        if 'ROI' not in pr_performance.columns:
            pr_performance['ROI'] = pr_performance['Estimated Value'] / pr_performance['Cost'].replace(0, float('nan'))
            
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estimated PR Value", f"${pr_performance['Estimated Value'].sum():,.2f}")
            
        with col2:
            st.metric("Total Engagement", f"{pr_performance['Engagement'].sum():,.0f}")
            
        with col3:
            avg_roi = pr_performance['ROI'].mean()
            st.metric("Average ROI", f"{avg_roi:.2f}x")
        
        # Create chart
        fig = px.bar(
            pr_performance,
            x='Type',
            y='Estimated Value',
            title='PR & Influencer Estimated Value by Type',
            color='Type',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PR & Influencer data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating PR & Influencer section: {e}")

# Footer with data timestamp
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Dashboard last updated: {current_time}")
st.caption("Nine To Five Marketing Dashboard - Created for tracking marketing performance across channels")
