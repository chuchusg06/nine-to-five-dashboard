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
    page_title="Customer Insights - Nine To Five Marketing",
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

# Load customer data
customers_data = load_data("Raw Data - Customers")

# Sidebar filters
st.sidebar.title("Customer Insights")

# Age Group filter
if not customers_data.empty and 'Age Group' in customers_data.columns:
    age_groups = ["All Age Groups"] + sorted(customers_data['Age Group'].unique().tolist())
    age_group_filter = st.sidebar.selectbox("Age Group", age_groups, index=0)
else:
    age_group_filter = "All Age Groups"

# Location filter
if not customers_data.empty and 'Location' in customers_data.columns:
    locations = ["All Locations"] + sorted(customers_data['Location'].unique().tolist())
    location_filter = st.sidebar.selectbox("Location", locations, index=0)
else:
    location_filter = "All Locations"

# Channel Source filter
if not customers_data.empty and 'Channel Source' in customers_data.columns:
    channels = ["All Channels"] + sorted(customers_data['Channel Source'].unique().tolist())
    channel_source_filter = st.sidebar.selectbox("Channel Source", channels, index=0)
else:
    channel_source_filter = "All Channels"

# Function to filter data
def filter_customers(df):
    if df.empty:
        return df
    
    # Apply age group filter
    if age_group_filter != "All Age Groups" and 'Age Group' in df.columns:
        df = df[df['Age Group'] == age_group_filter]
    
    # Apply location filter
    if location_filter != "All Locations" and 'Location' in df.columns:
        df = df[df['Location'] == location_filter]
    
    # Apply channel source filter
    if channel_source_filter != "All Channels" and 'Channel Source' in df.columns:
        df = df[df['Channel Source'] == channel_source_filter]
    
    return df

# Apply filters
filtered_customers = filter_customers(customers_data)

# Page Title
st.title("Customer Insights")
st.markdown("### Analysis of customer demographics and behavior")

# Customer Overview
st.markdown("## Customer Overview")

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

try:
    # Calculate KPIs
    avg_aov = filtered_customers['AOV'].mean() if not filtered_customers.empty and 'AOV' in filtered_customers.columns else 0
    avg_clv = filtered_customers['CLV'].mean() if not filtered_customers.empty and 'CLV' in filtered_customers.columns else 0
    avg_purchase_freq = filtered_customers['Purchase Frequency'].mean() if not filtered_customers.empty and 'Purchase Frequency' in filtered_customers.columns else 0
    avg_repeat_rate = filtered_customers['Repeat Purchase Rate'].mean() if not filtered_customers.empty and 'Repeat Purchase Rate' in filtered_customers.columns else 0
    
    # Display KPIs
    with kpi1:
        st.metric("Average Order Value", f"${avg_aov:.2f}")
    
    with kpi2:
        st.metric("Customer Lifetime Value", f"${avg_clv:.2f}")
    
    with kpi3:
        st.metric("Purchase Frequency", f"{avg_purchase_freq:.2f}")
    
    with kpi4:
        st.metric("Repeat Purchase Rate", f"{avg_repeat_rate:.1%}")
except Exception as e:
    st.error(f"Error calculating KPIs: {e}")

# Customer Demographics
st.markdown("## Customer Demographics")

col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_customers.empty and 'Age Group' in filtered_customers.columns:
            # Count by age group
            age_distribution = filtered_customers.groupby('Age Group').size().reset_index(name='Count')
            age_distribution = age_distribution.sort_values('Count', ascending=False)
            
            # Create age group chart
            fig = px.pie(
                age_distribution,
                values='Count',
                names='Age Group',
                title='Customer Distribution by Age Group',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No age group data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Age Group chart: {e}")

with col2:
    try:
        if not filtered_customers.empty and 'Location' in filtered_customers.columns:
            # Count by location
            location_distribution = filtered_customers.groupby('Location').size().reset_index(name='Count')
            location_distribution = location_distribution.sort_values('Count', ascending=False)
            
            # Create location chart
            fig = px.pie(
                location_distribution,
                values='Count',
                names='Location',
                title='Customer Distribution by Location',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Location chart: {e}")

# Channel Distribution
st.markdown("## Customer Acquisition Channels")

try:
    if not filtered_customers.empty and 'Channel Source' in filtered_customers.columns:
        # Count by channel
        channel_distribution = filtered_customers.groupby('Channel Source').size().reset_index(name='Count')
        channel_distribution = channel_distribution.sort_values('Count', ascending=False)
        
        # Calculate percentage
        total_customers = channel_distribution['Count'].sum()
        channel_distribution['Percentage'] = (channel_distribution['Count'] / total_customers * 100).round(1)
        
        # Create channel chart
        fig = px.bar(
            channel_distribution,
            x='Channel Source',
            y='Count',
            title='Customer Acquisition by Channel',
            color='Channel Source',
            text='Percentage',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No channel source data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Channel Distribution chart: {e}")

# Customer Value Analysis
st.markdown("## Customer Value Analysis")

try:
    if not filtered_customers.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Age Group' in filtered_customers.columns and 'CLV' in filtered_customers.columns:
                # CLV by age group
                clv_by_age = filtered_customers.groupby('Age Group')['CLV'].mean().reset_index()
                clv_by_age = clv_by_age.sort_values('CLV', ascending=False)
                
                # Create CLV by age chart
                fig = px.bar(
                    clv_by_age,
                    x='Age Group',
                    y='CLV',
                    title='Average Customer Lifetime Value by Age Group',
                    color='Age Group',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CLV by age group data available for the selected filters.")
        
        with col2:
            if 'Channel Source' in filtered_customers.columns and 'CLV' in filtered_customers.columns:
                # CLV by channel
                clv_by_channel = filtered_customers.groupby('Channel Source')['CLV'].mean().reset_index()
                clv_by_channel = clv_by_channel.sort_values('CLV', ascending=False)
                
                # Create CLV by channel chart
                fig = px.bar(
                    clv_by_channel,
                    x='Channel Source',
                    y='CLV',
                    title='Average Customer Lifetime Value by Channel',
                    color='Channel Source',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CLV by channel data available for the selected filters.")
    else:
        st.info("No customer value data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Customer Value Analysis section: {e}")

# Purchase Behavior
st.markdown("## Purchase Behavior Analysis")

try:
    if not filtered_customers.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Purchase Frequency' in filtered_customers.columns and 'Channel Source' in filtered_customers.columns:
                # Purchase frequency by channel
                freq_by_channel = filtered_customers.groupby('Channel Source')['Purchase Frequency'].mean().reset_index()
                freq_by_channel = freq_by_channel.sort_values('Purchase Frequency', ascending=False)
                
                # Create purchase frequency chart
                fig = px.bar(
                    freq_by_channel,
                    x='Channel Source',
                    y='Purchase Frequency',
                    title='Average Purchase Frequency by Channel',
                    color='Channel Source',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No purchase frequency data available for the selected filters.")
        
        with col2:
            if 'First Purchase Category' in filtered_customers.columns:
                # First purchase by category
                first_purchase = filtered_customers.groupby('First Purchase Category').size().reset_index(name='Count')
                first_purchase = first_purchase.sort_values('Count', ascending=False)
                
                # Create first purchase chart
                fig = px.pie(
                    first_purchase,
                    values='Count',
                    names='First Purchase Category',
                    title='First Purchase by Category',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No first purchase category data available for the selected filters.")
    else:
        st.info("No purchase behavior data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Purchase Behavior section: {e}")

# Customer Metrics Table
st.markdown("## Customer Metrics by Segment")

try:
    if not filtered_customers.empty and 'Age Group' in filtered_customers.columns:
        # Aggregate metrics by age group
        age_metrics = filtered_customers.groupby('Age Group').agg({
            'AOV': 'mean',
            'CLV': 'mean',
            'Purchase Frequency': 'mean',
            'Repeat Purchase Rate': 'mean'
        }).reset_index()
        
        # Format for display
        display_metrics = age_metrics.copy()
        display_metrics['AOV'] = display_metrics['AOV'].apply(lambda x: f"${x:.2f}")
        display_metrics['CLV'] = display_metrics['CLV'].apply(lambda x: f"${x:.2f}")
        display_metrics['Purchase Frequency'] = display_metrics['Purchase Frequency'].apply(lambda x: f"{x:.2f}")
        display_metrics['Repeat Purchase Rate'] = display_metrics['Repeat Purchase Rate'].apply(lambda x: f"{x:.1%}")
        
        # Display table
        st.dataframe(
            display_metrics,
            column_config={
                "Age Group": st.column_config.TextColumn("Age Group"),
                "AOV": st.column_config.TextColumn("Avg. Order Value"),
                "CLV": st.column_config.TextColumn("Customer Lifetime Value"),
                "Purchase Frequency": st.column_config.TextColumn("Purchase Frequency"),
                "Repeat Purchase Rate": st.column_config.TextColumn("Repeat Purchase Rate")
            },
            use_container_width=True
        )
    else:
        st.info("No customer metrics data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Customer Metrics table: {e}")

# Footer
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")
