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
        # Clean the dataframe (remove empty rows)
        df = df.dropna(how='all')
        # Convert date column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
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

# Rest of your code remains unchanged...
