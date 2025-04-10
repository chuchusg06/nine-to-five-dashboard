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
    page_title="Campaign Manager - Nine To Five Marketing",
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

# Load campaigns data
campaigns_data = load_data("Raw Data - Campaigns")

# Sidebar filters
st.sidebar.title("Campaign Manager")

# Campaign Type filter
if not campaigns_data.empty and 'Campaign Type' in campaigns_data.columns:
    campaign_types = ["All Types"] + sorted(campaigns_data['Campaign Type'].unique().tolist())
    campaign_type_filter = st.sidebar.selectbox("Campaign Type", campaign_types, index=0)
else:
    campaign_type_filter = "All Types"

# Channel filter
if not campaigns_data.empty and 'Channel' in campaigns_data.columns:
    channels = ["All Channels"] + sorted(campaigns_data['Channel'].unique().tolist())
    channel_filter = st.sidebar.selectbox("Channel", channels, index=0)
else:
    channel_filter = "All Channels"

# Target Audience filter
if not campaigns_data.empty and 'Target Audience' in campaigns_data.columns:
    audiences = ["All Audiences"] + sorted(campaigns_data['Target Audience'].unique().tolist())
    audience_filter = st.sidebar.selectbox("Target Audience", audiences, index=0)
else:
    audience_filter = "All Audiences"

# ROAS filter with slider
if not campaigns_data.empty and 'ROAS' in campaigns_data.columns:
    min_roas = float(campaigns_data['ROAS'].min())
    max_roas = float(campaigns_data['ROAS'].max())
    roas_range = st.sidebar.slider("ROAS Range", min_roas, max_roas, (min_roas, max_roas))
else:
    roas_range = (0, 100)

# Function to filter data
def filter_campaigns(df):
    if df.empty:
        return df
    
    # Apply campaign type filter
    if campaign_type_filter != "All Types" and 'Campaign Type' in df.columns:
        df = df[df['Campaign Type'] == campaign_type_filter]
    
    # Apply channel filter
    if channel_filter != "All Channels" and 'Channel' in df.columns:
        df = df[df['Channel'] == channel_filter]
    
    # Apply audience filter
    if audience_filter != "All Audiences" and 'Target Audience' in df.columns:
        df = df[df['Target Audience'] == audience_filter]
    
    # Apply ROAS filter
    if 'ROAS' in df.columns:
        df = df[(df['ROAS'] >= roas_range[0]) & (df['ROAS'] <= roas_range[1])]
    
    return df

# Apply filters
filtered_campaigns = filter_campaigns(campaigns_data)

# Page Title
st.title("Campaign Manager")
st.markdown("### Performance tracking and optimization for marketing campaigns")

# Campaign Performance Overview
st.markdown("## Campaign Performance Overview")

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

try:
    # Calculate KPIs
    total_spend = filtered_campaigns['Spend'].sum() if not filtered_campaigns.empty and 'Spend' in filtered_campaigns.columns else 0
    total_revenue = filtered_campaigns['Revenue'].sum() if not filtered_campaigns.empty and 'Revenue' in filtered_campaigns.columns else 0
    overall_roas = total_revenue / total_spend if total_spend > 0 else 0
    
    total_impressions = filtered_campaigns['Impressions'].sum() if not filtered_campaigns.empty and 'Impressions' in filtered_campaigns.columns else 0
    total_clicks = filtered_campaigns['Clicks'].sum() if not filtered_campaigns.empty and 'Clicks' in filtered_campaigns.columns else 0
    overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    
    total_conversions = filtered_campaigns['Conversions'].sum() if not filtered_campaigns.empty and 'Conversions' in filtered_campaigns.columns else 0
    overall_cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
    
    avg_cpa = total_spend / total_conversions if total_conversions > 0 else 0
    
    # Display KPIs
    with kpi1:
        st.metric("Total Spend", f"${total_spend:,.2f}")
    
    with kpi2:
        st.metric("Overall ROAS", f"{overall_roas:.2f}x")
    
    with kpi3:
        st.metric("Click-Through Rate", f"{overall_ctr:.2f}%")
    
    with kpi4:
        st.metric("Conversion Rate", f"{overall_cvr:.2f}%")
except Exception as e:
    st.error(f"Error calculating KPIs: {e}")

# Campaign Performance Comparison
st.markdown("## Campaign Performance Comparison")

try:
    if not filtered_campaigns.empty and 'Campaign Name' in filtered_campaigns.columns:
        # Get campaign metrics
        campaign_metrics = filtered_campaigns[['Campaign Name', 'Channel', 'Campaign Type', 'Spend', 'Revenue', 'ROAS', 'Conversions', 'CPA']].copy()
        
        # Sort by ROAS
        campaign_metrics = campaign_metrics.sort_values('ROAS', ascending=False)
        
        # Format for display
        display_campaigns = campaign_metrics.copy()
        display_campaigns['Spend'] = display_campaigns['Spend'].apply(lambda x: f"${x:,.2f}")
        display_campaigns['Revenue'] = display_campaigns['Revenue'].apply(lambda x: f"${x:,.2f}")
        display_campaigns['ROAS'] = display_campaigns['ROAS'].apply(lambda x: f"{x:.2f}x")
        display_campaigns['CPA'] = display_campaigns['CPA'].apply(lambda x: f"${x:.2f}")
        
        # Display table
        st.dataframe(
            display_campaigns,
            column_config={
                "Campaign Name": st.column_config.TextColumn("Campaign"),
                "Channel": st.column_config.TextColumn("Channel"),
                "Campaign Type": st.column_config.TextColumn("Type"),
                "Spend": st.column_config.TextColumn("Spend"),
                "Revenue": st.column_config.TextColumn("Revenue"),
                "ROAS": st.column_config.TextColumn("ROAS"),
                "Conversions": st.column_config.NumberColumn("Conversions", format="%d"),
                "CPA": st.column_config.TextColumn("Cost Per Acquisition")
            },
            use_container_width=True
        )
        
        # Create ROAS comparison chart
        fig = px.bar(
            campaign_metrics,
            x='Campaign Name',
            y='ROAS',
            title='ROAS by Campaign',
            color='Channel',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No campaign data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Campaign Performance Comparison: {e}")

# Campaign Type Analysis
st.markdown("## Campaign Type Analysis")

col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_campaigns.empty and 'Campaign Type' in filtered_campaigns.columns and 'Spend' in filtered_campaigns.columns:
            # Spend by campaign type
            spend_by_type = filtered_campaigns.groupby('Campaign Type')['Spend'].sum().reset_index()
            spend_by_type = spend_by_type.sort_values('Spend', ascending=False)
            
            # Create spend by type chart
            fig = px.pie(
                spend_by_type,
                values='Spend',
                names='Campaign Type',
                title='Spend Distribution by Campaign Type',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No campaign type data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Campaign Type Spend chart: {e}")

with col2:
    try:
        if not filtered_campaigns.empty and 'Campaign Type' in filtered_campaigns.columns and 'ROAS' in filtered_campaigns.columns:
            # ROAS by campaign type
            roas_by_type = filtered_campaigns.groupby('Campaign Type')['ROAS'].mean().reset_index()
            roas_by_type = roas_by_type.sort_values('ROAS', ascending=False)
            
            # Create ROAS by type chart
            fig = px.bar(
                roas_by_type,
                x='Campaign Type',
                y='ROAS',
                title='Average ROAS by Campaign Type',
                color='Campaign Type',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No campaign type ROAS data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Campaign Type ROAS chart: {e}")

# Channel Performance
st.markdown("## Channel Performance")

col1, col2 = st.columns(2)

with col1:
    try:
        if not filtered_campaigns.empty and 'Channel' in filtered_campaigns.columns and 'Spend' in filtered_campaigns.columns and 'Revenue' in filtered_campaigns.columns:
            # Aggregate by channel
            channel_performance = filtered_campaigns.groupby('Channel').agg({
                'Spend': 'sum',
                'Revenue': 'sum',
                'Impressions': 'sum',
                'Clicks': 'sum',
                'Conversions': 'sum'
            }).reset_index()
            
            # Calculate metrics
            channel_performance['ROAS'] = channel_performance['Revenue'] / channel_performance['Spend']
            channel_performance['CTR'] = channel_performance['Clicks'] / channel_performance['Impressions'] * 100
            channel_performance['CVR'] = channel_performance['Conversions'] / channel_performance['Clicks'] * 100
            
            # Create ROAS by channel chart
            fig = px.bar(
                channel_performance.sort_values('ROAS', ascending=False),
                x='Channel',
                y='ROAS',
                title='ROAS by Channel',
                color='Channel',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No channel performance data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Channel ROAS chart: {e}")

with col2:
    try:
        if not filtered_campaigns.empty and 'Channel' in filtered_campaigns.columns and 'CTR %' in filtered_campaigns.columns:
            # CTR by channel
            ctr_by_channel = filtered_campaigns.groupby('Channel')['CTR %'].mean().reset_index()
            ctr_by_channel = ctr_by_channel.sort_values('CTR %', ascending=False)
            
            # Create CTR by channel chart
            fig = px.bar(
                ctr_by_channel,
                x='Channel',
                y='CTR %',
                title='Average CTR by Channel (%)',
                color='Channel',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif 'CTR' in locals() and 'channel_performance' in locals():
            # Use calculated CTR if original CTR % not available
            fig = px.bar(
                channel_performance.sort_values('CTR', ascending=False),
                x='Channel',
                y='CTR',
                title='CTR by Channel (%)',
                color='Channel',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CTR data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Channel CTR chart: {e}")

# Target Audience Analysis
st.markdown("## Target Audience Analysis")

try:
    if not filtered_campaigns.empty and 'Target Audience' in filtered_campaigns.columns and 'ROAS' in filtered_campaigns.columns:
        # Metrics by audience
        audience_metrics = filtered_campaigns.groupby('Target Audience').agg({
            'ROAS': 'mean',
            'CTR %': 'mean',
            'CR %': 'mean',
            'CPA': 'mean'
        }).reset_index()
        
        # Create a comparison chart
        fig = go.Figure()
        
        # Add ROAS bars
        fig.add_trace(
            go.Bar(
                x=audience_metrics['Target Audience'],
                y=audience_metrics['ROAS'],
                name='ROAS',
                marker_color='#FF6B6B'
            )
        )
        
        # Add CTR line
        fig.add_trace(
            go.Scatter(
                x=audience_metrics['Target Audience'],
                y=audience_metrics['CTR %'],
                name='CTR %',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#4ECDC4', width=2)
            )
        )
        
        # Update layout for dual y-axis
        fig.update_layout(
            title='Performance Metrics by Target Audience',
            yaxis=dict(title='ROAS'),
            yaxis2=dict(
                title='CTR %',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0.1, y=1.15, orientation='h')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics table
        display_audience = audience_metrics.copy()
        display_audience['ROAS'] = display_audience['ROAS'].apply(lambda x: f"{x:.2f}x")
        display_audience['CTR %'] = display_audience['CTR %'].apply(lambda x: f"{x:.2f}%")
        display_audience['CR %'] = display_audience['CR %'].apply(lambda x: f"{x:.2f}%")
        display_audience['CPA'] = display_audience['CPA'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            display_audience,
            column_config={
                "Target Audience": st.column_config.TextColumn("Audience"),
                "ROAS": st.column_config.TextColumn("ROAS"),
                "CTR %": st.column_config.TextColumn("CTR"),
                "CR %": st.column_config.TextColumn("Conversion Rate"),
                "CPA": st.column_config.TextColumn("Cost Per Acquisition")
            },
            use_container_width=True
        )
    else:
        st.info("No target audience data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Target Audience Analysis: {e}")

# Campaign Recommendations
st.markdown("## Campaign Optimization Recommendations")

try:
    if not filtered_campaigns.empty:
        # Identify high and low performing campaigns
        if 'ROAS' in filtered_campaigns.columns:
            avg_roas = filtered_campaigns['ROAS'].mean()
            high_roas = filtered_campaigns[filtered_campaigns['ROAS'] > avg_roas * 1.2].sort_values('ROAS', ascending=False)
            low_roas = filtered_campaigns[filtered_campaigns['ROAS'] < avg_roas * 0.8].sort_values('ROAS')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Top Performing Campaigns")
                if not high_roas.empty:
                    for idx, row in high_roas.head(3).iterrows():
                        with st.container():
                            st.markdown(f"**{row['Campaign Name']}** ({row['Channel']})")
                            st.markdown(f"ROAS: **{row['ROAS']:.2f}x** (vs avg {avg_roas:.2f}x)")
                            st.markdown(f"Recommendation: *Consider increasing budget allocation.*")
                            st.markdown("---")
                else:
                    st.info("No top performing campaigns identified.")
            
            with col2:
                st.markdown("### Underperforming Campaigns")
                if not low_roas.empty:
                    for idx, row in low_roas.head(3).iterrows():
                        with st.container():
                            st.markdown(f"**{row['Campaign Name']}** ({row['Channel']})")
                            st.markdown(f"ROAS: **{row['ROAS']:.2f}x** (vs avg {avg_roas:.2f}x)")
                            st.markdown(f"Recommendation: *Review targeting or creative assets.*")
                            st.markdown("---")
                else:
                    st.info("No underperforming campaigns identified.")
        else:
            st.info("No ROAS data available for generating recommendations.")
    else:
        st.info("No campaign data available for generating recommendations.")
except Exception as e:
    st.error(f"Error creating Campaign Recommendations: {e}")

# Footer
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")
