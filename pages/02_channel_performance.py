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
    page_title="Channel Performance - Nine To Five Marketing",
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
st.sidebar.title("Channel Performance")

# Time period filter
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
    index=4  # Default to All Time
)

# Channel selection
channel_selection = st.sidebar.radio(
    "Select Channel",
    ["All Channels", "Social Paid", "Social Organic", "Email", "Affiliates"]
)

# Load data
revenue_data = load_data("Raw Data - Revenue")
social_data = load_data("Raw Data - Social Media")
email_data = load_data("Raw Data - Email")
affiliates_data = load_data("Raw Data - Affiliates")
campaigns_data = load_data("Raw Data - Campaigns")

# Filter by selected channel
def filter_by_channel(df, channel_col='Channel'):
    if channel_selection == "All Channels" or channel_col not in df.columns or df.empty:
        return df
    else:
        return df[df[channel_col] == channel_selection]

# Apply channel filter
filtered_revenue = filter_by_channel(revenue_data)
filtered_social = filter_by_channel(social_data, 'Platforms') if channel_selection in ["Social Paid", "Social Organic"] else social_data
filtered_email = email_data if channel_selection == "Email" else email_data
filtered_affiliates = affiliates_data if channel_selection == "Affiliates" else affiliates_data
filtered_campaigns = filter_by_channel(campaigns_data)

# Page Title
st.title("Channel Performance Analysis")
st.markdown(f"### Performance metrics for {channel_selection}")

# Performance Overview
st.markdown("## Performance Overview")

# KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

try:
    # Calculate KPIs based on selected channel
    total_revenue = filtered_revenue["Revenue"].sum() if not filtered_revenue.empty and "Revenue" in filtered_revenue.columns else 0
    total_orders = filtered_revenue["Orders"].sum() if not filtered_revenue.empty and "Orders" in filtered_revenue.columns else 0
    conversion_rate = (total_orders / filtered_revenue["Web Visitors"].sum() * 100) if not filtered_revenue.empty and "Web Visitors" in filtered_revenue.columns and filtered_revenue["Web Visitors"].sum() > 0 else 0
    roas = total_revenue / filtered_revenue["Marketing Spend"].sum() if not filtered_revenue.empty and "Marketing Spend" in filtered_revenue.columns and filtered_revenue["Marketing Spend"].sum() > 0 else 0
    
    with kpi1:
        st.metric("Revenue", f"${total_revenue:,.2f}")
    
    with kpi2:
        st.metric("Orders", f"{total_orders:,}")
    
    with kpi3:
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    
    with kpi4:
        st.metric("ROAS", f"{roas:.2f}x")
except Exception as e:
    st.error(f"Error calculating KPIs: {e}")

# Performance Trend
st.markdown("## Performance Trend")

try:
    if not filtered_revenue.empty and 'Date' in filtered_revenue.columns:
        # Create a figure with multiple metrics
        fig = go.Figure()
        
        if 'Revenue' in filtered_revenue.columns:
            daily_revenue = filtered_revenue.groupby('Date')['Revenue'].sum().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue['Date'],
                    y=daily_revenue['Revenue'],
                    name='Revenue',
                    mode='lines+markers',
                    line=dict(color='blue', width=2)
                )
            )
        
        if 'Orders' in filtered_revenue.columns:
            daily_orders = filtered_revenue.groupby('Date')['Orders'].sum().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=daily_orders['Date'],
                    y=daily_orders['Orders'],
                    name='Orders',
                    mode='lines+markers',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                )
            )
        
        # Update layout for dual y-axis
        fig.update_layout(
            title=f'{channel_selection} Performance Trend',
            yaxis=dict(title='Revenue ($)'),
            yaxis2=dict(
                title='Orders',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0.1, y=1.15, orientation='h')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No performance trend data available for the selected channel.")
except Exception as e:
    st.error(f"Error creating Performance Trend chart: {e}")

# Channel-specific metrics
if channel_selection in ["Social Paid", "Social Organic"]:
    st.markdown("## Social Media Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if not filtered_social.empty and 'Platforms' in filtered_social.columns:
                # Group by platform
                platform_metrics = filtered_social.groupby('Platforms').agg({
                    'Followers': 'mean',
                    'Posts': 'sum',
                    'Impressions': 'sum',
                    'Engagement': 'sum'
                }).reset_index()
                
                # Calculate engagement rate
                platform_metrics['Engagement Rate (%)'] = (platform_metrics['Engagement'] / platform_metrics['Impressions'] * 100).round(2)
                
                # Create bar chart for engagement rate
                fig = px.bar(
                    platform_metrics,
                    x='Platforms',
                    y='Engagement Rate (%)',
                    title='Engagement Rate by Platform',
                    color='Platforms',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No social media data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating Social Media Engagement chart: {e}")
    
    with col2:
        try:
            if not filtered_social.empty and 'Date' in filtered_social.columns:
                # Group by date
                daily_followers = filtered_social.groupby('Date')['New Followers'].sum().reset_index()
                
                # Create line chart for follower growth
                fig = px.line(
                    daily_followers,
                    x='Date',
                    y='New Followers',
                    title='Daily New Followers',
                    markers=True,
                    color_discrete_sequence=['#FF6B6B']
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No follower data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating Follower Growth chart: {e}")
    
    # Social post performance
    st.markdown("### Social Post Metrics")
    
    try:
        if not filtered_social.empty:
            # Create metrics for social performance
            total_posts = filtered_social['Posts'].sum() if 'Posts' in filtered_social.columns else 0
            total_impressions = filtered_social['Impressions'].sum() if 'Impressions' in filtered_social.columns else 0
            total_engagement = filtered_social['Engagement'].sum() if 'Engagement' in filtered_social.columns else 0
            
            # Calculate averages
            avg_impressions_per_post = total_impressions / total_posts if total_posts > 0 else 0
            avg_engagement_per_post = total_engagement / total_posts if total_posts > 0 else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Posts", f"{total_posts}")
            
            with col2:
                st.metric("Avg. Impressions per Post", f"{avg_impressions_per_post:,.0f}")
            
            with col3:
                st.metric("Avg. Engagement per Post", f"{avg_engagement_per_post:,.0f}")
        else:
            st.info("No social post data available for the selected filters.")
    except Exception as e:
        st.error(f"Error calculating Social Post Metrics: {e}")

elif channel_selection == "Email":
    st.markdown("## Email Marketing Metrics")
    
    try:
        if not filtered_email.empty:
            # Create a summary table by email type
            email_summary = filtered_email.groupby('Email Type').agg({
                'Email Sent': 'sum',
                'Open Rate %': 'mean',
                'Click Rate %': 'mean',
                'Conversion Rate %': 'mean',
                'Revenue': 'sum'
            }).reset_index()
            
            # Format for display
            display_email = email_summary.copy()
            display_email['Open Rate %'] = display_email['Open Rate %'].round(2)
            display_email['Click Rate %'] = display_email['Click Rate %'].round(2)
            display_email['Conversion Rate %'] = display_email['Conversion Rate %'].round(2)
            display_email['Revenue'] = display_email['Revenue'].apply(lambda x: f"${x:,.2f}")
            display_email['Revenue Per Email'] = (email_summary['Revenue'] / email_summary['Email Sent']).apply(lambda x: f"${x:.2f}")
            
            # Display as table
            st.dataframe(
                display_email,
                column_config={
                    "Email Type": st.column_config.TextColumn("Email Type"),
                    "Email Sent": st.column_config.NumberColumn("Emails Sent", format="%d"),
                    "Open Rate %": st.column_config.NumberColumn("Open Rate", format="%.2f%%"),
                    "Click Rate %": st.column_config.NumberColumn("Click Rate", format="%.2f%%"),
                    "Conversion Rate %": st.column_config.NumberColumn("Conv. Rate", format="%.2f%%"),
                    "Revenue": st.column_config.TextColumn("Revenue"),
                    "Revenue Per Email": st.column_config.TextColumn("Rev. Per Email")
                },
                use_container_width=True
            )
            
            # Create comparison chart for email metrics
            fig = px.bar(
                email_summary,
                x='Email Type',
                y=['Open Rate %', 'Click Rate %', 'Conversion Rate %'],
                title='Email Performance Metrics by Type',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No email marketing data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Email Marketing Metrics: {e}")

elif channel_selection == "Affiliates":
    st.markdown("## Affiliate & Influencer Metrics")
    
    try:
        if not filtered_affiliates.empty and 'Type' in filtered_affiliates.columns:
            # Group by type
            affiliate_metrics = filtered_affiliates.groupby('Type').agg({
                'Estimated Value': 'sum',
                'Cost': 'sum',
                'Engagement': 'sum',
                'Conversions': 'sum'
            }).reset_index()
            
            # Calculate ROI
            affiliate_metrics['ROI'] = (affiliate_metrics['Estimated Value'] / affiliate_metrics['Cost']).round(2)
            
            # Create ROI chart
            fig = px.bar(
                affiliate_metrics,
                x='Type',
                y='ROI',
                title='ROI by Affiliate Type',
                color='Type',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            total_value = affiliate_metrics['Estimated Value'].sum()
            total_cost = affiliate_metrics['Cost'].sum()
            total_engagement = affiliate_metrics['Engagement'].sum()
            total_conversions = affiliate_metrics['Conversions'].sum()
            
            with col1:
                st.metric("Estimated Value", f"${total_value:,.2f}")
            
            with col2:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            
            with col3:
                st.metric("ROI", f"{(total_value / total_cost if total_cost > 0 else 0):.2f}x")
            
            with col4:
                st.metric("Conversions", f"{total_conversions:,}")
            
            # Display detailed metrics table
            st.markdown("### Affiliate Performance Details")
            
            # Format for display
            display_metrics = affiliate_metrics.copy()
            display_metrics['Estimated Value'] = display_metrics['Estimated Value'].apply(lambda x: f"${x:,.2f}")
            display_metrics['Cost'] = display_metrics['Cost'].apply(lambda x: f"${x:,.2f}")
            display_metrics['ROI'] = display_metrics['ROI'].apply(lambda x: f"{x:.2f}x")
            display_metrics['Cost Per Conversion'] = (affiliate_metrics['Cost'] / affiliate_metrics['Conversions']).apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "$0.00")
            
            st.dataframe(
                display_metrics,
                column_config={
                    "Type": st.column_config.TextColumn("Affiliate Type"),
                    "Estimated Value": st.column_config.TextColumn("Est. Value"),
                    "Cost": st.column_config.TextColumn("Cost"),
                    "Engagement": st.column_config.NumberColumn("Engagement", format="%d"),
                    "Conversions": st.column_config.NumberColumn("Conversions", format="%d"),
                    "ROI": st.column_config.TextColumn("ROI"),
                    "Cost Per Conversion": st.column_config.TextColumn("Cost/Conv.")
                },
                use_container_width=True
            )
        else:
            st.info("No affiliate data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Affiliate Metrics: {e}")

else:  # All Channels comparison
    st.markdown("## Channel Comparison")
    
    try:
        if not revenue_data.empty and 'Channel' in revenue_data.columns:
            # Group by channel
            channel_comparison = revenue_data.groupby('Channel').agg({
                'Revenue': 'sum',
                'Orders': 'sum',
                'Web Visitors': 'sum',
                'Marketing Spend': 'sum'
            }).reset_index()
            
          # Calculate metrics
            channel_comparison['Conversion Rate (%)'] = (channel_comparison['Orders'] / channel_comparison['Web Visitors'] * 100).round(2)
            channel_comparison['ROAS'] = (channel_comparison['Revenue'] / channel_comparison['Marketing Spend']).round(2)
            channel_comparison['CPC'] = (channel_comparison['Marketing Spend'] / channel_comparison['Web Visitors']).round(2)
            
            # Create comparison chart
            fig = px.bar(
                channel_comparison,
                x='Channel',
                y='Revenue',
                title='Revenue by Channel',
                color='Channel',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Conversion Rate comparison
                fig = px.bar(
                    channel_comparison,
                    x='Channel',
                    y='Conversion Rate (%)',
                    title='Conversion Rate by Channel',
                    color='Channel',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROAS comparison
                fig = px.bar(
                    channel_comparison,
                    x='Channel',
                    y='ROAS',
                    title='ROAS by Channel',
                    color='Channel',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display comparison table
            st.markdown("### Channel Performance Metrics")
            
            # Format for display
            display_comparison = channel_comparison.copy()
            display_comparison['Revenue'] = display_comparison['Revenue'].apply(lambda x: f"${x:,.2f}")
            display_comparison['Marketing Spend'] = display_comparison['Marketing Spend'].apply(lambda x: f"${x:,.2f}")
            display_comparison['CPC'] = display_comparison['CPC'].apply(lambda x: f"${x:.2f}")
            display_comparison['ROAS'] = display_comparison['ROAS'].apply(lambda x: f"{x:.2f}x")
            
            st.dataframe(
                display_comparison,
                column_config={
                    "Channel": st.column_config.TextColumn("Channel"),
                    "Revenue": st.column_config.TextColumn("Revenue"),
                    "Orders": st.column_config.NumberColumn("Orders", format="%d"),
                    "Web Visitors": st.column_config.NumberColumn("Visitors", format="%d"),
                    "Marketing Spend": st.column_config.TextColumn("Ad Spend"),
                    "Conversion Rate (%)": st.column_config.NumberColumn("Conv. Rate", format="%.2f%%"),
                    "ROAS": st.column_config.TextColumn("ROAS"),
                    "CPC": st.column_config.TextColumn("Cost Per Click")
                },
                use_container_width=True
            )
        else:
            st.info("No channel comparison data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating Channel Comparison: {e}")

# Campaign Performance
st.markdown("## Campaign Performance")

try:
    if not filtered_campaigns.empty and 'Campaign Name' in filtered_campaigns.columns:
        # Select relevant columns
        campaign_data = filtered_campaigns[['Campaign Name', 'Channel', 'Campaign Type', 'Revenue', 'Spend', 'ROAS']].copy()
        
        # Sort by revenue
        campaign_data = campaign_data.sort_values('Revenue', ascending=False)
        
        # Convert to proper formats for display
        display_campaign = campaign_data.copy()
        display_campaign['Revenue'] = display_campaign['Revenue'].apply(lambda x: f"${x:,.2f}")
        display_campaign['Spend'] = display_campaign['Spend'].apply(lambda x: f"${x:,.2f}")
        display_campaign['ROAS'] = display_campaign['ROAS'].apply(lambda x: f"{x:.2f}x")
        
        # Display as table
        st.dataframe(
            display_campaign,
            column_config={
                "Campaign Name": st.column_config.TextColumn("Campaign"),
                "Channel": st.column_config.TextColumn("Channel"),
                "Campaign Type": st.column_config.TextColumn("Type"),
                "Revenue": st.column_config.TextColumn("Revenue"),
                "Spend": st.column_config.TextColumn("Spend"),
                "ROAS": st.column_config.TextColumn("ROAS")
            },
            use_container_width=True
        )
        
        # Campaign type performance
        campaign_type_perf = filtered_campaigns.groupby('Campaign Type').agg({
            'Revenue': 'sum',
            'Spend': 'sum'
        }).reset_index()
        
        campaign_type_perf['ROAS'] = (campaign_type_perf['Revenue'] / campaign_type_perf['Spend']).round(2)
        
        # Create chart
        fig = px.bar(
            campaign_type_perf,
            x='Campaign Type',
            y='ROAS',
            title='ROAS by Campaign Type',
            color='Campaign Type',
            color_discrete_sequence=px.colors.qualitative.Dark2
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No campaign performance data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating Campaign Performance section: {e}")

# Footer
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")
