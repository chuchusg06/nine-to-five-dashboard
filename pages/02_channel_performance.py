# Save this as pages/02_channel_performance.py
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
    page_title="Channel Performance - Nine To Five",
    page_icon="📊",
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

# Sidebar filters
st.sidebar.title("Analysis Controls")

# Allow selection of specific channel for detailed analysis
selected_channel = st.sidebar.selectbox(
    "Select Channel",
    ["All Channels", "Social Organic", "Social Paid", "Affiliates", "Email"],
    index=0
)

# Time period filter
time_period = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
    index=1
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

# Function to filter data based on selected channel
def filter_by_channel(df):
    if selected_channel == "All Channels" or 'Channel' not in df.columns or df.empty:
        return df
    else:
        return df[df['Channel'] == selected_channel]

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
    campaigns_data = load_data("Raw Data - Campaigns")
    social_data = load_data("Raw Data - Social Media")
    affiliates_data = load_data("Raw Data - Affiliates")
    email_data = load_data("Raw Data - Email")

# Apply filters
filtered_revenue = apply_filters(revenue_data)
filtered_campaigns = apply_filters(campaigns_data)
filtered_social = filter_by_time(social_data)  # Only time filter for social
filtered_affiliates = filter_by_time(affiliates_data)  # Only time filter for affiliates
filtered_email = filter_by_time(email_data)  # Only time filter for email

# Page title based on selected channel
if selected_channel == "All Channels":
    st.title("Channel Performance Analysis")
    st.markdown("Comparative analysis of all marketing channels")
else:
    st.title(f"{selected_channel} Performance Analysis")
    st.markdown(f"Detailed analysis of {selected_channel} channel metrics and performance")

# Channel Comparison (show only for All Channels)
if selected_channel == "All Channels":
    st.markdown("## Channel Comparison")
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        try:
            if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns:
                # Calculate key metrics by channel
                channel_metrics = filtered_revenue.groupby('Channel').agg({
                    'Revenue': 'sum',
                    'Orders': 'sum',
                    'New Customers': 'sum',
                    'Marketing Spend': 'sum'
                }).reset_index()
                
                # Calculate derived metrics
                channel_metrics['ROAS'] = channel_metrics['Revenue'] / channel_metrics['Marketing Spend'].replace(0, float('nan'))
                channel_metrics['CPA'] = channel_metrics['Marketing Spend'] / channel_metrics['Orders'].replace(0, float('nan'))
                channel_metrics['CAC'] = channel_metrics['Marketing Spend'] / channel_metrics['New Customers'].replace(0, float('nan'))
                
                # Format for display
                display_metrics = channel_metrics.copy()
                display_metrics['Revenue'] = display_metrics['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_metrics['ROAS'] = display_metrics['ROAS'].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "N/A")
                display_metrics['CPA'] = display_metrics['CPA'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
                display_metrics['CAC'] = display_metrics['CAC'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
                
                # Display as table
                st.dataframe(
                    display_metrics[['Channel', 'Revenue', 'Orders', 'New Customers', 'ROAS', 'CPA', 'CAC']],
                    column_config={
                        "Channel": st.column_config.TextColumn("Channel"),
                        "Revenue": st.column_config.TextColumn("Revenue"),
                        "Orders": st.column_config.NumberColumn("Orders"),
                        "New Customers": st.column_config.NumberColumn("New Customers"),
                        "ROAS": st.column_config.TextColumn("ROAS"),
                        "CPA": st.column_config.TextColumn("Cost per Acquisition"),
                        "CAC": st.column_config.TextColumn("Customer Acq. Cost"),
                    },
                    use_container_width=True
                )
            else:
                st.info("No channel comparison data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating channel comparison table: {e}")
    
    with row1_col2:
        try:
            if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns and 'Revenue' in filtered_revenue.columns:
                # Calculate revenue share
                revenue_share = filtered_revenue.groupby('Channel')['Revenue'].sum().reset_index()
                
                # Create pie chart
                fig = px.pie(
                    revenue_share,
                    values='Revenue',
                    names='Channel',
                    title='Revenue Share by Channel',
                    color='Channel',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No revenue share data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating revenue share chart: {e}")
    
    # Channel efficiency comparison
    st.markdown("### Channel Efficiency")
    try:
        if not filtered_revenue.empty and 'Channel' in filtered_revenue.columns:
            # Prepare metrics for radar chart (normalize all to 0-1 scale)
            channels = filtered_revenue['Channel'].unique()
            
            metrics_by_channel = {}
            for channel in channels:
                channel_data = filtered_revenue[filtered_revenue['Channel'] == channel]
                
                # Calculate key efficiency metrics
                revenue = channel_data['Revenue'].sum()
                spend = channel_data['Marketing Spend'].sum()
                new_customers = channel_data['New Customers'].sum()
                orders = channel_data['Orders'].sum()
                visitors = channel_data['Web Visitors'].sum()
                
                roas = revenue / spend if spend > 0 else float('nan')
                conv_rate = (orders / visitors) * 100 if visitors > 0 else 0
                cac = spend / new_customers if new_customers > 0 else float('nan')
                
                metrics_by_channel[channel] = {
                    'ROAS': roas,
                    'Conversion Rate': conv_rate,
                    'Revenue': revenue,
                }
            
            # Create dataframe for radar chart
            radar_data = []
            for channel, metrics in metrics_by_channel.items():
                for metric, value in metrics.items():
                    radar_data.append({
                        'Channel': channel,
                        'Metric': metric,
                        'Value': value
                    })
            
            radar_df = pd.DataFrame(radar_data)
            
            # Normalize values
            for metric in radar_df['Metric'].unique():
                metric_min = radar_df[radar_df['Metric'] == metric]['Value'].min()
                metric_max = radar_df[radar_df['Metric'] == metric]['Value'].max()
                
                if metric_max > metric_min:
                    radar_df.loc[radar_df['Metric'] == metric, 'Normalized'] = (
                        (radar_df.loc[radar_df['Metric'] == metric, 'Value'] - metric_min) / 
                        (metric_max - metric_min)
                    )
                else:
                    radar_df.loc[radar_df['Metric'] == metric, 'Normalized'] = 0.5
            
            # Create comparative radar chart for channels
            radar_pivot = radar_df.pivot_table(
                index='Metric', 
                columns='Channel', 
                values='Normalized'
            ).reset_index()
            
            # Use Plotly for radar chart
            fig = go.Figure()
            
            for channel in radar_pivot.columns[1:]:
                fig.add_trace(go.Scatterpolar(
                    r=radar_pivot[channel],
                    theta=radar_pivot['Metric'],
                    fill='toself',
                    name=channel
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Channel Efficiency Comparison (Normalized)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show original values in a table
            st.subheader("Channel Efficiency Metrics (Original Values)")
            
            original_metrics = pd.DataFrame(metrics_by_channel).T.reset_index()
            original_metrics.columns = ['Channel', 'ROAS', 'Conversion Rate (%)', 'Revenue ($)']
            
            # Format for display
            original_metrics['ROAS'] = original_metrics['ROAS'].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "N/A")
            original_metrics['Conversion Rate (%)'] = original_metrics['Conversion Rate (%)'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
            original_metrics['Revenue ($)'] = original_metrics['Revenue ($)'].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(original_metrics, use_container_width=True)
        else:
            st.info("No channel efficiency data available for the selected filters.")
    except Exception as e:
        st.error(f"Error creating channel efficiency comparison: {e}")

# Channel-specific analysis - show based on selection
if selected_channel == "Social Organic" or selected_channel == "Social Paid":
    st.markdown(f"## {selected_channel} Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Show social media metrics if available
            if not filtered_social.empty and 'Platforms' in filtered_social.columns:
                # Calculate top-level social metrics
                total_followers = filtered_social['Followers'].max() if 'Followers' in filtered_social.columns else 0
                total_engagement = filtered_social['Engagement'].sum() if 'Engagement' in filtered_social.columns else 0
                total_impressions = filtered_social['Impressions'].sum() if 'Impressions' in filtered_social.columns else 0
                
                # Create metrics display
                social_metrics = {
                    "Total Followers": f"{total_followers:,}",
                    "Total Engagement": f"{total_engagement:,}",
                    "Total Impressions": f"{total_impressions:,}",
                    "Engagement Rate": f"{(total_engagement / total_impressions * 100) if total_impressions > 0 else 0:.2f}%"
                }
                
                # Display metrics
                for metric, value in social_metrics.items():
                    st.metric(metric, value)
                
                # Platform breakdown
                if 'Platforms' in filtered_social.columns and 'Engagement' in filtered_social.columns:
                    platform_metrics = filtered_social.groupby('Platforms').agg({
                        'Engagement': 'sum',
                        'Impressions': 'sum',
                        'Followers': 'max',
                        'Website Clicks': 'sum'
                    }).reset_index()
                    
                    platform_metrics['Engagement Rate'] = (platform_metrics['Engagement'] / platform_metrics['Impressions']) * 100
                    
                    # Format for display
                    platform_metrics = platform_metrics.sort_values('Engagement', ascending=False)
                    
                    # Create bar chart for engagement by platform
                    fig = px.bar(
                        platform_metrics,
                        x='Platforms',
                        y='Engagement',
                        title='Social Engagement by Platform',
                        color='Platforms'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No social media data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating social media analysis: {e}")
    
    with col2:
        try:
            # Show social conversion data if available
            if not filtered_revenue.empty and selected_channel in filtered_revenue['Channel'].values:
                # Filter to just this channel
                channel_data = filtered_revenue[filtered_revenue['Channel'] == selected_channel]
                
                # Create time series of metrics
                metrics_over_time = channel_data.groupby('Date').agg({
                    'Revenue': 'sum',
                    'Orders': 'sum',
                    'Web Visitors': 'sum'
                }).reset_index()
                
                metrics_over_time['Conversion Rate'] = (metrics_over_time['Orders'] / metrics_over_time['Web Visitors']) * 100
                
                # Create line chart for conversion rate over time
                fig = px.line(
                    metrics_over_time,
                    x='Date',
                    y='Conversion Rate',
                    title=f'{selected_channel} Conversion Rate Over Time (%)',
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create revenue over time chart
                fig2 = px.line(
                    metrics_over_time,
                    x='Date',
                    y='Revenue',
                    title=f'{selected_channel} Revenue Over Time',
                    markers=True
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(f"No {selected_channel} conversion data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating social conversion analysis: {e}")

elif selected_channel == "Affiliates":
    st.markdown("## Affiliates Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if not filtered_affiliates.empty and 'Type' in filtered_affiliates.columns:
                # Group affiliates by type
                affiliate_types = filtered_affiliates.groupby('Type').agg({
                    'Estimated Value': 'sum',
                    'Cost': 'sum',
                    'Link Clicks': 'sum',
                    'Conversions': 'sum'
                }).reset_index()
                
                # Calculate ROI
                affiliate_types['ROI'] = affiliate_types['Estimated Value'] / affiliate_types['Cost'].replace(0, float('nan'))
                
                # Create bar chart of value by type
                fig = px.bar(
                    affiliate_types,
                    x='Type',
                    y='Estimated Value',
                    title='Affiliate Value by Type',
                    color='Type'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display affiliate metrics table
                affiliate_types['Estimated Value'] = affiliate_types['Estimated Value'].apply(lambda x: f"${x:,.2f}")
                affiliate_types['Cost'] = affiliate_types['Cost'].apply(lambda x: f"${x:,.2f}")
                affiliate_types['ROI'] = affiliate_types['ROI'].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "N/A")
                
                st.dataframe(
                    affiliate_types,
                    column_config={
                        "Type": st.column_config.TextColumn("Affiliate Type"),
                        "Estimated Value": st.column_config.TextColumn("Value"),
                        "Cost": st.column_config.TextColumn("Cost"),
                        "Link Clicks": st.column_config.NumberColumn("Clicks"),
                        "Conversions": st.column_config.NumberColumn("Conversions"),
                        "ROI": st.column_config.TextColumn("ROI"),
                    },
                    use_container_width=True
                )
            else:
                st.info("No affiliate data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating affiliate analysis: {e}")
    
    with col2:
        try:
            if not filtered_affiliates.empty and 'Style Type' in filtered_affiliates.columns:
                # Group by style type
                style_performance = filtered_affiliates.groupby('Style Type').agg({
                    'Estimated Value': 'sum',
                    'Conversions': 'sum'
                }).reset_index()
                
                # Create pie chart
                fig = px.pie(
                    style_performance,
                    values='Estimated Value',
                    names='Style Type',
                    title='Affiliate Value by Style Type',
                    color='Style Type'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top affiliate campaigns if available
                if 'Product Featured' in filtered_affiliates.columns:
                    top_products = filtered_affiliates.groupby('Product Featured').agg({
                        'Estimated Value': 'sum',
                        'Conversions': 'sum'
                    }).reset_index()
                    
                    top_products = top_products.sort_values('Estimated Value', ascending=False).head(5)
                    
                    st.subheader("Top Products in Affiliate Campaigns")
                    
                    top_products['Estimated Value'] = top_products['Estimated Value'].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(
                        top_products,
                        column_config={
                            "Product Featured": st.column_config.TextColumn("Product"),
                            "Estimated Value": st.column_config.TextColumn("Value"),
                            "Conversions": st.column_config.NumberColumn("Conversions"),
                        },
                        use_container_width=True
                    )
            else:
                st.info("No affiliate style data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating affiliate style analysis: {e}")

elif selected_channel == "Email":
    st.markdown("## Email Marketing Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if not filtered_email.empty and 'Email Type' in filtered_email.columns:
                # Calculate key email metrics
                total_subscribers = filtered_email['Subscribers'].max() if 'Subscribers' in filtered_email.columns else 0
                total_sent = filtered_email['Email Sent'].sum() if 'Email Sent' in filtered_email.columns else 0
                total_revenue = filtered_email['Revenue'].sum() if 'Revenue' in filtered_email.columns else 0
                
                # Calculate averages
                avg_open_rate = filtered_email['Open Rate %'].mean() if 'Open Rate %' in filtered_email.columns else 0
                avg_click_rate = filtered_email['Click Rate %'].mean() if 'Click Rate %' in filtered_email.columns else 0
                avg_conversion_rate = filtered_email['Conversion Rate %'].mean() if 'Conversion Rate %' in filtered_email.columns else 0
                
                # Calculate revenue per email
                revenue_per_email = total_revenue / total_sent if total_sent > 0 else 0
                
                # Display key metrics
                email_metrics = {
                    "Total Subscribers": f"{total_subscribers:,}",
                    "Total Emails Sent": f"{total_sent:,}",
                    "Total Revenue": f"${total_revenue:,.2f}",
                    "Revenue per Email": f"${revenue_per_email:.2f}"
                }
                
                for metric, value in email_metrics.items():
                    st.metric(metric, value)
                
                # Group by email type
                email_types = filtered_email.groupby('Email Type').agg({
                    'Revenue': 'sum',
                    'Open Rate %': 'mean',
                    'Click Rate %': 'mean',
                    'Conversion Rate %': 'mean'
                }).reset_index()
                
                # Create bar chart for email performance by type
                fig = px.bar(
                    email_types,
                    x='Email Type',
                    y='Revenue',
                    title='Email Revenue by Type',
                    color='Email Type'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No email marketing data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating email marketing analysis: {e}")
    
    with col2:
        try:
            if not filtered_email.empty and 'Email Type' in filtered_email.columns:
                # Create metrics comparison chart
                email_metrics_long = pd.melt(
                    filtered_email,
                    id_vars=['Email Type', 'Campaign Name'],
                    value_vars=['Open Rate %', 'Click Rate %', 'Conversion Rate %'],
                    var_name='Metric',
                    value_name='Percentage'
                )
                
                # Group by type and metric
                email_metrics_summary = email_metrics_long.groupby(['Email Type', 'Metric'])['Percentage'].mean().reset_index()
                
                # Create grouped bar chart
                fig = px.bar(
                    email_metrics_summary,
                    x='Email Type',
                    y='Percentage',
                    color='Metric',
                    title='Email Performance Metrics by Type (%)',
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show campaign performance if available
                st.subheader("Top Email Campaigns")
                
                top_campaigns = filtered_email.sort_values('Revenue', ascending=False).head(5)
                
                top_campaigns_display = top_campaigns[['Campaign Name', 'Email Type', 'Open Rate %', 'Click Rate %', 'Conversion Rate %', 'Revenue']].copy()
                
                # Format for display
                top_campaigns_display['Open Rate %'] = top_campaigns_display['Open Rate %'].apply(lambda x: f"{x:.2f}%")
                top_campaigns_display['Click Rate %'] = top_campaigns_display['Click Rate %'].apply(lambda x: f"{x:.2f}%")
                top_campaigns_display['Conversion Rate %'] = top_campaigns_display['Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                top_campaigns_display['Revenue'] = top_campaigns_display['Revenue'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(
                    top_campaigns_display,
                    column_config={
                        "Campaign Name": st.column_config.TextColumn("Campaign"),
                        "Email Type": st.column_config.TextColumn("Type"),
                        "Open Rate %": st.column_config.TextColumn("Open Rate"),
                        "Click Rate %": st.column_config.TextColumn("Click Rate"),
                        "Conversion Rate %": st.column_config.TextColumn("Conv. Rate"),
                        "Revenue": st.column_config.TextColumn("Revenue"),
                    },
                    use_container_width=True
                )
            else:
                st.info("No email campaign data available for the selected filters.")
        except Exception as e:
            st.error(f"Error creating email campaign analysis: {e}")

# Campaign Performance by Channel (show for specific channels or all)
st.markdown(f"## {'Channel' if selected_channel == 'All Channels' else selected_channel} Campaign Performance")

try:
    if not filtered_campaigns.empty and 'Channel' in filtered_campaigns.columns:
        # Filter campaigns by selected channel if needed
        if selected_channel != "All Channels":
            channel_campaigns = filtered_campaigns[filtered_campaigns['Channel'] == selected_channel]
        else:
            channel_campaigns = filtered_campaigns
        
        if not channel_campaigns.empty:
            # Calculate campaign metrics
            campaign_metrics = channel_campaigns[['Campaign Name', 'Channel', 'Campaign Type', 'Revenue', 'Spend', 'Conversions', 'ROAS']].copy()
            
            # Sort by revenue
            campaign_metrics = campaign_metrics.sort_values('Revenue', ascending=False)
            
            # Format for display
            display_metrics = campaign_metrics.copy()
            display_metrics['Revenue'] = display_metrics['Revenue'].apply(lambda x: f"${x:,.2f}")
            display_metrics['Spend'] = display_metrics['Spend'].apply(lambda x: f"${x:,.2f}")
            display_metrics['ROAS'] = display_metrics['ROAS'].apply(lambda x: f"{x:.2f}x")
            
            # Display as table
            st.dataframe(
                display_metrics,
                column_config={
                    "Campaign Name": st.column_config.TextColumn("Campaign"),
                    "Channel": st.column_config.TextColumn("Channel"),
                    "Campaign Type": st.column_config.TextColumn("Type"),
                    "Revenue": st.column_config.TextColumn("Revenue"),
                    "Spend": st.column_config.TextColumn("Spend"),
                    "Conversions": st.column_config.NumberColumn("Conversions"),
                    "ROAS": st.column_config.TextColumn("ROAS"),
                },
                use_container_width=True
            )
            
            # Create campaign type comparison (if more than one type)
            campaign_types = channel_campaigns['Campaign Type'].unique()
            
            if len(campaign_types) > 1:
                campaign_type_metrics = channel_campaigns.groupby('Campaign Type').agg({
                    'Revenue': 'sum',
                    'Spend': 'sum',
                    'Conversions': 'sum'
                }).reset_index()
                
                campaign_type_metrics['ROAS'] = campaign_type_metrics['Revenue'] / campaign_type_metrics['Spend']
                campaign_type_metrics['CPA'] = campaign_type_metrics['Spend'] / campaign_type_metrics['Conversions']
                
                # Create bar chart for campaign type performance
                fig = px.bar(
                    campaign_type_metrics,
                    x='Campaign Type',
                    y='Revenue',
                    title='Campaign Performance by Type',
                    color='Campaign Type'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No {selected_channel} campaign data available for the selected filters.")
    else:
        st.info("No campaign data available for the selected filters.")
except Exception as e:
    st.error(f"Error creating campaign performance analysis: {e}")

# Footer with update timestamp
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Data last updated: {current_time}")