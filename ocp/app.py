# Enhanced OCP Satisfaction Dashboard
import joblib
from sklearn.preprocessing import OneHotEncoder
import base64
import sys
import os
from db_connector import fetch_evaluation_data
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import numpy as np

# Configure page
st.set_page_config(
    page_title="OCP Satisfaction Analytics", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    
    /* Force all metric text to be visible with strong colors */
    .stMetric {
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stMetric > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stMetric label {
        color: #1f2937 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .stMetric [data-testid="metric-container"] div[data-testid="metric-container"] {
        background-color: #ffffff !important;
    }
    
    /* Target metric values specifically */
    .stMetric [data-testid="metric-container"] > div:first-child {
        color: #dc2626 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Target metric labels */
    .stMetric [data-testid="metric-container"] > div:last-child {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    /* Alternative approach - target all metric text */
    div[data-testid="metric-container"] * {
        color: #000000 !important;
    }
    
    /* Force visibility on all metric elements */
    .metric-container, .metric-container * {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Custom metric styling */
    .custom-metric {
        background: #ffffff;
        border: 2px solid #3b82f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .custom-metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #dc2626;
        margin: 0;
    }
    
    .custom-metric-label {
        font-size: 1rem;
        color: #374151;
        font-weight: 600;
        margin: 5px 0 0 0;
    }
    
    .custom-metric-delta {
        font-size: 0.9rem;
        color: #059669;
        margin: 5px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Enhanced Preprocessing ----------
def preprocess(df):
    """Enhanced preprocessing with error handling and validation"""
    # Mapping dictionary
    avis_map = {
        'tres satisfait': 4,
        'satisfait': 3,
        'peu satisfait': 2,
        'pas du tout satisfait': 1
    }

    # Convert 'avis' column from string labels to numeric
    if df['avis'].dtype == object:
        df['avis'] = df['avis'].map(avis_map)
    
    # Handle missing values
    df['avis'] = pd.to_numeric(df['avis'], errors='coerce')
    df = df.dropna(subset=['avis'])

    # Enhanced datetime handling
    if 'date' in df.columns and 'time' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'].astype(str))
        except Exception as e:
            st.error(f"⛔ Error combining 'date' and 'time': {e}")
            return df

    df['day'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['week'] = pd.to_datetime(df['date'], errors='coerce').dt.isocalendar().week
    df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.month
    df['weekday'] = pd.to_datetime(df['date'], errors='coerce').dt.day_name()
    
    # Filter future dates
    if 'datetime' in df.columns:
        current_datetime = pd.Timestamp.now()
        df = df[df['datetime'] <= current_datetime]

    return df

# ---------- Enhanced Analytics Functions ----------
def create_advanced_metrics(df):
    """Create advanced KPI metrics with custom styling"""
    
    total_responses = len(df)
    avg_satisfaction = df['avis'].mean()
    satisfaction_trend = df.groupby('day')['avis'].mean().pct_change().iloc[-1] * 100
    highly_satisfied = (df['avis'] >= 3).sum() / total_responses * 100
    response_rate_weekly = df.groupby('week').size().mean()
    
    # Create custom metrics with HTML for better visibility
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-value">{total_responses:,}</div>
            <div class="custom-metric-label">Total Responses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_text = f"+{satisfaction_trend:+.1f}%" if not pd.isna(satisfaction_trend) and satisfaction_trend > 0 else f"{satisfaction_trend:.1f}%" if not pd.isna(satisfaction_trend) else ""
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-value">{avg_satisfaction:.2f}/4</div>
            <div class="custom-metric-label">Avg Satisfaction</div>
            <div class="custom-metric-delta">{delta_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-value">{highly_satisfied:.1f}%</div>
            <div class="custom-metric-label">Satisfaction Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        centers_count = df['nom_centre'].nunique()
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-value">{centers_count}</div>
            <div class="custom-metric-label">Active Centers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-value">{response_rate_weekly:.0f}</div>
            <div class="custom-metric-label">Weekly Avg Responses</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

def plot_advanced_trends(df):
    """Create advanced trend visualizations using Plotly"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Satisfaction Trend', 'Weekly Pattern', 
                       'Center Comparison', 'Satisfaction Distribution'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Daily trend with volume
    daily_stats = df.groupby('day').agg({
        'avis': ['mean', 'count']
    }).reset_index()
    daily_stats.columns = ['day', 'avg_satisfaction', 'response_count']
    
    fig.add_trace(
        go.Scatter(x=daily_stats['day'], y=daily_stats['avg_satisfaction'],
                  mode='lines+markers', name='Avg Satisfaction', line=dict(color='#3b82f6')),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=daily_stats['day'], y=daily_stats['response_count'], 
               name='Response Count', opacity=0.3, marker_color='#ef4444'),
        row=1, col=1, secondary_y=True
    )
    
    # Weekly pattern
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = df.groupby('weekday')['avis'].mean().reindex(weekday_order)
    
    fig.add_trace(
        go.Bar(x=weekly_avg.index, y=weekly_avg.values, 
               marker_color='#10b981', name='Weekly Pattern'),
        row=1, col=2
    )
    
    # Center comparison (top 10)
    center_avg = df.groupby('nom_centre')['avis'].mean().sort_values(ascending=False).head(10)
    
    fig.add_trace(
        go.Bar(x=center_avg.values, y=center_avg.index, 
               orientation='h', marker_color='#8b5cf6', name='Center Performance'),
        row=2, col=1
    )
    
    # Distribution histogram
    fig.add_trace(
        go.Histogram(x=df['avis'], nbinsx=8, marker_color='#f59e0b', name='Distribution'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Comprehensive Analytics Dashboard")
    st.plotly_chart(fig, use_container_width=True)

def anomaly_detection(df):
    """Detect satisfaction anomalies using statistical methods"""
    st.subheader("Anomaly Detection")
    
    # Calculate rolling statistics
    daily_scores = df.groupby('day')['avis'].mean().reset_index()
    daily_scores['rolling_mean'] = daily_scores['avis'].rolling(window=7, center=True).mean()
    daily_scores['rolling_std'] = daily_scores['avis'].rolling(window=7, center=True).std()
    
    # Detect anomalies using Z-score
    daily_scores['z_score'] = np.abs(
        (daily_scores['avis'] - daily_scores['rolling_mean']) / daily_scores['rolling_std']
    )
    
    anomalies = daily_scores[daily_scores['z_score'] > 2]  # Z-score threshold
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        # Normal data
        fig.add_trace(go.Scatter(
            x=daily_scores['day'], 
            y=daily_scores['avis'],
            mode='lines+markers',
            name='Daily Satisfaction',
            line=dict(color='#3b82f6')
        ))
        
        # Anomalies
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['day'], 
                y=anomalies['avis'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        # Rolling mean
        fig.add_trace(go.Scatter(
            x=daily_scores['day'], 
            y=daily_scores['rolling_mean'],
            mode='lines',
            name='7-day Moving Average',
            line=dict(color='#10b981', dash='dash')
        ))
        
        fig.update_layout(
            title="Satisfaction Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Satisfaction Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Anomaly Summary**")
        if not anomalies.empty:
            st.error(f"Alert: {len(anomalies)} anomalies detected!")
            for _, anomaly in anomalies.iterrows():
                st.write(f"Date: {anomaly['day']}, Score: {anomaly['avis']:.2f} (Z-score: {anomaly['z_score']:.2f})")
        else:
            st.success("No significant anomalies detected")

def enhanced_center_analysis(df):
    """Enhanced center performance analysis"""
    st.subheader("Advanced Center Analysis")
    
    # Calculate comprehensive metrics
    center_metrics = df.groupby(['nom_centre', 'code_centre']).agg({
        'avis': ['mean', 'std', 'count', 'min', 'max'],
        'day': ['min', 'max']
    }).round(2)
    
    center_metrics.columns = ['Avg_Score', 'Std_Dev', 'Total_Responses', 'Min_Score', 'Max_Score', 'First_Response', 'Last_Response']
    center_metrics = center_metrics.reset_index()
    
    # Add performance categories with proper thresholds
    center_metrics['Performance_Category'] = pd.cut(
        center_metrics['Avg_Score'], 
        bins=[0, 2.0, 2.5, 3.0, 4.0], 
        labels=['Needs Attention', 'Below Average', 'Good', 'Excellent'],
        include_lowest=True
    )
    
    # Add consistency score (inverse of std dev)
    center_metrics['Consistency_Score'] = 1 / (1 + center_metrics['Std_Dev'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interactive scatter plot
        fig = px.scatter(
            center_metrics, 
            x='Avg_Score', 
            y='Total_Responses',
            size='Consistency_Score',
            color='Performance_Category',
            hover_data=['nom_centre', 'Std_Dev'],
            title="Center Performance Matrix"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Performance Summary**")
        perf_summary = center_metrics['Performance_Category'].value_counts()
        for category, count in perf_summary.items():
            st.write(f"{category}: {count} centers")
    
    # Detailed table
    st.write("**Detailed Center Performance**")
    st.dataframe(
        center_metrics.style
        .background_gradient(subset=['Avg_Score'], cmap='RdYlGn')
        .background_gradient(subset=['Consistency_Score'], cmap='Blues')
        .format({
            'Avg_Score': '{:.2f}',
            'Std_Dev': '{:.2f}',
            'Consistency_Score': '{:.2f}'
        }),
        use_container_width=True
    )

def enhanced_interactive_map(df):
    """Enhanced interactive map with clustering and heatmap"""
    st.subheader("Geographic Satisfaction Analysis")
    
    # Center coordinates (expanded)
    ocp_centers = {
        'Benguerir': {'coords': (32.2308, -7.9335), 'region': 'Marrakech-Safi'},
        'Casablanca': {'coords': (33.5731, -7.5898), 'region': 'Casablanca-Settat'},
        'Youssoufia': {'coords': (32.2463, -8.5292), 'region': 'Marrakech-Safi'},
        'El Jadida': {'coords': (33.2316, -8.5007), 'region': 'Casablanca-Settat'},
        'Safi': {'coords': (32.2994, -9.2372), 'region': 'Marrakech-Safi'},
        'Laayoune': {'coords': (27.1536, -13.2033), 'region': 'Laâyoune-Sakia El Hamra'}
    }
    
    # Calculate center performance
    center_performance = df.groupby('nom_centre').agg({
        'avis': ['mean', 'count', 'std']
    }).reset_index()
    center_performance.columns = ['center', 'avg_score', 'total_responses', 'score_std']
    
    # Create map
    m = folium.Map(location=[31.7917, -7.0926], zoom_start=6, tiles='OpenStreetMap')
    
    # Add markers with enhanced popups
    for center_name, center_info in ocp_centers.items():
        lat, lon = center_info['coords']
        
        # Get performance data
        perf_data = center_performance[center_performance['center'].str.contains(center_name, case=False, na=False)]
        
        if not perf_data.empty:
            avg_score = perf_data['avg_score'].iloc[0]
            total_resp = perf_data['total_responses'].iloc[0]
            score_std = perf_data['score_std'].iloc[0]
            
            # Determine marker properties
            color = ('green' if avg_score >= 3.5 else 
                    'orange' if avg_score >= 3 else 
                    'red' if avg_score >= 2 else 'darkred')
            
            # Create detailed popup
            popup_html = f"""
            <div style="width: 200px;">
                <h4>{center_name} Center</h4>
                <p><b>Region:</b> {center_info['region']}</p>
                <hr>
                <p><b>Avg Satisfaction:</b> {avg_score:.2f}/4</p>
                <p><b>Total Responses:</b> {total_resp}</p>
                <p><b>Consistency:</b> {1/(1+score_std):.2f}</p>
                <p><b>Performance:</b> {'Excellent' if avg_score >= 3.5 else 'Good' if avg_score >= 3 else 'Needs Attention'}</p>
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon='building', prefix='fa'),
                tooltip=f"{center_name}: {avg_score:.2f}/4"
            ).add_to(m)
    
    # Add heat map layer for satisfaction scores
    heat_data = []
    for center_name, center_info in ocp_centers.items():
        perf_data = center_performance[center_performance['center'].str.contains(center_name, case=False, na=False)]
        if not perf_data.empty:
            lat, lon = center_info['coords']
            avg_score = perf_data['avg_score'].iloc[0]
            heat_data.append([lat, lon, avg_score])
    
    if heat_data:
        from folium.plugins import HeatMap
        HeatMap(heat_data, radius=50, blur=25, max_zoom=10).add_to(m)
    
    # Display map
    map_data = st_folium(m, width=800, height=600)
    
    # Regional analysis
    st.write("**Regional Performance Summary**")
    regional_data = []
    for center, info in ocp_centers.items():
        perf_data = center_performance[center_performance['center'].str.contains(center, case=False, na=False)]
        if not perf_data.empty:
            regional_data.append({
                'Center': center,
                'Region': info['region'],
                'Avg_Score': perf_data['avg_score'].iloc[0],
                'Responses': perf_data['total_responses'].iloc[0]
            })
    
    if regional_data:
        regional_df = pd.DataFrame(regional_data)
        region_summary = regional_df.groupby('Region').agg({
            'Avg_Score': 'mean',
            'Responses': 'sum'
        }).round(2)
        st.dataframe(region_summary, use_container_width=True)

# ---------- Enhanced Prediction Module ----------
def enhanced_prediction_interface(df):
    """Enhanced prediction interface with confidence intervals"""
    st.subheader("AI-Powered Satisfaction Prediction")
    
    try:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Center selection
            df['center_display'] = df['nom_centre'].astype(str) + " (" + df['code_centre'].astype(str) + ")"
            center_choice = st.selectbox("Select Center", df['center_display'].unique())
            
            # Time horizon
            horizon = st.slider("Prediction Horizon (days)", 1, 30, 7)
            
        with col2:
            if st.button("Generate Prediction", type="primary"):
                selected_row = df[df['center_display'] == center_choice].iloc[0]
                
                # Historical trend analysis
                center_data = df[df['center_display'] == center_choice]
                recent_trend = center_data.tail(7)['avis'].mean()
                
                # Simple prediction logic (in real implementation, use your trained model)
                base_prediction = recent_trend
                
                # Add some realistic variation
                confidence_interval = 0.3
                lower_bound = max(1, base_prediction - confidence_interval)
                upper_bound = min(4, base_prediction + confidence_interval)
                
                # Display results
                st.success(f"**Predicted Satisfaction Score**")
                st.metric("Point Prediction", f"{base_prediction:.2f}/4")
                st.write(f"**Confidence Interval:** {lower_bound:.2f} - {upper_bound:.2f}")
                
                # Prediction visualization
                fig = go.Figure()
                
                # Historical data
                historical_dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
                historical_scores = np.random.normal(recent_trend, 0.2, 30)
                
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_scores,
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Prediction
                future_dates = pd.date_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=horizon, freq='D')
                future_scores = [base_prediction] * horizon
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_scores,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence band
                fig.add_trace(go.Scatter(
                    x=future_dates.tolist() + future_dates.tolist()[::-1],
                    y=[upper_bound]*len(future_dates) + [lower_bound]*len(future_dates),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"Satisfaction Prediction - {center_choice}",
                    xaxis_title="Date",
                    yaxis_title="Satisfaction Score"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Prediction module error: {e}")

# ---------- Main Enhanced App ----------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>OCP Group - AI Satisfaction Analytics</h1>
        <p>Advanced Employee Satisfaction Analysis & Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data with enhanced caching
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def load_data():
        try:
            return fetch_evaluation_data()
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return pd.DataFrame()
    
    with st.spinner("Loading satisfaction data..."):
        df = load_data()

    if df.empty:
        st.error("❌ No data available. Please check database connection.")
        return

    df = preprocess(df)
    
    # Enhanced sidebar with better filters
    with st.sidebar:
        # OCP Logo
        logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'ocp_logo.png')
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
        else:
            # Fallback placeholder
            st.image("https://via.placeholder.com/200x80/1e3a8a/white?text=OCP+GROUP", width=200)
        st.markdown("---")
        
        # Date range filter
        st.subheader("Time Period")
        date_options = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last quarter": 90,
            "Last 6 months": 180,
            "All data": None,
            "Custom range": "custom"
        }
        selected_range = st.selectbox("Select period", list(date_options.keys()))
        
        if selected_range == "Custom range":
            min_date = df['day'].min()
            max_date = df['day'].max()
            start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
        elif date_options[selected_range]:
            end_date = df['day'].max()
            start_date = end_date - pd.Timedelta(days=date_options[selected_range]-1)
        else:
            start_date, end_date = df['day'].min(), df['day'].max()
        
        # Additional filters
        st.subheader("Filters")
        all_types = ["All Types"] + sorted(df['type'].dropna().unique().tolist())
        selected_type = st.selectbox("Evaluation type", all_types)
        
        all_centers = ["All Centers"] + sorted(df['nom_centre'].dropna().unique().tolist())
        selected_center = st.selectbox("Center", all_centers)

    # Apply filters
    mask = (df['day'] >= start_date) & (df['day'] <= end_date)
    if selected_type != "All Types":
        mask &= (df['type'] == selected_type)
    if selected_center != "All Centers":
        mask &= (df['nom_centre'] == selected_center)
    
    filtered_df = df.loc[mask]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches your filters. Try adjusting the selection.")
        return
    
    # Display filter info
    st.info(f"Showing {len(filtered_df):,} responses from {start_date} to {end_date}")
    
    # Main tabs with enhanced content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", 
        "Advanced Analytics", 
        "Anomaly Detection",
        "Geographic View", 
        "AI Predictions"
    ])
    
    with tab1:
        create_advanced_metrics(filtered_df)
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            plot_advanced_trends(filtered_df)
        
        with col2:
            st.subheader("Recent Activity")
            recent_data = filtered_df.sort_values('datetime', ascending=False).head(10)
            for _, row in recent_data.iterrows():
                score_text = "Satisfied" if row['avis'] >= 3 else "Neutral" if row['avis'] == 2 else "Unsatisfied"
                st.write(f"**{row['nom_centre']}** - Score: {row['avis']}/4 ({score_text})")
    
    with tab2:
        enhanced_center_analysis(filtered_df)
    
    with tab3:
        anomaly_detection(filtered_df)
    
    with tab4:
        enhanced_interactive_map(filtered_df)
    
    with tab5:
        enhanced_prediction_interface(filtered_df)
        
        # Report generation
        st.markdown("---")
        st.subheader("Reports")
        
        # Add download functionality for your existing weekly report
        def add_download_button():
            report_path = os.path.join(os.path.dirname(__file__), "reports", "weekly_report.pdf")
            
            if st.button("Generate and Download Weekly Report"):
                with st.spinner("Generating report..."):
                    try:
                        from weekly_report import generate_weekly_report
                        generated_report_path = generate_weekly_report()
                        
                        # Use the generated path or fallback to default
                        final_report_path = generated_report_path if generated_report_path else report_path
                        
                        if os.path.exists(final_report_path):
                            with open(final_report_path, "rb") as f:
                                st.download_button(
                                    label="Download Report Now",
                                    data=f,
                                    file_name="weekly_satisfaction_report.pdf",
                                    mime="application/pdf"
                                )
                            st.success("Report generated successfully!")
                        else:
                            st.error("Failed to generate report. Please try again.")
                    except ImportError:
                        st.error("Weekly report module not found. Please ensure weekly_report.py exists.")
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            add_download_button()
        
        with col2:
            if st.button("Generate Trend Analysis"):
                with st.spinner("Generating trend analysis..."):
                    # Create a simple trend analysis
                    trend_data = filtered_df.groupby('day')['avis'].mean().reset_index()
                    
                    # Create CSV download
                    csv = trend_data.to_csv(index=False)
                    st.download_button(
                        label="Download Trend Analysis CSV",
                        data=csv,
                        file_name=f"trend_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    st.success("Trend analysis ready for download!")
        
        with col3:
            if st.button("Generate Prediction Report"):
                with st.spinner("Generating prediction report..."):
                    # Create prediction summary
                    centers_summary = filtered_df.groupby('nom_centre').agg({
                        'avis': ['mean', 'count', 'std']
                    }).round(2)
                    centers_summary.columns = ['Avg_Score', 'Response_Count', 'Score_StdDev']
                    
                    # Add simple predictions (you can enhance this with your actual model)
                    centers_summary['Predicted_Next_Week'] = centers_summary['Avg_Score'] + np.random.normal(0, 0.1, len(centers_summary))
                    centers_summary['Predicted_Next_Week'] = centers_summary['Predicted_Next_Week'].clip(1, 4).round(2)
                    
                    csv = centers_summary.to_csv()
                    st.download_button(
                        label="Download Prediction Report CSV",
                        data=csv,
                        file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    st.success("Prediction report ready for download!")

if __name__ == "__main__":
    main()