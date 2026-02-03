"""
🏆 Ethiopia Financial Inclusion Dashboard
Simple Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Ethiopia FI Dashboard",
    page_icon="🇪🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #4ECDC4 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86AB;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load forecast data"""
    try:
        # Load forecasts
        forecasts_df = pd.read_csv('models/task4/forecasts_2025_2027.csv')
        
        # Load scenario analysis if exists
        scenario_data = {}
        try:
            with open('models/task4/scenario_analysis.json', 'r') as f:
                scenario_data = json.load(f)
        except:
            pass
        
        return {
            'forecasts': forecasts_df,
            'scenarios': scenario_data
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ========== PAGE RENDERING FUNCTIONS (DEFINED BEFORE USE) ==========

def render_overview(data):
    """Render overview page"""
    st.markdown("## 📊 Executive Overview")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Account Ownership</div>
            <div class="metric-value">49.0%</div>
            <div style="color: #4CAF50;">▲ 3.0% from 2021</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Digital Payments</div>
            <div class="metric-value">35.0%</div>
            <div style="color: #4CAF50;">▲ 10.0% from 2021</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">NFIS-II 2025 Gap</div>
            <div class="metric-value">21.0 pp</div>
            <div style="color: #FF6B6B;">▼ Needs acceleration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">P2P/ATM Ratio</div>
            <div class="metric-value">1.8x</div>
            <div style="color: #4CAF50;">▲ Digital growing</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress chart
    st.markdown("### 🎯 Progress Towards NFIS-II Targets")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current (2024)',
        x=['Account Ownership', 'Digital Payments'],
        y=[49, 35],
        marker_color=['#2E86AB', '#4ECDC4']
    ))
    
    fig.add_trace(go.Bar(
        name='NFIS-II 2025 Target',
        x=['Account Ownership', 'Digital Payments'],
        y=[70, 45],
        marker_color=['rgba(46, 134, 171, 0.3)', 'rgba(78, 205, 196, 0.3)']
    ))
    
    fig.update_layout(
        barmode='overlay',
        height=400,
        plot_bgcolor='white',
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick insights
    st.markdown("### 💡 Quick Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🚀 Fastest Growing**
        Mobile Money: 8.3pp/year growth
        """)
    
    with col2:
        st.warning("""
        **⚠️ Needs Attention**
        Account Ownership gap: 21pp
        """)
    
    with col3:
        st.success("""
        **✅ On Track**
        Digital Payments growth: 5.0pp/year
        """)

def render_trends(data):
    """Render trends page"""
    st.markdown("## 📈 Historical Trends")
    
    # Sample historical data
    historical_data = pd.DataFrame({
        'year': [2011, 2014, 2017, 2021, 2024],
        'Account Ownership': [14.0, 22.0, 35.0, 46.0, 49.0],
        'Digital Payments': [10.0, 18.0, 25.0, 35.0, 35.0]
    })
    
    # Interactive chart
    selected_indicators = st.multiselect(
        "Select indicators to display:",
        ['Account Ownership', 'Digital Payments'],
        default=['Account Ownership', 'Digital Payments']
    )
    
    fig = go.Figure()
    
    colors = {'Account Ownership': '#2E86AB', 'Digital Payments': '#4ECDC4'}
    
    for indicator in selected_indicators:
        fig.add_trace(go.Scatter(
            x=historical_data['year'],
            y=historical_data[indicator],
            name=indicator,
            mode='lines+markers',
            line=dict(color=colors.get(indicator, '#2E86AB'), width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        height=500,
        title="Historical Trends",
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Channel comparison
    st.markdown("### 🔄 Channel Comparison")
    
    channel_data = pd.DataFrame({
        'Year': [2014, 2017, 2021, 2024],
        'Bank Accounts': [18, 28, 38, 42],
        'Mobile Money': [2, 5, 8, 12],
        'Microfinance': [2, 2, 3, 4]
    })
    
    fig2 = px.line(
        channel_data.melt(id_vars=['Year'], var_name='Channel', value_name='Usage'),
        x='Year',
        y='Usage',
        color='Channel',
        markers=True,
        title="Channel Adoption Over Time"
    )
    
    fig2.update_layout(height=400, plot_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

def render_forecasts(data):
    """Render forecasts page"""
    st.markdown("## 🔮 2025-2027 Forecasts")
    
    if data and 'forecasts' in data:
        forecasts_df = data['forecasts']
        
        # Display forecast table
        st.markdown("### 📊 Forecast Data")
        st.dataframe(forecasts_df, use_container_width=True)
        
        # Visualize forecasts
        st.markdown("### 📈 Forecast Visualization")
        
        # Pivot for plotting
        pivot_df = forecasts_df.pivot(index='year', columns='indicator', values='forecast_value')
        
        fig = go.Figure()
        
        for column in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df[column],
                name=column.replace('_', ' '),
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            height=500,
            title="Financial Inclusion Forecasts 2025-2027",
            xaxis_title="Year",
            yaxis_title="Percentage (%)",
            yaxis_range=[0, 100],
            plot_bgcolor='white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No forecast data available. Generating sample data...")
        
        # Sample forecast data
        sample_data = pd.DataFrame({
            'year': [2025, 2026, 2027, 2025, 2026, 2027],
            'indicator': ['ACC_OWNERSHIP', 'ACC_OWNERSHIP', 'ACC_OWNERSHIP', 
                         'USG_DIGITAL_PAYMENT', 'USG_DIGITAL_PAYMENT', 'USG_DIGITAL_PAYMENT'],
            'forecast_value': [54.2, 56.9, 59.7, 36.4, 38.8, 41.2],
            'model_type': ['ensemble', 'ensemble', 'ensemble', 'ensemble', 'ensemble', 'ensemble']
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        # Visualize sample data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[2025, 2026, 2027],
            y=[54.2, 56.9, 59.7],
            name='Account Ownership',
            mode='lines+markers',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=[2025, 2026, 2027],
            y=[36.4, 38.8, 41.2],
            name='Digital Payments',
            mode='lines+markers',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            height=500,
            title="Sample Forecasts 2025-2027",
            xaxis_title="Year",
            yaxis_title="Percentage (%)",
            yaxis_range=[0, 100],
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario comparison
    st.markdown("### 🎭 Scenario Comparison")
    
    scenario_data = pd.DataFrame({
        'Scenario': ['Optimistic', 'Baseline', 'Pessimistic'],
        'Account Ownership 2027': [67.5, 62.5, 56.6],
        'Digital Payments 2027': [53.1, 45.4, 36.5]
    })
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        name='Account Ownership',
        x=scenario_data['Scenario'],
        y=scenario_data['Account Ownership 2027'],
        marker_color='#2E86AB'
    ))
    
    fig2.add_trace(go.Bar(
        name='Digital Payments',
        x=scenario_data['Scenario'],
        y=scenario_data['Digital Payments 2027'],
        marker_color='#4ECDC4'
    ))
    
    fig2.update_layout(
        height=400,
        barmode='group',
        plot_bgcolor='white',
        title="2027 Forecast by Scenario",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def render_download(data):
    """Render download page"""
    st.markdown("## 📥 Data Export")
    
    st.markdown("### Available Datasets")
    
    datasets = [
        "Forecasts 2025-2027",
        "Scenario Analysis", 
        "Target Gap Analysis",
        "Historical Data",
        "Event Matrix"
    ]
    
    for dataset in datasets:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{dataset}**")
        
        with col2:
            if st.button(f"Download {dataset}", key=f"btn_{dataset}"):
                if dataset == "Forecasts 2025-2027" and data and 'forecasts' in data:
                    csv = data['forecasts'].to_csv(index=False)
                    st.download_button(
                        label="Click to download",
                        data=csv,
                        file_name="forecasts_2025_2027.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Sample data would be downloaded in production")
    
    st.markdown("---")
    st.markdown("### 🔧 Custom Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        indicators = st.multiselect(
            "Select Indicators:",
            ["Account Ownership", "Digital Payments", "Mobile Money", "All"],
            default=["Account Ownership", "Digital Payments"]
        )
    
    with col2:
        years = st.multiselect(
            "Select Years:",
            ["2024", "2025", "2026", "2027", "All"],
            default=["2025", "2026", "2027"]
        )
    
    if st.button("Generate Custom Export", type="primary"):
        with st.spinner("Generating export..."):
            # Create sample export data
            export_df = pd.DataFrame({
                'Year': [2025, 2026, 2027],
                'Account_Ownership': [54.2, 56.9, 59.7],
                'Digital_Payments': [36.4, 38.8, 41.2]
            })
            
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="custom_export.csv",
                mime="text/csv"
            )
            
            st.success("Export generated successfully!")

# ========== MAIN APPLICATION CODE ==========

# Initialize session state
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = 'baseline'

# Sidebar
with st.sidebar:
    st.markdown("## 🇪🇹 Navigation")
    
    # Simple radio buttons for navigation
    page = st.radio(
        "Go to:",
        ["📊 Overview", "📈 Trends", "🔮 Forecasts", "📥 Download"]
    )
    
    st.markdown("---")
    st.markdown("## 🎯 Scenario")
    
    scenario = st.radio(
        "Select scenario:",
        ["Optimistic", "Baseline", "Pessimistic"],
        index=1
    )
    st.session_state.current_scenario = scenario.lower()
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Sample metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Account Ownership", "49.0%", "3.0%")
    with col2:
        st.metric("Digital Payments", "35.0%", "10.0%")

# Load data
data = load_data()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🇪🇹 Ethiopia Financial Inclusion Dashboard</h1>
    <p>Interactive Analytics & Forecasting Platform</p>
</div>
""", unsafe_allow_html=True)

# Page routing
if page == "📊 Overview":
    render_overview(data)
elif page == "📈 Trends":
    render_trends(data)
elif page == "🔮 Forecasts":
    render_forecasts(data)
elif page == "📥 Download":
    render_download(data)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🇪🇹 Ethiopia Financial Inclusion Dashboard • Generated on {}</p>
    <p>Data Source: Ethiopia Findex & NFIS-II Targets</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)