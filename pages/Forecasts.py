"""
Forecasts Page
"""

import streamlit as st
import pandas as pd
from components.charts import create_forecast_chart, create_scenario_chart
from components.utils import get_sample_forecasts

def render_forecasts_page(data):
    """Render forecasts page"""
    st.markdown("## 🔮 2025-2027 Forecasts")
    
    # Get forecast data
    if data and 'forecasts' in data:
        forecast_df = data['forecasts']
    else:
        forecast_df = get_sample_forecasts()
        st.info("⚠️ Showing sample forecast data. Real data will load from models/task4/forecasts_2025_2027.csv")
    
    # Display forecast table
    st.markdown("### 📊 Forecast Data")
    st.dataframe(forecast_df, use_container_width=True)
    
    # Visualize forecasts
    st.markdown("### 📈 Forecast Visualization")
    forecast_chart = create_forecast_chart(forecast_df)
    st.plotly_chart(forecast_chart, use_container_width=True)
    
    # Scenario comparison
    st.markdown("### 🎭 Scenario Comparison")
    scenario_chart = create_scenario_chart()
    st.plotly_chart(scenario_chart, use_container_width=True)
    
    # Forecast insights
    st.markdown("### 💡 Forecast Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Account Ownership 2027",
            "62.5%",
            "13.5% growth"
        )
    
    with col2:
        st.metric(
            "Digital Payments 2027",
            "45.4%",
            "10.4% growth"
        )
    
    with col3:
        current_scenario = st.session_state.get('current_scenario', 'baseline')
        st.metric(
            "Current Scenario",
            current_scenario.title(),
            "Viewing"
        )
    
    # Model information
    with st.expander("📋 About the Forecast Model"):
        st.markdown("""
        **Ensemble Forecasting Approach:**
        
        - Combines multiple statistical models
        - Accounts for economic indicators
        - Includes policy intervention scenarios
        - Updated quarterly
        
        **Key Assumptions:**
        1. Steady economic growth (6-8%)
        2. Continued digital infrastructure expansion
        3. Policy implementation as planned
        4. No major economic disruptions
        
        **Confidence Intervals:** 90% confidence level applied
        """)