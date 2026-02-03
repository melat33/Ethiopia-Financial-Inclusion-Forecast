"""
Trends Page
"""

import streamlit as st
from components.charts import create_trend_chart, create_channel_chart
from components.utils import get_historical_data, get_channel_data

def render_trends_page(data):
    """Render trends page"""
    st.markdown("## 📈 Historical Trends")
    
    # Get data
    historical_data = get_historical_data()
    channel_data = get_channel_data()
    
    # Interactive chart
    st.markdown('<div class="chart-title">Financial Inclusion Trends</div>', unsafe_allow_html=True)
    
    selected_indicators = st.multiselect(
        "Select indicators to display:",
        ['Account Ownership', 'Digital Payments'],
        default=['Account Ownership', 'Digital Payments']
    )
    
    if selected_indicators:
        trend_chart = create_trend_chart(historical_data, selected_indicators)
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.warning("Please select at least one indicator to display.")
    
    # Channel comparison
    st.markdown("### 🔄 Channel Comparison")
    
    channel_chart = create_channel_chart(channel_data)
    st.plotly_chart(channel_chart, use_container_width=True)
    
    # Data table
    with st.expander("View Historical Data"):
        st.dataframe(historical_data, use_container_width=True)
    
    # Insights
    st.markdown("### 📊 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Key Trends:**
        - Account ownership grew 3.5x since 2011
        - Digital payments adoption accelerating
        - Mobile money showing exponential growth
        """)
    
    with col2:
        st.markdown("""
        **🎯 Implications:**
        - Digital channels driving growth
        - Need to accelerate account ownership
        - Mobile infrastructure critical
        """)