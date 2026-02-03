"""
Overview Page
"""

import streamlit as st
from components.metrics import render_metric_grid
from components.charts import create_progress_chart

def render_overview_page(data):
    """Render overview page"""
    st.markdown("## 📊 Executive Overview")
    
    # Key metrics
    render_metric_grid()
    
    # Progress chart
    st.markdown("### 🎯 Progress Towards NFIS-II Targets")
    progress_chart = create_progress_chart()
    st.plotly_chart(progress_chart, use_container_width=True)
    
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
    
    # Additional context
    st.markdown("### 📋 Context")
    
    with st.expander("About NFIS-II Targets"):
        st.markdown("""
        **National Financial Inclusion Strategy II (NFIS-II) 2025 Targets:**
        
        - **Account Ownership**: 70% (currently 49%)
        - **Digital Payments**: 45% (currently 35%)
        - **Mobile Money**: 30% (currently 12%)
        
        **Key Focus Areas:**
        1. Women's financial inclusion
        2. Rural outreach
        3. Digital infrastructure
        4. Financial literacy
        
        **Timeframe**: 2021-2025
        """)