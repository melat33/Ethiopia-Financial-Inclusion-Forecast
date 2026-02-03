"""
Sidebar Component
"""

import streamlit as st
import config

def render_sidebar():
    """Render the sidebar"""
    with st.sidebar:
        # Logo and title
        st.markdown("## 🇪🇹 Navigation")
        
        # Page selection
        page = st.radio(
            "Go to:",
            ["📊 Overview", "📈 Trends", "🔮 Forecasts", "📥 Download"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("## 🎯 Scenario")
        
        # Scenario selection
        scenario = st.radio(
            "Select scenario:",
            config.SCENARIOS,
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
        
        # Info section
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666;">
            <strong>Data Sources:</strong><br>
            • Ethiopia Findex<br>
            • NFIS-II Targets<br>
            • Ensemble Forecasts
        </div>
        """, unsafe_allow_html=True)
    
    return page