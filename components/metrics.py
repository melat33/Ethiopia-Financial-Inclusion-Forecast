"""
Metrics Components
"""

import streamlit as st
import config

class MetricCard:
    """Reusable metric card component"""
    
    @staticmethod
    def render(label, value, change=None, change_label=None, color=None):
        """
        Render a metric card
        
        Args:
            label: Metric label
            value: Metric value
            change: Change value
            change_label: Change description
            color: Card color
        """
        if color is None:
            color = config.COLORS["primary"]
        
        # Determine change styling
        if change is not None:
            if change > 0:
                change_class = "change-positive"
                change_icon = "▲"
            else:
                change_class = "change-negative"
                change_icon = "▼"
            
            change_html = f'<div class="metric-change {change_class}">{change_icon} {abs(change):.1f}% {change_label or ""}</div>'
        else:
            change_html = ""
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
            {change_html}
        </div>
        """, unsafe_allow_html=True)


def render_metric_grid():
    """Render a grid of key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        MetricCard.render(
            label="Account Ownership",
            value="49.0%",
            change=3.0,
            change_label="from 2021",
            color=config.COLORS["primary"]
        )
    
    with col2:
        MetricCard.render(
            label="Digital Payments",
            value="35.0%",
            change=10.0,
            change_label="from 2021",
            color=config.COLORS["secondary"]
        )
    
    with col3:
        MetricCard.render(
            label="NFIS-II 2025 Gap",
            value="21.0 pp",
            change=-5.0,
            change_label="needs acceleration",
            color=config.COLORS["warning"]
        )
    
    with col4:
        MetricCard.render(
            label="P2P/ATM Ratio",
            value="1.8x",
            change=0.3,
            change_label="digital growing",
            color=config.COLORS["success"]
        )