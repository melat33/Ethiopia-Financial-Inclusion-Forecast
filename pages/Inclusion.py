"""
Inclusion Page - Progress towards 60% target
"""

import streamlit as st
import pandas as pd
from components.header import render_header
from components.inclusion_components import (
    render_target_progress_chart,
    render_current_status_gauge,
    render_scenario_cards,
    render_scenario_trajectories,
    render_required_interventions,
    render_implementation_roadmap,
    render_progress_breakdown
)

def main():
    """Main function for Inclusion page"""
    
    # Render header
    render_header(
        title="🎯 Financial Inclusion Projections",
        subtitle="Track progress towards 60% financial inclusion target",
        show_status=True
    )
    
    # Target progress visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chart-title">🏁 Progress Towards 60% Financial Inclusion Target</div>', unsafe_allow_html=True)
        
        # Get the chart and related data
        progress_chart, current_rate, progress_percent, target = render_target_progress_chart()
        st.plotly_chart(progress_chart, use_container_width=True)
        
        # Progress breakdown
        st.markdown("#### 📊 Progress Breakdown")
        render_progress_breakdown(current_rate, progress_percent, target)
    
    with col2:
        st.markdown('<div class="chart-title">📊 Current Status</div>', unsafe_allow_html=True)
        
        # Render gauge
        gauge_chart = render_current_status_gauge(current_rate)
        st.plotly_chart(gauge_chart, use_container_width=True)
        
        # Quick stats
        st.markdown("""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #2E86AB;">📈 Quick Stats</h4>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Annual Growth</div>
                    <div style="font-size: 1.2rem; font-weight: bold;">4.8%</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #666;">Trend</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #4CAF50;">↑ Positive</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Scenario selector
    st.markdown("---")
    st.markdown("### 🎭 Scenario Analysis")
    
    # Render scenario cards
    render_scenario_cards()
    
    # Interactive scenario visualization
    st.markdown("---")
    st.markdown("### 📈 Scenario Trajectories")
    
    # Get scenario trajectories chart
    scenario_chart = render_scenario_trajectories()
    st.plotly_chart(scenario_chart, use_container_width=True)
    
    # Scenario insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🏆 Best Case Scenario:**
        - Achieves 60% target by 2026
        - Requires aggressive intervention
        - Dependent on economic stability
        """)
    
    with col2:
        st.warning("""
        **⚠️ Key Risks:**
        - Policy implementation delays
        - Economic headwinds
        - Infrastructure gaps
        """)
    
    # Required interventions
    st.markdown("---")
    st.markdown("### 🚀 Required Interventions to Reach 60%")
    
    # Get interventions data
    interventions, styled_interventions = render_required_interventions()
    
    # Display interventions with filtering
    col1, col2 = st.columns([3, 1])
    
    with col2:
        min_impact = st.slider(
            "Minimum Impact:",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5
        )
        
        cost_filter = st.multiselect(
            "Cost Level:",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )
    
    with col1:
        # Filter interventions
        filtered_interventions = interventions[
            (interventions['Potential Impact'] >= min_impact) &
            (interventions['Cost'].isin(cost_filter))
        ]
        
        st.dataframe(
            filtered_interventions.style.background_gradient(
                subset=['Potential Impact'], 
                cmap='Greens'
            ).format({
                'Potential Impact': '{:.1f}%'
            }),
            use_container_width=True,
            height=300
        )
    
    # Implementation roadmap
    st.markdown("#### 🗺️ Implementation Roadmap")
    
    roadmap_df = render_implementation_roadmap()
    
    # Create tabs for each phase
    tabs = st.tabs([f"Phase {i+1}" for i in range(len(roadmap_df))])
    
    for idx, (_, row) in enumerate(roadmap_df.iterrows()):
        with tabs[idx]:
            st.markdown(f"### {row['Phase']}")
            st.markdown(f"**Target Impact:** {row['Target Impact']}")
            st.markdown(f"**Key Activities:**")
            
            activities = row['Key Activities'].split(', ')
            for activity in activities:
                st.markdown(f"- {activity}")
            
            # Progress bar
            progress_value = 0.25 * (idx + 1)
            st.progress(progress_value)
            
            if idx == 0:
                st.caption("Completed: Foundation phase")
            elif idx == 1:
                st.caption("In Progress: Acceleration phase")
            else:
                st.caption("Planned: Future phase")
    
    # Summary
    st.markdown("---")
    st.markdown("### 📋 Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.success("""
        **✅ What's Working:**
        - Steady growth trajectory
        - Strong mobile money adoption
        - Government commitment
        """)
    
    with summary_col2:
        st.error("""
        **❌ Challenges:**
        - Below required growth rate
        - Regional disparities
        - Digital literacy gaps
        """)
    
    # Call to action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2E86AB 0%, #4ECDC4 100%); 
                color: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3 style="margin: 0 0 10px 0;">🚀 Next Steps</h3>
        <p style="margin: 0;">To reach the 60% target by 2027, Ethiopia needs to:</p>
        <ul style="margin: 10px 0;">
            <li>Accelerate digital infrastructure rollout</li>
            <li>Implement targeted financial literacy programs</li>
            <li>Enhance mobile money interoperability</li>
            <li>Focus on women and rural populations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()