"""
Download Page
"""

import streamlit as st
import pandas as pd
from components.utils import load_data

def render_download_page(data):
    """Render download page"""
    st.markdown("## 📥 Data Export")
    
    # Available datasets
    st.markdown("### Available Datasets")
    
    datasets = [
        {"name": "Forecasts 2025-2027", "description": "Complete forecast data for 2025-2027"},
        {"name": "Scenario Analysis", "description": "Optimistic, Baseline, Pessimistic scenarios"},
        {"name": "Target Gap Analysis", "description": "NFIS-II target gap calculations"},
        {"name": "Historical Trends", "description": "Historical financial inclusion data"}
    ]
    
    for dataset in datasets:
        with st.expander(f"📁 {dataset['name']}"):
            st.write(dataset['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Preview {dataset['name']}", key=f"preview_{dataset['name']}"):
                    if dataset['name'] == "Forecasts 2025-2027" and data and 'forecasts' in data:
                        st.dataframe(data['forecasts'].head(), use_container_width=True)
                    else:
                        st.info("Sample data preview would be shown here")
            
            with col2:
                if st.button(f"Download {dataset['name']}", key=f"download_{dataset['name']}"):
                    if dataset['name'] == "Forecasts 2025-2027" and data and 'forecasts' in data:
                        csv = data['forecasts'].to_csv(index=False)
                        st.download_button(
                            label="Click to download CSV",
                            data=csv,
                            file_name="forecasts_2025_2027.csv",
                            mime="text/csv",
                            key=f"dl_{dataset['name']}"
                        )
                    else:
                        # Generate sample data for other datasets
                        sample_data = pd.DataFrame({
                            'Year': [2025, 2026, 2027],
                            'Value': [50.0, 55.0, 60.0],
                            'Scenario': ['Baseline', 'Baseline', 'Baseline']
                        })
                        csv = sample_data.to_csv(index=False)
                        st.download_button(
                            label="Download Sample CSV",
                            data=csv,
                            file_name=f"{dataset['name'].lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"sample_{dataset['name']}"
                        )
    
    # Custom export
    st.markdown("---")
    st.markdown("### 🔧 Custom Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        indicators = st.multiselect(
            "Select Indicators:",
            ["Account Ownership", "Digital Payments", "Mobile Money"],
            default=["Account Ownership", "Digital Payments"]
        )
    
    with col2:
        years = st.multiselect(
            "Select Years:",
            ["2024", "2025", "2026", "2027"],
            default=["2025", "2026", "2027"]
        )
    
    export_format = st.radio(
        "Export Format:",
        ["CSV", "JSON"],
        horizontal=True
    )
    
    if st.button("🚀 Generate Custom Export", type="primary"):
        with st.spinner("Generating export..."):
            # Create export data
            export_data = pd.DataFrame({
                'Year': years,
                'Account_Ownership': [49.0, 54.2, 56.9, 59.7][:len(years)],
                'Digital_Payments': [35.0, 36.4, 38.8, 41.2][:len(years)]
            })
            
            if export_format == "CSV":
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="custom_export.csv",
                    mime="text/csv"
                )
            else:
                json_data = export_data.to_json(orient='records')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="custom_export.json",
                    mime="application/json"
                )
            
            st.success("✅ Export generated successfully!")