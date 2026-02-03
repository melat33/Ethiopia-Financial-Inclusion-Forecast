"""
Utility Functions
"""

import streamlit as st
import pandas as pd
import json
import os

@st.cache_data
def load_data():
    """Load forecast data"""
    try:
        # Try to load forecasts
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
        st.warning(f"Could not load forecast data: {e}")
        return None

def format_percentage(value, decimals=1):
    """Format value as percentage string"""
    return f"{value:.{decimals}f}%"

def get_sample_forecasts():
    """Get sample forecast data"""
    return pd.DataFrame({
        'year': [2025, 2026, 2027, 2025, 2026, 2027],
        'indicator': ['ACC_OWNERSHIP', 'ACC_OWNERSHIP', 'ACC_OWNERSHIP', 
                     'USG_DIGITAL_PAYMENT', 'USG_DIGITAL_PAYMENT', 'USG_DIGITAL_PAYMENT'],
        'forecast_value': [54.2, 56.9, 59.7, 36.4, 38.8, 41.2],
        'model_type': ['ensemble', 'ensemble', 'ensemble', 'ensemble', 'ensemble', 'ensemble']
    })

def get_historical_data():
    """Get historical data"""
    return pd.DataFrame({
        'year': [2011, 2014, 2017, 2021, 2024],
        'Account Ownership': [14.0, 22.0, 35.0, 46.0, 49.0],
        'Digital Payments': [10.0, 18.0, 25.0, 35.0, 35.0]
    })

def get_channel_data():
    """Get channel comparison data"""
    return pd.DataFrame({
        'Year': [2014, 2017, 2021, 2024],
        'Bank Accounts': [18, 28, 38, 42],
        'Mobile Money': [2, 5, 8, 12],
        'Microfinance': [2, 2, 3, 4]
    })