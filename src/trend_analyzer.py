"""Trend analysis for financial inclusion indicators"""
import pandas as pd
import numpy as np
from datetime import datetime

class TrendAnalyzer:
    def __init__(self, df):
        self.df = df
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for trend analysis"""
        self.df['obs_date'] = pd.to_datetime(self.df['observation_date'])
        self.df['year'] = self.df['obs_date'].dt.year
    
    def analyze_indicator_trend(self, indicator_code):
        """Analyze trend for specific indicator"""
        indicator_data = self.df[self.df['indicator_code'] == indicator_code]
        return self._calculate_trend_metrics(indicator_data)
    
    def _calculate_trend_metrics(self, data):
        """Calculate trend metrics for data"""
        if data.empty:
            return {}
        
        # Group by year
        yearly = data.groupby('year')['value_numeric'].agg(['mean', 'count']).reset_index()
        
        # Calculate growth rates
        metrics = {
            'period': f"{yearly['year'].min()}-{yearly['year'].max()}",
            'data_points': len(data),
            'years_covered': len(yearly),
            'trend': self._assess_trend_direction(yearly),
            'latest_value': yearly.iloc[-1]['mean'] if len(yearly) > 0 else None
        }
        
        return metrics
    
    def _assess_trend_direction(self, yearly_data):
        """Assess if trend is increasing, decreasing, or stable"""
        if len(yearly_data) < 2:
            return "Insufficient data"
        
        # Simple trend assessment
        first = yearly_data.iloc[0]['mean']
        last = yearly_data.iloc[-1]['mean']
        
        if last > first + 2:  # More than 2 percentage points increase
            return "Increasing"
        elif last < first - 2:
            return "Decreasing"
        else:
            return "Stable"