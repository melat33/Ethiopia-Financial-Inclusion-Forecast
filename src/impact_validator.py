"""
Impact Validator - Validate models against historical data
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class ImpactValidator:
    """Validate event impact models against historical observations"""
    
    def __init__(self, observations: pd.DataFrame):
        self.observations = observations.copy()
    
    def validate_telebirr_impact(self) -> Dict:
        """Validate Telebirr launch impact against historical data"""
        # Get pre and post Telebirr data
        pre_telebirr = self.observations[
            (self.observations['indicator_code'] == 'ACC_MM_ACCOUNT') &
            (self.observations['observation_date'] < '2021-05-01')
        ]
        
        post_telebirr = self.observations[
            (self.observations['indicator_code'] == 'ACC_MM_ACCOUNT') &
            (self.observations['observation_date'] >= '2021-05-01')
        ]
        
        if len(pre_telebirr) > 0 and len(post_telebirr) > 0:
            pre_value = pre_telebirr.iloc[-1]['value_numeric']
            post_value = post_telebirr.iloc[-1]['value_numeric']
            actual_change = post_value - pre_value
            
            return {
                'event': 'Telebirr Launch',
                'actual_change': round(actual_change, 2),
                'pre_value': pre_value,
                'post_value': post_value,
                'validation_status': 'VALID' if 3 <= actual_change <= 8 else 'REVIEW',
                'confidence': 'high' if 3 <= actual_change <= 8 else 'medium'
            }
        
        return {'error': 'Insufficient data'}
    
    def compare_with_historical(self, event_date: str, indicator: str, 
                               window_months: int = 12) -> Dict:
        """Compare event impact with historical trend"""
        # Convert to datetime
        event_dt = pd.to_datetime(event_date)
        
        # Get pre and post periods
        pre_start = event_dt - pd.DateOffset(months=window_months)
        post_end = event_dt + pd.DateOffset(months=window_months)
        
        pre_data = self.observations[
            (self.observations['indicator_code'] == indicator) &
            (self.observations['observation_date'] >= pre_start) &
            (self.observations['observation_date'] < event_dt)
        ]
        
        post_data = self.observations[
            (self.observations['indicator_code'] == indicator) &
            (self.observations['observation_date'] > event_dt) &
            (self.observations['observation_date'] <= post_end)
        ]
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_avg = pre_data['value_numeric'].mean()
            post_avg = post_data['value_numeric'].mean()
            change = post_avg - pre_avg
            
            return {
                'pre_period_avg': pre_avg,
                'post_period_avg': post_avg,
                'actual_change': change,
                'data_points': {'pre': len(pre_data), 'post': len(post_data)}
            }
        
        return {'error': 'Insufficient data for comparison'}