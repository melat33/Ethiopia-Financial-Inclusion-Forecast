"""
Event impact modeling with reference code integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class EnhancedEventImpactModeler:
    def __init__(self, df: pd.DataFrame, ref_codes: pd.DataFrame = None):
        self.df = df
        self.ref_codes = ref_codes
        self.events = df[df['record_type'] == 'event'].copy()
        self.impact_links = df[df['record_type'] == 'impact_link'].copy()
        self.observations = df[df['record_type'] == 'observation'].copy()
    
    def quantify_impacts_with_reference_codes(self) -> Dict:
        """Quantify impacts using reference-based methods"""
        impacts = {}
        
        for _, link in self.impact_links.iterrows():
            event_id = link['parent_id']
            event = self.events[self.events['record_id'] == event_id]
            
            if not event.empty:
                magnitude = self._calculate_impact_magnitude(link, event.iloc[0])
                impacts[link['record_id']] = {
                    'event_id': event_id,
                    'event_name': event.iloc[0]['value_text'],
                    'indicator': link['related_indicator'],
                    'magnitude': magnitude,
                    'direction': link.get('impact_direction', 'positive'),
                    'lag_months': link.get('lag_months', 0),
                    'confidence': link.get('confidence', 'medium')
                }
        
        return impacts
    
    def _calculate_impact_magnitude(self, link: pd.Series, event: pd.Series) -> float:
        """Calculate impact magnitude using multiple methods"""
        # Method 1: Direct estimate if available
        if pd.notna(link.get('impact_estimate')):
            return float(link['impact_estimate'])
        
        # Method 2: Category-based from reference codes
        if self.ref_codes is not None and 'category' in event:
            return self._get_category_based_magnitude(event['category'])
        
        # Method 3: Default based on magnitude category
        magnitude_map = {
            'very_low': 0.5,
            'low': 1.0,
            'medium': 2.0,
            'high': 4.0,
            'very_high': 6.0
        }
        
        return magnitude_map.get(link.get('impact_magnitude', 'medium'), 2.0)
    
    def _get_category_based_magnitude(self, category: str) -> float:
        """Get magnitude based on reference code category"""
        category_mapping = {
            'policy': 2.0,
            'product_launch': 4.0,
            'infrastructure': 3.0,
            'market_entry': 3.5,
            'milestone': 1.5
        }
        
        return category_mapping.get(category, 2.0)
    
    def create_association_matrix(self) -> pd.DataFrame:
        """Create event-indicator association matrix"""
        impacts = self.quantify_impacts_with_reference_codes()
        
        # Create matrix structure
        events = {}
        for impact in impacts.values():
            event_id = impact['event_id']
            if event_id not in events:
                event_data = self.events[self.events['record_id'] == event_id].iloc[0]
                events[event_id] = {
                    'event_name': event_data['value_text'],
                    'category': event_data.get('category', 'unknown'),
                    'event_date': event_data.get('event_date', 'unknown')
                }
        
        # Get unique indicators
        indicators = sorted(set(impact['indicator'] for impact in impacts.values()))
        
        # Build matrix
        matrix_data = []
        for event_id, event_info in events.items():
            row = {'event_id': event_id, **event_info}
            
            # Initialize all indicators with 0
            for indicator in indicators:
                row[indicator] = 0.0
            
            # Fill in impacts
            event_impacts = [i for i in impacts.values() if i['event_id'] == event_id]
            for impact in event_impacts:
                row[impact['indicator']] = impact['magnitude'] if impact['direction'] == 'positive' else -impact['magnitude']
            
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data).set_index('event_id')
    
    def get_baseline_values(self, year: int = 2024) -> Dict:
        """Get baseline values for key indicators"""
        baseline = {}
        
        key_indicators = ['ACC_OWNERSHIP', 'ACC_MM_ACCOUNT', 'USG_DIGITAL_PAYMENT']
        
        for indicator in key_indicators:
            obs = self.observations[self.observations['indicator_code'] == indicator]
            if not obs.empty:
                latest = obs.sort_values('observation_date').iloc[-1]
                baseline[indicator] = float(latest['value_numeric'])
        
        return baseline