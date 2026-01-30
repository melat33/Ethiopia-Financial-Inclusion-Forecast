"""
Event Impact Modeling Module
Models the impact of events on financial inclusion indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import warnings

class EventImpactModeler:
    """Models and quantifies event impacts on financial indicators"""
    
    def __init__(self, data_path: str):
        """
        Initialize the event impact modeler
        
        Args:
            data_path: Path to enriched dataset
        """
        self.df = pd.read_csv(data_path)
        self.events = self.df[self.df['record_type'] == 'event'].copy()
        self.impact_links = self.df[self.df['record_type'] == 'impact_link'].copy()
        self.observations = self.df[self.df['record_type'] == 'observation'].copy()
        
        # Convert dates
        for col in ['observation_date', 'event_date', 'collection_date']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        self.impact_matrix = None
        self.event_effects = {}
        
    def build_impact_matrix(self) -> pd.DataFrame:
        """
        Build event-indicator impact matrix
        
        Returns:
            DataFrame with events as rows, indicators as columns
        """
        # Filter only relevant events with impact links
        events_with_impact = self.impact_links['parent_id'].unique()
        relevant_events = self.events[self.events['record_id'].isin(events_with_impact)].copy()
        
        # Get all unique indicators
        indicators = sorted(self.observations['indicator_code'].dropna().unique())
        
        # Initialize impact matrix
        impact_matrix = pd.DataFrame(
            index=relevant_events['record_id'],
            columns=indicators,
            dtype=object
        )
        
        # Fill with event details
        event_details = {}
        for _, event in relevant_events.iterrows():
            event_id = event['record_id']
            event_details[event_id] = {
                'event_name': event.get('value_text', ''),
                'event_date': event.get('event_date'),
                'category': event.get('category', ''),
                'confidence': event.get('confidence', 'medium')
            }
        
        # Populate impact values from impact links
        for _, link in self.impact_links.iterrows():
            event_id = link['parent_id']
            indicator = link['related_indicator']
            
            if event_id in impact_matrix.index and indicator in impact_matrix.columns:
                impact = {
                    'direction': link.get('impact_direction', 'positive'),
                    'magnitude': link.get('impact_magnitude', 'medium'),
                    'estimate': link.get('impact_estimate'),
                    'lag_months': link.get('lag_months', 0),
                    'confidence': link.get('confidence', 'medium'),
                    'evidence': link.get('evidence_basis', '')
                }
                impact_matrix.loc[event_id, indicator] = impact
        
        # Add event metadata
        impact_matrix['event_name'] = impact_matrix.index.map(
            lambda x: event_details.get(x, {}).get('event_name', '')
        )
        impact_matrix['event_date'] = impact_matrix.index.map(
            lambda x: event_details.get(x, {}).get('event_date')
        )
        impact_matrix['category'] = impact_matrix.index.map(
            lambda x: event_details.get(x, {}).get('category', '')
        )
        impact_matrix['confidence'] = impact_matrix.index.map(
            lambda x: event_details.get(x, {}).get('confidence', 'medium')
        )
        
        self.impact_matrix = impact_matrix
        return impact_matrix
    
    def quantify_impacts(self) -> Dict:
        """
        Quantify impact estimates with numerical values
        
        Returns:
            Dictionary with quantified impacts
        """
        if self.impact_matrix is None:
            self.build_impact_matrix()
        
        # Define quantification rules
        magnitude_map = {
            'very_low': 0.5,
            'low': 1.0,
            'medium': 2.0,
            'high': 4.0,
            'very_high': 6.0
        }
        
        direction_map = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
        
        quantified_impacts = {}
        
        for event_id in self.impact_matrix.index:
            event_impacts = {}
            
            for indicator in self.impact_matrix.columns:
                if indicator in ['event_name', 'event_date', 'category', 'confidence']:
                    continue
                    
                impact = self.impact_matrix.loc[event_id, indicator]
                if isinstance(impact, dict):
                    # Calculate numerical impact
                    magnitude = magnitude_map.get(
                        impact.get('magnitude', 'medium'), 
                        2.0
                    )
                    direction = direction_map.get(
                        impact.get('direction', 'positive'),
                        1
                    )
                    
                    if impact.get('estimate') is not None:
                        try:
                            value = float(impact['estimate'])
                        except:
                            value = magnitude * direction
                    else:
                        value = magnitude * direction
                    
                    event_impacts[indicator] = {
                        'value': value,
                        'lag_months': impact.get('lag_months', 0),
                        'confidence': impact.get('confidence', 'medium'),
                        'evidence': impact.get('evidence', '')
                    }
            
            if event_impacts:
                quantified_impacts[event_id] = event_impacts
        
        self.event_effects = quantified_impacts
        return quantified_impacts
    
    def model_event_effects(self, base_values: Dict, target_date: datetime) -> Dict:
        """
        Model cumulative effects of events on indicators
        
        Args:
            base_values: Dictionary of baseline indicator values
            target_date: Date to project to
            
        Returns:
            Dictionary with projected values
        """
        if not self.event_effects:
            self.quantify_impacts()
        
        projections = base_values.copy()
        event_contributions = {}
        
        for event_id, impacts in self.event_effects.items():
            event_date = pd.to_datetime(
                self.impact_matrix.loc[event_id, 'event_date']
            )
            
            # Check if event occurred before target date
            if event_date <= target_date:
                months_elapsed = (target_date.year - event_date.year) * 12 + \
                               (target_date.month - event_date.month)
                
                for indicator, impact in impacts.items():
                    if indicator in base_values:
                        lag_months = impact.get('lag_months', 0)
                        
                        # Apply effect if lag period has passed
                        if months_elapsed >= lag_months:
                            effect_size = impact['value']
                            
                            # Adjust for time (effects often build gradually)
                            if months_elapsed - lag_months < 12:
                                # First year: effect builds
                                time_factor = (months_elapsed - lag_months + 1) / 12
                                effect_size = effect_size * min(time_factor, 1.0)
                            
                            # Apply to projection
                            projections[indicator] = projections.get(indicator, 0) + effect_size
                            
                            # Track contribution
                            if indicator not in event_contributions:
                                event_contributions[indicator] = {}
                            event_contributions[indicator][event_id] = effect_size
        
        return {
            'projections': projections,
            'contributions': event_contributions,
            'target_date': target_date
        }
    
    def generate_scenarios(self, base_values: Dict, target_year: int) -> Dict:
        """
        Generate optimistic, base, and pessimistic scenarios
        
        Args:
            base_values: Baseline indicator values
            target_year: Year to project to
            
        Returns:
            Dictionary with scenarios
        """
        target_date = datetime(target_year, 12, 31)
        
        # Base scenario
        base_result = self.model_event_effects(base_values, target_date)
        
        # Optimistic scenario (25% higher impact)
        optimistic_effects = {}
        for event_id, impacts in self.event_effects.items():
            optimistic_effects[event_id] = {}
            for indicator, impact in impacts.items():
                optimistic_impact = impact.copy()
                optimistic_impact['value'] = impact['value'] * 1.25
                optimistic_effects[event_id][indicator] = optimistic_impact
        
        # Temporarily replace event effects
        original_effects = self.event_effects.copy()
        self.event_effects = optimistic_effects
        optimistic_result = self.model_event_effects(base_values, target_date)
        self.event_effects = original_effects
        
        # Pessimistic scenario (25% lower impact)
        pessimistic_effects = {}
        for event_id, impacts in self.event_effects.items():
            pessimistic_effects[event_id] = {}
            for indicator, impact in impacts.items():
                pessimistic_impact = impact.copy()
                pessimistic_impact['value'] = impact['value'] * 0.75
                pessimistic_effects[event_id][indicator] = pessimistic_impact
        
        self.event_effects = pessimistic_effects
        pessimistic_result = self.model_event_effects(base_values, target_date)
        self.event_effects = original_effects
        
        return {
            'optimistic': optimistic_result['projections'],
            'base': base_result['projections'],
            'pessimistic': pessimistic_result['projections'],
            'contributions': base_result['contributions']
        }
    
    def save_results(self, output_dir: str = 'models/'):
        """Save impact modeling results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save impact matrix
        if self.impact_matrix is not None:
            impact_matrix_path = os.path.join(output_dir, 'event_impact_matrix.csv')
            self.impact_matrix.to_csv(impact_matrix_path)
        
        # Save quantified impacts
        if self.event_effects:
            impacts_path = os.path.join(output_dir, 'quantified_impacts.json')
            with open(impacts_path, 'w') as f:
                json.dump(self.event_effects, f, indent=2, default=str)
        
        return {
            'impact_matrix': impact_matrix_path if self.impact_matrix is not None else None,
            'quantified_impacts': impacts_path if self.event_effects else None
        }