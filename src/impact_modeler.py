"""
Event Impact Modeler - Core modeling engine for Ethiopia Financial Inclusion
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

class EventImpactModeler:
    """Models and quantifies event impacts on financial inclusion indicators"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize modeler with enriched dataset
        
        Args:
            df: Enriched dataset from Task 1
        """
        self.df = df.copy()
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and separate data components"""
        # Separate record types
        self.events = self.df[self.df['record_type'] == 'event'].copy()
        self.impact_links = self.df[self.df['record_type'] == 'impact_link'].copy()
        self.observations = self.df[self.df['record_type'] == 'observation'].copy()
        
        # Convert dates
        for col in ['event_date', 'observation_date', 'collection_date']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        print(f"✅ Data prepared: {len(self.events)} events, {len(self.impact_links)} impact links")
        
    def build_event_indicator_matrix(self) -> pd.DataFrame:
        """
        Build comprehensive event-indicator matrix
        
        Returns:
            DataFrame with events as rows, indicators as columns
        """
        # Find event name column
        event_name_col = None
        for col in ['value_text', 'event_name', 'name', 'description']:
            if col in self.events.columns and not self.events[col].isna().all():
                event_name_col = col
                break
        
        if not event_name_col:
            event_name_col = 'record_id'
            self.events['record_id_display'] = self.events['record_id']
            event_name_col = 'record_id_display'
        
        # Select available columns
        available_cols = []
        for col in ['record_id', event_name_col, 'event_date', 'category']:
            if col in self.events.columns:
                available_cols.append(col)
        
        # Merge impact links with events
        merged = pd.merge(
            self.impact_links,
            self.events[available_cols],
            left_on='parent_id',
            right_on='record_id',
            how='left',
            suffixes=('_impact', '_event')
        )
        
        # Create simple matrix using pivot
        matrix_data = []
        
        for _, row in merged.iterrows():
            event_id = row['parent_id']
            event_name = row[event_name_col] if event_name_col in row else event_id
            event_date = row.get('event_date')
            category = row.get('category')
            indicator = row['related_indicator']
            impact_estimate = row.get('impact_estimate')
            impact_direction = row.get('impact_direction', 'positive')
            lag_months = row.get('lag_months', 0)
            confidence = row.get('confidence', 'medium')
            
            matrix_data.append({
                'event_id': event_id,
                'event_name': event_name,
                'event_date': event_date,
                'category': category,
                'indicator': indicator,
                'impact_estimate': impact_estimate,
                'impact_direction': impact_direction,
                'lag_months': lag_months,
                'confidence': confidence
            })
        
        matrix_df = pd.DataFrame(matrix_data)
        
        # Create pivot matrix
        pivot_matrix = matrix_df.pivot_table(
            index=['event_id', 'event_name', 'event_date', 'category'],
            columns='indicator',
            values='impact_estimate',
            aggfunc='first'
        ).reset_index()
        
        # Add direction, lag, and confidence as separate matrices
        for metric in ['impact_direction', 'lag_months', 'confidence']:
            metric_pivot = matrix_df.pivot_table(
                index=['event_id', 'event_name', 'event_date', 'category'],
                columns='indicator',
                values=metric,
                aggfunc='first'
            )
            
            # Add metric as suffix to column names
            if not metric_pivot.empty:
                metric_pivot.columns = [f"{col}_{metric}" for col in metric_pivot.columns]
                pivot_matrix = pd.merge(
                    pivot_matrix,
                    metric_pivot.reset_index(),
                    on=['event_id', 'event_name', 'event_date', 'category'],
                    how='left'
                )
        
        self.event_indicator_matrix = pivot_matrix
        print(f"✅ Event-Indicator Matrix created: {pivot_matrix.shape}")
        
        return pivot_matrix
    
    def quantify_impacts(self) -> Dict[str, Dict[str, Dict]]:
        """
        Quantify impacts with numerical values
        
        Returns:
            Dictionary with quantified impacts {event_id: {indicator: impact_data}}
        """
        quantified = {}
        
        for _, link in self.impact_links.iterrows():
            event_id = link['parent_id']
            indicator = link['related_indicator']
            
            if pd.isna(event_id) or pd.isna(indicator):
                continue
            
            # Get impact estimate
            if 'impact_estimate' in link and pd.notna(link['impact_estimate']):
                try:
                    value = float(link['impact_estimate'])
                except:
                    value = 0
            else:
                # Estimate based on magnitude
                magnitude_map = {
                    'very_low': 0.5, 'low': 1.0, 'medium': 2.0,
                    'high': 4.0, 'very_high': 6.0, 'small': 1.0,
                    'medium': 2.0, 'large': 4.0
                }
                magnitude = link.get('impact_magnitude', 'medium')
                direction = 1 if link.get('impact_direction') == 'positive' else -1
                value = magnitude_map.get(magnitude, 2.0) * direction
            
            if event_id not in quantified:
                quantified[event_id] = {}
            
            quantified[event_id][indicator] = {
                'value': value,
                'lag_months': link.get('lag_months', 0),
                'direction': link.get('impact_direction', 'positive'),
                'magnitude': link.get('impact_magnitude', 'medium'),
                'confidence': link.get('confidence', 'medium'),
                'evidence': link.get('evidence_basis', '')
            }
        
        self.quantified_impacts = quantified
        print(f"✅ Quantified impacts for {len(quantified)} events")
        
        return quantified
    
    def create_association_matrix(self) -> pd.DataFrame:
        """
        Create numerical association matrix
        
        Returns:
            DataFrame with numerical impact values
        """
        if not hasattr(self, 'quantified_impacts'):
            self.quantify_impacts()
        
        # Get all unique indicators
        all_indicators = set()
        for impacts in self.quantified_impacts.values():
            all_indicators.update(impacts.keys())
        
        all_indicators = sorted(list(all_indicators))
        
        # Create matrix
        association_matrix = pd.DataFrame(
            index=list(self.quantified_impacts.keys()),
            columns=all_indicators,
            dtype=float
        )
        
        # Fill with values
        for event_id, impacts in self.quantified_impacts.items():
            for indicator, data in impacts.items():
                association_matrix.loc[event_id, indicator] = data['value']
        
        # Add event metadata
        event_info = {}
        for _, event in self.events.iterrows():
            event_id = event['record_id']
            
            # Find event name
            event_name = event.get('value_text')
            if pd.isna(event_name):
                event_name = event.get('event_name', f"Event {event_id}")
            if pd.isna(event_name):
                event_name = f"Event {event_id}"
            
            event_info[event_id] = {
                'event_name': event_name,
                'event_date': event.get('event_date'),
                'category': event.get('category', '')
            }
        
        association_matrix['event_name'] = association_matrix.index.map(
            lambda x: event_info.get(x, {}).get('event_name', str(x)))
        association_matrix['event_date'] = association_matrix.index.map(
            lambda x: event_info.get(x, {}).get('event_date'))
        association_matrix['category'] = association_matrix.index.map(
            lambda x: event_info.get(x, {}).get('category', ''))
        
        # Reorder columns to have metadata first
        cols = ['event_name', 'event_date', 'category'] + [c for c in association_matrix.columns 
                                                          if c not in ['event_name', 'event_date', 'category']]
        association_matrix = association_matrix[cols]
        
        self.association_matrix = association_matrix
        print(f"✅ Association matrix created: {association_matrix.shape}")
        
        return association_matrix
    
    def model_event_effects(self, base_year: int = 2024, target_year: int = 2027) -> Dict:
        """
        Model cumulative effects of events
        
        Args:
            base_year: Baseline year
            target_year: Target year for projection
            
        Returns:
            Dictionary with projections
        """
        if not hasattr(self, 'quantified_impacts'):
            self.quantify_impacts()
        
        # Get baseline values
        baseline = self._get_baseline_values(base_year)
        
        # Calculate cumulative impacts
        projections = baseline.copy()
        contributions = {}
        
        for event_id, impacts in self.quantified_impacts.items():
            # Get event date
            event_date = None
            event_row = self.events[self.events['record_id'] == event_id]
            if not event_row.empty:
                event_date = event_row.iloc[0].get('event_date')
            
            if pd.isna(event_date):
                continue
            
            try:
                # Parse event date
                if isinstance(event_date, str):
                    event_dt = pd.to_datetime(event_date, errors='coerce')
                else:
                    event_dt = event_date
                
                if pd.isna(event_dt):
                    continue
                
                event_year = event_dt.year
                
                # Check if event occurred before target year
                if event_year < target_year:
                    for indicator, impact in impacts.items():
                        if indicator in projections:
                            # Apply impact with time adjustment
                            years_since = target_year - event_year
                            months_since = years_since * 12
                            lag_months = impact.get('lag_months', 0)
                            
                            if months_since >= lag_months:
                                effect = impact['value']
                                
                                # Gradual effect build-up (if within 12 months of lag period)
                                if months_since - lag_months < 12:
                                    build_factor = (months_since - lag_months) / 12
                                    effect = effect * build_factor
                                
                                # Apply effect
                                current_value = projections.get(indicator, 0)
                                projections[indicator] = current_value + effect
                                
                                # Track contribution
                                if indicator not in contributions:
                                    contributions[indicator] = {}
                                contributions[indicator][event_id] = {
                                    'effect': effect,
                                    'event_name': event_row.iloc[0].get('value_text', event_id),
                                    'event_year': event_year
                                }
            except Exception as e:
                print(f"Warning: Could not process event {event_id}: {e}")
                continue
        
        return {
            'projections': projections,
            'contributions': contributions,
            'baseline': baseline,
            'target_year': target_year
        }
    
    def _get_baseline_values(self, year: int) -> Dict[str, float]:
        """Get baseline values for given year"""
        # Default baseline values (2024)
        baseline = {
            'ACC_OWNERSHIP': 49.0,      # 2024 Findex
            'ACC_MM_ACCOUNT': 9.45,     # 2024 Findex
            'USG_DIGITAL_PAYMENT': 35.0,# 2024 estimate
            'INF_AGENT_DENSITY': 25.0,  # 2024 estimate
            'ACC_MALE': 56.0,           # 2024 Findex
            'ACC_FEMALE': 42.0,         # 2024 Findex
            'GEN_GAP_ACC': 14.0         # 2024 Findex
        }
        
        # Try to get actual data for the year
        if year == 2021:
            baseline['ACC_OWNERSHIP'] = 46.0
            baseline['ACC_MM_ACCOUNT'] = 4.7
            baseline['USG_DIGITAL_PAYMENT'] = 22.0
        elif year == 2017:
            baseline['ACC_OWNERSHIP'] = 35.0
        elif year == 2014:
            baseline['ACC_OWNERSHIP'] = 22.0
        elif year == 2011:
            baseline['ACC_OWNERSHIP'] = 14.0
        
        return baseline
    
    def generate_scenarios(self, base_year: int = 2024) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Generate optimistic, base, and pessimistic scenarios"""
        scenarios = {}
        
        for year in [2025, 2026, 2027]:
            year_scenarios = {}
            
            # Base scenario
            base_result = self.model_event_effects(base_year, year)
            
            # Create adjusted impact dictionaries
            original_impacts = self.quantified_impacts.copy()
            
            # Optimistic scenario (30% higher impacts)
            optimistic_impacts = {}
            for event_id, impacts in original_impacts.items():
                optimistic_impacts[event_id] = {}
                for indicator, data in impacts.items():
                    optimistic_impacts[event_id][indicator] = data.copy()
                    optimistic_impacts[event_id][indicator]['value'] = data['value'] * 1.3
            
            # Temporarily replace and calculate
            self.quantified_impacts = optimistic_impacts
            optimistic_result = self.model_event_effects(base_year, year)
            
            # Pessimistic scenario (30% lower impacts)
            pessimistic_impacts = {}
            for event_id, impacts in original_impacts.items():
                pessimistic_impacts[event_id] = {}
                for indicator, data in impacts.items():
                    pessimistic_impacts[event_id][indicator] = data.copy()
                    pessimistic_impacts[event_id][indicator]['value'] = data['value'] * 0.7
            
            self.quantified_impacts = pessimistic_impacts
            pessimistic_result = self.model_event_effects(base_year, year)
            
            # Restore original
            self.quantified_impacts = original_impacts
            
            year_scenarios = {
                'optimistic': optimistic_result['projections'],
                'base': base_result['projections'],
                'pessimistic': pessimistic_result['projections']
            }
            
            scenarios[year] = year_scenarios
        
        return scenarios
    
    def save_matrices(self, output_dir: str = '../models'):
        """Save all matrices to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        matrices = {}
        
        # Save event indicator matrix
        if hasattr(self, 'event_indicator_matrix'):
            path = f'{output_dir}/event_indicator_matrix.csv'
            self.event_indicator_matrix.to_csv(path, index=False, encoding='utf-8')
            matrices['event_indicator_matrix'] = path
        
        # Save association matrix
        if hasattr(self, 'association_matrix'):
            path = f'{output_dir}/association_matrix.csv'
            self.association_matrix.to_csv(path, index=False, encoding='utf-8')
            matrices['association_matrix'] = path
        
        # Save quantified impacts
        if hasattr(self, 'quantified_impacts'):
            # Convert to DataFrame
            impact_data = []
            for event_id, impacts in self.quantified_impacts.items():
                for indicator, data in impacts.items():
                    impact_data.append({
                        'event_id': event_id,
                        'indicator': indicator,
                        'impact_value': data['value'],
                        'lag_months': data['lag_months'],
                        'direction': data['direction'],
                        'confidence': data['confidence']
                    })
            
            impacts_df = pd.DataFrame(impact_data)
            path = f'{output_dir}/quantified_impacts.csv'
            impacts_df.to_csv(path, index=False, encoding='utf-8')
            matrices['quantified_impacts'] = path
        
        # Generate and save scenarios
        scenarios = self.generate_scenarios()
        scenario_data = []
        for year, year_scenarios in scenarios.items():
            for scenario_name, values in year_scenarios.items():
                for indicator, value in values.items():
                    scenario_data.append({
                        'year': year,
                        'scenario': scenario_name,
                        'indicator': indicator,
                        'value': value
                    })
        
        scenarios_df = pd.DataFrame(scenario_data)
        path = f'{output_dir}/scenario_projections.csv'
        scenarios_df.to_csv(path, index=False, encoding='utf-8')
        matrices['scenario_projections'] = path
        
        return matrices