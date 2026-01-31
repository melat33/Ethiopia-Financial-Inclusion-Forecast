"""
Scenario Generation Module for Ethiopia Financial Inclusion
Generates forecast scenarios with event impacts
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json


class ScenarioGenerator:
    """
    Generates forecast scenarios for financial inclusion indicators.
    
    This class handles:
    1. Baseline trend forecasting
    2. Event-impact augmented forecasting
    3. Scenario generation (optimistic/pessimistic/baseline)
    4. Uncertainty quantification
    """
    
    def __init__(self, baseline_year: int = 2024, forecast_years: List[int] = None):
        """
        Initialize the scenario generator.
        
        Args:
            baseline_year: Year to use as baseline for forecasting
            forecast_years: Years to generate forecasts for
        """
        self.baseline_year = baseline_year
        
        if forecast_years is None:
            self.forecast_years = [2025, 2026, 2027]
        else:
            self.forecast_years = forecast_years
        
        # Default trend growth rates (can be overridden)
        self.trend_growth_rates = self._get_default_growth_rates()
        
        # Scenario multipliers
        self.scenario_multipliers = {
            'pessimistic': 0.7,
            'baseline': 1.0,
            'optimistic': 1.3
        }
    
    def _get_default_growth_rates(self) -> Dict[str, float]:
        """Get default trend growth rates for indicators."""
        return {
            'ACC_OWNERSHIP': 1.0,       # Slowing growth: 1.0pp/year
            'ACC_MM_ACCOUNT': 1.5,      # Accelerating: 1.5pp/year
            'USG_DIGITAL_PAYMENT': 2.0, # Fastest growing: 2.0pp/year
            'USG_P2P_COUNT': 25.0,      # Rapid transaction growth
            'GEN_GAP_ACC': -0.5,        # Gradually closing gap
            'INF_AGENT_DENSITY': 10.0,  # Infrastructure expansion
            'USG_ACTIVE_RATE': 2.0      # Improving activation
        }
    
    def set_baseline_values(self, observations: pd.DataFrame) -> Dict[str, float]:
        """
        Set baseline values from historical observations.
        
        Args:
            observations: DataFrame with observation data
            
        Returns:
            Dictionary of baseline values
        """
        baseline_values = {}
        
        for indicator in self.trend_growth_rates.keys():
            indicator_data = observations[
                observations['indicator_code'] == indicator
            ]
            
            if not indicator_data.empty:
                # Get most recent observation
                latest_obs = indicator_data.sort_values('observation_date').iloc[-1]
                baseline_values[indicator] = latest_obs['value_numeric']
            else:
                # Use reasonable defaults if no data
                baseline_values[indicator] = self._get_default_baseline(indicator)
        
        return baseline_values
    
    def _get_default_baseline(self, indicator: str) -> float:
        """Get default baseline value for an indicator."""
        defaults = {
            'ACC_OWNERSHIP': 49.0,      # 2024 Findex
            'ACC_MM_ACCOUNT': 9.45,     # 2024 Findex
            'USG_DIGITAL_PAYMENT': 35.0, # Estimated
            'USG_P2P_COUNT': 128.3,     # Million transactions
            'GEN_GAP_ACC': 18.0,        # Percentage points
            'INF_AGENT_DENSITY': 15.0,  # Agents per 10k adults
            'USG_ACTIVE_RATE': 58.0     # Percentage
        }
        return defaults.get(indicator, 0.0)
    
    def generate_trend_scenarios(self, baseline_values: Dict[str, float],
                                association_matrix: pd.DataFrame = None) -> Dict:
        """
        Generate trend-based forecast scenarios.
        
        Args:
            baseline_values: Dictionary of baseline values
            association_matrix: Optional association matrix for event impacts
            
        Returns:
            Nested dictionary of scenarios by year and scenario type
        """
        scenarios = {}
        
        for year in self.forecast_years:
            year_scenarios = {}
            
            for scenario_name, multiplier in self.scenario_multipliers.items():
                values = {}
                
                for indicator, baseline in baseline_values.items():
                    # Calculate trend component
                    trend = self._calculate_trend_component(
                        indicator, baseline, year, multiplier
                    )
                    
                    # Add event impacts if association matrix provided
                    event_impact = 0.0
                    if association_matrix is not None:
                        event_impact = self._calculate_event_impact(
                            indicator, year, association_matrix
                        )
                    
                    # Calculate forecast
                    forecast = baseline + trend + event_impact
                    
                    # Apply realistic limits
                    forecast = self._apply_limits(indicator, forecast, baseline)
                    
                    values[indicator] = round(forecast, 1)
                
                year_scenarios[scenario_name] = values
            
            scenarios[year] = year_scenarios
        
        return scenarios
    
    def _calculate_trend_component(self, indicator: str, baseline: float,
                                 year: int, multiplier: float) -> float:
        """Calculate trend component for forecasting."""
        years_ahead = year - self.baseline_year
        
        if years_ahead <= 0:
            return 0.0
        
        annual_growth = self.trend_growth_rates.get(indicator, 1.0)
        
        # Apply diminishing returns for some indicators
        if indicator in ['ACC_OWNERSHIP', 'ACC_MM_ACCOUNT']:
            # Diminishing returns as saturation approaches
            saturation = 100.0 if indicator == 'ACC_OWNERSHIP' else 90.0
            remaining_gap = saturation - baseline
            if remaining_gap > 0:
                # Adjust growth based on remaining gap
                adjustment = min(1.0, remaining_gap / 50.0)
                annual_growth *= adjustment
        
        return annual_growth * years_ahead * multiplier
    
    def _calculate_event_impact(self, indicator: str, year: int,
                              association_matrix: pd.DataFrame) -> float:
        """Calculate cumulative event impact for an indicator in a given year."""
        if association_matrix is None or association_matrix.empty:
            return 0.0
        
        total_impact = 0.0
        
        for _, event_row in association_matrix.iterrows():
            impact_col = f'{indicator}_impact'
            lag_col = f'{indicator}_lag'
            
            if impact_col in event_row and lag_col in event_row:
                impact_str = str(event_row[impact_col])
                lag_months = event_row[lag_col]
                
                if 'pp' in impact_str:
                    # Extract numeric impact
                    try:
                        magnitude = float(impact_str.replace('+', '').replace('-', '').replace('pp', ''))
                        if magnitude > 0:
                            # Simple implementation: full impact after lag
                            # More sophisticated: distributed over time
                            total_impact += magnitude
                    except:
                        continue
        
        return total_impact
    
    def _apply_limits(self, indicator: str, forecast: float, 
                     baseline: float) -> float:
        """Apply realistic limits to forecasts."""
        if indicator == 'ACC_OWNERSHIP':
            return min(forecast, 100.0)  # Can't exceed 100%
        
        elif indicator == 'ACC_MM_ACCOUNT':
            # Mobile money can't exceed account ownership
            max_mm = baseline * 1.5  # Allow some overshoot
            return min(forecast, max_mm)
        
        elif indicator == 'GEN_GAP_ACC':
            # Gender gap should be positive and decreasing
            return max(0.0, forecast)  # Can't be negative
        
        elif indicator in ['USG_ACTIVE_RATE', 'INF_AGENT_DENSITY']:
            return min(forecast, 100.0)  # Percentage limits
        
        return forecast
    
    def generate_event_augmented_scenarios(self, baseline_values: Dict[str, float],
                                         association_matrix: pd.DataFrame,
                                         event_schedule: Dict[int, List[str]] = None) -> Dict:
        """
        Generate scenarios with detailed event impact modeling.
        
        Args:
            baseline_values: Dictionary of baseline values
            association_matrix: Event-indicator association matrix
            event_schedule: Optional schedule of future events by year
            
        Returns:
            Enhanced scenarios with event impacts
        """
        if event_schedule is None:
            event_schedule = self._get_default_event_schedule()
        
        scenarios = {}
        
        # Initialize with baseline
        current_values = baseline_values.copy()
        
        for year in sorted(self.forecast_years):
            year_scenarios = {}
            
            for scenario_name, multiplier in self.scenario_multipliers.items():
                # Start from previous year or baseline
                if year == self.forecast_years[0]:
                    base_values = current_values.copy()
                else:
                    base_values = scenarios[year-1]['baseline'].copy()
                
                values = {}
                
                for indicator, base_value in base_values.items():
                    # Calculate trend
                    trend = self._calculate_trend_component(
                        indicator, base_value, year, multiplier
                    )
                    
                    # Calculate event impacts
                    event_impact = self._calculate_scheduled_event_impact(
                        indicator, year, event_schedule, association_matrix
                    )
                    
                    # Calculate forecast
                    forecast = base_value + trend + event_impact
                    forecast = self._apply_limits(indicator, forecast, base_value)
                    
                    values[indicator] = round(forecast, 1)
                
                year_scenarios[scenario_name] = values
            
            scenarios[year] = year_scenarios
        
        return scenarios
    
    def _get_default_event_schedule(self) -> Dict[int, List[str]]:
        """Get default schedule of future events."""
        return {
            2025: ['EVT_ETHIOPAY', 'EVT_MPESA_INTEROP'],
            2026: ['EVT_FAYDA_EXPANSION', 'EVT_AGENT_NETWORK'],
            2027: ['EVT_NFIS3', 'EVT_DIGITAL_LITERACY']
        }
    
    def _calculate_scheduled_event_impact(self, indicator: str, year: int,
                                        event_schedule: Dict[int, List[str]],
                                        association_matrix: pd.DataFrame) -> float:
        """Calculate impact from scheduled events."""
        total_impact = 0.0
        
        for event_year, event_ids in event_schedule.items():
            if event_year <= year:
                for event_id in event_ids:
                    event_impact = self._get_event_impact_for_indicator(
                        event_id, indicator, association_matrix
                    )
                    
                    # Apply time decay for past events
                    if event_year < year:
                        decay_factor = 0.8 ** (year - event_year)  # 20% decay per year
                        event_impact *= decay_factor
                    
                    total_impact += event_impact
        
        return total_impact
    
    def _get_event_impact_for_indicator(self, event_id: str, indicator: str,
                                       association_matrix: pd.DataFrame) -> float:
        """Get impact magnitude for a specific event and indicator."""
        event_row = association_matrix[
            association_matrix['event_id'] == event_id
        ]
        
        if event_row.empty:
            return 0.0
        
        impact_col = f'{indicator}_impact'
        if impact_col in event_row.columns:
            impact_str = str(event_row[impact_col].iloc[0])
            if 'pp' in impact_str:
                try:
                    return float(impact_str.replace('+', '').replace('-', '').replace('pp', ''))
                except:
                    return 0.0
        
        return 0.0
    
    def quantify_uncertainty(self, scenarios: Dict) -> Dict:
        """
        Quantify uncertainty in forecast scenarios.
        
        Args:
            scenarios: Dictionary of forecast scenarios
            
        Returns:
            Dictionary with uncertainty metrics
        """
        uncertainty_metrics = {}
        
        for year in scenarios:
            year_metrics = {}
            
            for indicator in self.trend_growth_rates.keys():
                if (indicator in scenarios[year]['pessimistic'] and 
                    indicator in scenarios[year]['optimistic']):
                    
                    pessimistic = scenarios[year]['pessimistic'][indicator]
                    optimistic = scenarios[year]['optimistic'][indicator]
                    baseline = scenarios[year]['baseline'][indicator]
                    
                    # Calculate various uncertainty metrics
                    range_size = optimistic - pessimistic
                    midpoint = (pessimistic + optimistic) / 2
                    
                    # Relative uncertainty
                    if baseline > 0:
                        relative_uncertainty = (range_size / (2 * baseline)) * 100
                    else:
                        relative_uncertainty = 0.0
                    
                    # Coefficient of variation (simplified)
                    if midpoint > 0:
                        cv = (range_size / 2) / midpoint
                    else:
                        cv = 0.0
                    
                    year_metrics[indicator] = {
                        'range_pp': round(range_size, 2),
                        'midpoint': round(midpoint, 2),
                        'uncertainty_margin': round(range_size / 2, 2),
                        'relative_uncertainty_percent': round(relative_uncertainty, 1),
                        'coefficient_of_variation': round(cv, 3),
                        'pessimistic': pessimistic,
                        'optimistic': optimistic,
                        'baseline': baseline
                    }
            
            uncertainty_metrics[year] = year_metrics
        
        return uncertainty_metrics
    
    def generate_scenario_report(self, scenarios: Dict, 
                               uncertainty_metrics: Dict = None) -> Dict:
        """
        Generate comprehensive scenario report.
        
        Args:
            scenarios: Dictionary of forecast scenarios
            uncertainty_metrics: Optional uncertainty metrics
            
        Returns:
            Comprehensive scenario report
        """
        if uncertainty_metrics is None:
            uncertainty_metrics = self.quantify_uncertainty(scenarios)
        
        report = {
            'scenario_overview': {
                'baseline_year': self.baseline_year,
                'forecast_years': self.forecast_years,
                'scenario_types': list(self.scenario_multipliers.keys()),
                'indicators_modeled': list(self.trend_growth_rates.keys()),
                'report_generated': datetime.now().isoformat()
            },
            'scenarios': scenarios,
            'uncertainty_analysis': uncertainty_metrics,
            'key_findings': self._extract_key_findings(scenarios, uncertainty_metrics),
            'methodology': {
                'approach': 'Trend extrapolation with scenario analysis',
                'growth_rates': self.trend_growth_rates,
                'scenario_multipliers': self.scenario_multipliers,
                'limits_applied': ['Saturation limits', 'Realistic bounds']
            }
        }
        
        return report
    
    def _extract_key_findings(self, scenarios: Dict, 
                            uncertainty_metrics: Dict) -> List[str]:
        """Extract key findings from scenarios and uncertainty analysis."""
        findings = []
        
        # Find key indicators
        key_indicators = ['ACC_OWNERSHIP', 'ACC_MM_ACCOUNT']
        
        for indicator in key_indicators:
            if (2027 in scenarios and 
                indicator in scenarios[2027]['baseline']):
                
                baseline_2027 = scenarios[2027]['baseline'][indicator]
                baseline_2025 = scenarios[2025]['baseline'][indicator]
                
                growth = baseline_2027 - baseline_2025
                annual_growth = growth / 2
                
                findings.append(
                    f"{indicator}: Projected to reach {baseline_2027}% by 2027 "
                    f"(annual growth: {annual_growth:.1f}pp/year)"
                )
        
        # Add uncertainty findings
        if 2027 in uncertainty_metrics:
            for indicator in key_indicators:
                if indicator in uncertainty_metrics[2027]:
                    uncertainty = uncertainty_metrics[2027][indicator]
                    findings.append(
                        f"{indicator} uncertainty: "
                        f"±{uncertainty['uncertainty_margin']:.1f}pp "
                        f"({uncertainty['relative_uncertainty_percent']:.1f}% relative)"
                    )
        
        # Add NFIS-II target analysis
        if 'ACC_OWNERSHIP' in scenarios[2025]['baseline']:
            target_2025 = 70.0
            baseline_2025 = scenarios[2025]['baseline']['ACC_OWNERSHIP']
            gap = target_2025 - baseline_2025
            
            findings.append(
                f"NFIS-II target gap: {gap:.1f}pp short of 70% target for 2025"
            )
        
        return findings