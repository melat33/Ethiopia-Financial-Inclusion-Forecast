"""
Scenario generation engine for financial inclusion forecasts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ScenarioParameters:
    """Parameters for scenario generation"""
    scenario_names: List[str] = None
    growth_multipliers: Dict[str, float] = None
    event_probabilities: Dict[str, float] = None
    shock_magnitudes: Dict[str, float] = None
    
    def __post_init__(self):
        if self.scenario_names is None:
            self.scenario_names = ['pessimistic', 'baseline', 'optimistic']
        if self.growth_multipliers is None:
            self.growth_multipliers = {'pessimistic': 0.7, 'baseline': 1.0, 'optimistic': 1.3}
        if self.event_probabilities is None:
            self.event_probabilities = {'pessimistic': 0.3, 'baseline': 0.7, 'optimistic': 0.9}
        if self.shock_magnitudes is None:
            self.shock_magnitudes = {'pessimistic': -2.0, 'baseline': 0.0, 'optimistic': 2.0}


class ScenarioGenerator:
    """Generate and analyze forecast scenarios"""
    
    def __init__(self, params: ScenarioParameters = None):
        self.params = params or ScenarioParameters()
        self.scenarios = {}
        
    def generate_all_scenarios(self, forecast_results: Dict) -> Dict:
        """Generate comprehensive scenarios from base forecasts"""
        
        scenarios = {}
        
        for indicator in forecast_results.keys():
            if 'ensemble' in forecast_results[indicator]:
                base_forecasts = forecast_results[indicator]['ensemble']['forecasts']
                indicator_scenarios = self._generate_indicator_scenarios(indicator, base_forecasts)
                scenarios[indicator] = indicator_scenarios
        
        self.scenarios = scenarios
        return scenarios
    
    def _generate_indicator_scenarios(self, indicator: str, 
                                    base_forecasts: Dict[int, float]) -> Dict:
        """Generate scenarios for a single indicator"""
        
        indicator_scenarios = {}
        
        for scenario_name in self.params.scenario_names:
            scenario_forecasts = {}
            
            for year, base_value in base_forecasts.items():
                # Apply scenario adjustments
                adjusted_value = base_value
                
                # Growth multiplier
                growth_multiplier = self.params.growth_multipliers[scenario_name]
                years_from_base = year - 2024
                
                # Estimate annual growth from historical trend
                annual_growth = self._estimate_annual_growth(indicator)
                growth_adjustment = annual_growth * (growth_multiplier - 1) * years_from_base
                adjusted_value += growth_adjustment
                
                # Event probability adjustment
                event_prob = self.params.event_probabilities[scenario_name]
                event_impact = self._estimate_event_impact(indicator) * event_prob
                adjusted_value += event_impact
                
                # Random shocks (for Monte Carlo)
                shock = self.params.shock_magnitudes[scenario_name]
                if scenario_name != 'baseline':
                    adjusted_value += shock
                
                # Apply bounds
                adjusted_value = max(0, min(100, adjusted_value))
                scenario_forecasts[year] = round(adjusted_value, 2)
            
            indicator_scenarios[scenario_name] = {
                'forecasts': scenario_forecasts,
                'description': self._get_scenario_description(scenario_name),
                'assumptions': self._get_scenario_assumptions(scenario_name, indicator),
                'probability': self._estimate_scenario_probability(scenario_name)
            }
        
        return indicator_scenarios
    
    def _estimate_annual_growth(self, indicator: str) -> float:
        """Estimate annual growth rate for indicator"""
        # Historical growth rates from Findex
        historical_growth = {
            'ACC_OWNERSHIP': 2.5,  # pp/year average
            'USG_DIGITAL_PAYMENT': 5.0,
            'ACC_MM_ACCOUNT': 3.0
        }
        return historical_growth.get(indicator, 2.0)
    
    def _estimate_event_impact(self, indicator: str) -> float:
        """Estimate total event impact for indicator"""
        # Based on Task 3 event matrix analysis
        event_impacts = {
            'ACC_OWNERSHIP': 4.0,  # Total pp impact from events
            'USG_DIGITAL_PAYMENT': 6.0,
            'ACC_MM_ACCOUNT': 8.0
        }
        return event_impacts.get(indicator, 3.0)
    
    def _get_scenario_description(self, scenario_name: str) -> str:
        """Get descriptive text for scenario"""
        descriptions = {
            'pessimistic': "Slow growth, regulatory challenges, economic headwinds, limited event implementation",
            'baseline': "Current trajectory continuation, moderate policy reforms, steady infrastructure growth",
            'optimistic': "Strong reforms, accelerated digital adoption, favorable economic conditions, successful event implementation"
        }
        return descriptions.get(scenario_name, "Standard scenario")
    
    def _get_scenario_assumptions(self, scenario_name: str, indicator: str) -> List[str]:
        """Get key assumptions for scenario"""
        assumptions = {
            'pessimistic': [
                "GDP growth below 5%",
                "Limited infrastructure investment",
                "Slow regulatory reforms",
                "Weak event implementation"
            ],
            'baseline': [
                "GDP growth 6-7%",
                "Moderate infrastructure investment",
                "Steady regulatory progress",
                "Partial event implementation"
            ],
            'optimistic': [
                "GDP growth above 8%",
                "Strong infrastructure push",
                "Accelerated regulatory reforms",
                "Full event implementation"
            ]
        }
        return assumptions.get(scenario_name, [])
    
    def _estimate_scenario_probability(self, scenario_name: str) -> float:
        """Estimate subjective probability of scenario"""
        probabilities = {
            'pessimistic': 0.2,   # 20% probability
            'baseline': 0.6,      # 60% probability
            'optimistic': 0.2     # 20% probability
        }
        return probabilities.get(scenario_name, 0.33)
    
    def calculate_scenario_statistics(self, scenarios: Dict) -> pd.DataFrame:
        """Calculate statistics across scenarios"""
        records = []
        
        for indicator, indicator_scenarios in scenarios.items():
            for year in [2025, 2026, 2027]:
                values = []
                probabilities = []
                
                for scenario_name, scenario_data in indicator_scenarios.items():
                    if year in scenario_data['forecasts']:
                        values.append(scenario_data['forecasts'][year])
                        probabilities.append(scenario_data.get('probability', 0.33))
                
                if values:
                    # Weighted statistics
                    weights = np.array(probabilities) / sum(probabilities)
                    weighted_mean = np.average(values, weights=weights)
                    weighted_std = np.sqrt(np.average((values - weighted_mean)**2, weights=weights))
                    
                    records.append({
                        'indicator': indicator,
                        'year': year,
                        'weighted_mean': weighted_mean,
                        'weighted_std': weighted_std,
                        'min': min(values),
                        'max': max(values),
                        'range': max(values) - min(values),
                        'scenario_count': len(values)
                    })
        
        return pd.DataFrame(records)