"""
Scenario generation with reference code integration
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class ScenarioGenerator:
    def __init__(self):
        self.scenario_definitions = {
            'pessimistic': {'multiplier': 0.7, 'description': 'Slow adoption, challenges'},
            'baseline': {'multiplier': 1.0, 'description': 'Business as usual'},
            'optimistic': {'multiplier': 1.3, 'description': 'Accelerated growth'}
        }
    
    def generate_reference_based_scenarios(self, modeler, ref_codes, years, 
                                          scenario_types=None):
        """Generate scenarios with reference context"""
        if scenario_types is None:
            scenario_types = ['pessimistic', 'baseline', 'optimistic']
        
        # Get baseline values
        baseline_2024 = modeler.get_baseline_values(2024)
        
        # Generate scenarios for each year
        scenarios = {}
        for year in years:
            scenarios[year] = {}
            
            for scenario in scenario_types:
                multiplier = self.scenario_definitions[scenario]['multiplier']
                
                # Calculate projected values
                projected = {}
                for indicator, baseline in baseline_2024.items():
                    # Base growth + scenario adjustment
                    annual_growth = self._get_annual_growth(indicator)
                    years_ahead = year - 2024
                    
                    projected[indicator] = baseline * (1 + annual_growth * years_ahead * multiplier / 100)
                
                scenarios[year][scenario] = projected
        
        return scenarios
    
    def _get_annual_growth(self, indicator: str) -> float:
        """Get annual growth rate for indicator"""
        growth_rates = {
            'ACC_OWNERSHIP': 2.5,  # 2.5% annual growth
            'ACC_MM_ACCOUNT': 10.0,  # 10% annual growth
            'USG_DIGITAL_PAYMENT': 5.0   # 5% annual growth
        }
        
        return growth_rates.get(indicator, 2.0)
    
    def get_scenario_description(self, scenario: str) -> str:
        """Get description for a scenario"""
        return self.scenario_definitions.get(scenario, {}).get('description', 'Unknown')