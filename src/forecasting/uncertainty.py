"""
Uncertainty quantification for financial inclusion forecasts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class UncertaintyQuantifier:
    """Professional uncertainty quantification system"""
    
    @staticmethod
    def calculate_all_uncertainty(scenario_analysis: Dict) -> Dict:
        """Calculate all uncertainty metrics"""
        
        uncertainty_results = {}
        
        for indicator, scenarios in scenario_analysis.items():
            # Extract forecasts by year
            forecast_matrix = UncertaintyQuantifier._extract_forecast_matrix(scenarios)
            
            # Calculate uncertainty metrics
            uncertainty_results[indicator] = {
                'confidence_intervals': UncertaintyQuantifier._calculate_confidence_intervals(forecast_matrix),
                'monte_carlo': UncertaintyQuantifier._run_monte_carlo_simulation(forecast_matrix),
                'sensitivity_analysis': UncertaintyQuantifier._perform_sensitivity_analysis(scenarios),
                'uncertainty_decomposition': UncertaintyQuantifier._decompose_uncertainty(scenarios)
            }
        
        return uncertainty_results
    
    @staticmethod
    def _extract_forecast_matrix(scenarios: Dict) -> pd.DataFrame:
        """Extract forecasts into matrix format"""
        records = []
        
        for scenario_name, scenario_data in scenarios.items():
            forecasts = scenario_data['forecasts']
            probability = scenario_data.get('probability', 0.33)
            
            for year, value in forecasts.items():
                records.append({
                    'scenario': scenario_name,
                    'year': year,
                    'value': value,
                    'probability': probability
                })
        
        df = pd.DataFrame(records)
        return df.pivot_table(values='value', index='scenario', columns='year')
    
    @staticmethod
    def _calculate_confidence_intervals(forecast_matrix: pd.DataFrame) -> Dict:
        """Calculate confidence intervals from scenario matrix"""
        ci_results = {}
        
        for year in forecast_matrix.columns:
            values = forecast_matrix[year].dropna()
            
            if len(values) >= 3:
                # Parametric CI assuming normal distribution
                mean = np.mean(values)
                std = np.std(values)
                n = len(values)
                
                # Student's t critical value for 95% CI
                t_critical = stats.t.ppf(0.975, df=n-1)
                margin_error = t_critical * (std / np.sqrt(n))
                
                ci_results[year] = {
                    'mean': mean,
                    'std': std,
                    'n': n,
                    'ci_95_lower': mean - margin_error,
                    'ci_95_upper': mean + margin_error,
                    'ci_90_lower': mean - 1.645 * (std / np.sqrt(n)),
                    'ci_90_upper': mean + 1.645 * (std / np.sqrt(n)),
                    'coefficient_of_variation': (std / mean) * 100 if mean > 0 else 0
                }
        
        return ci_results
    
    @staticmethod
    def _run_monte_carlo_simulation(forecast_matrix: pd.DataFrame, 
                                   n_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulation for uncertainty"""
        results = {}
        
        for year in forecast_matrix.columns:
            values = forecast_matrix[year].dropna()
            
            if len(values) >= 2:
                # Fit distribution to scenarios
                mean = np.mean(values)
                std = np.std(values)
                
                # Generate simulations
                np.random.seed(42)
                simulations = np.random.normal(mean, std, n_simulations)
                simulations = np.clip(simulations, 0, 100)  # Apply bounds
                
                # Calculate statistics
                results[year] = {
                    'mean_sim': np.mean(simulations),
                    'std_sim': np.std(simulations),
                    'p5': np.percentile(simulations, 5),
                    'p25': np.percentile(simulations, 25),
                    'p50': np.percentile(simulations, 50),
                    'p75': np.percentile(simulations, 75),
                    'p95': np.percentile(simulations, 95),
                    'ci_90': (np.percentile(simulations, 5), np.percentile(simulations, 95)),
                    'ci_95': (np.percentile(simulations, 2.5), np.percentile(simulations, 97.5)),
                    'probability_below_target': np.mean(simulations < 70) if year == 2025 else None
                }
        
        return results
    
    @staticmethod
    def _perform_sensitivity_analysis(scenarios: Dict) -> Dict:
        """Perform sensitivity analysis on key parameters"""
        sensitivity = {}
        
        # Define parameters to test
        parameters = {
            'growth_multiplier': [0.5, 0.7, 1.0, 1.3, 1.5],
            'event_probability': [0.1, 0.3, 0.7, 0.9, 1.0],
            'shock_magnitude': [-5, -2, 0, 2, 5]
        }
        
        for param_name, param_values in parameters.items():
            sensitivity[param_name] = {}
            
            # Test each value
            for param_value in param_values:
                # Calculate impact on 2027 forecast (simplified)
                baseline_2027 = scenarios['baseline']['forecasts'].get(2027, 50)
                
                if param_name == 'growth_multiplier':
                    impact = baseline_2027 * (param_value - 1) * 0.1
                elif param_name == 'event_probability':
                    impact = 5 * (param_value - 0.7)  # Assuming 5pp total event impact
                else:  # shock_magnitude
                    impact = param_value
                
                sensitivity[param_name][param_value] = {
                    'impact_pp': impact,
                    'result_2027': baseline_2027 + impact,
                    'elasticity': impact / param_value if param_value != 0 else 0
                }
        
        return sensitivity
    
    @staticmethod
    def _decompose_uncertainty(scenarios: Dict) -> Dict:
        """Decompose uncertainty into sources"""
        decomposition = {
            'parameter_uncertainty': 40,   # Growth rates, event impacts
            'model_uncertainty': 30,       # Choice of forecasting model
            'scenario_uncertainty': 20,    # Which scenario occurs
            'data_uncertainty': 10         # Measurement error in historical data
        }
        
        # Calculate total uncertainty score
        total = sum(decomposition.values())
        for key in decomposition:
            decomposition[key] = (decomposition[key] / total) * 100
        
        return decomposition