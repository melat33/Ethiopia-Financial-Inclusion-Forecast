"""
Core forecasting models for Ethiopia financial inclusion
Professional, modular implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ForecastMethod(Enum):
    LINEAR_TREND = "linear_trend"
    EVENT_AUGMENTED = "event_augmented"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"


@dataclass
class ForecastParameters:
    """Configuration for forecasting models"""
    forecast_horizon: List[int] = None
    baseline_year: int = 2024
    confidence_level: float = 0.95
    n_simulations: int = 1000
    
    def __post_init__(self):
        if self.forecast_horizon is None:
            self.forecast_horizon = [2025, 2026, 2027]


class FinancialInclusionForecaster:
    """Professional forecasting system for Ethiopia FI indicators"""
    
    # Historical benchmarks (from Global Findex)
    HISTORICAL_BENCHMARKS = {
        'ACC_OWNERSHIP': {
            2011: 14.0,
            2014: 22.0,
            2017: 35.0,
            2021: 46.0,
            2024: 49.0
        },
        'USG_DIGITAL_PAYMENT': {
            2014: 10.0,
            2017: 18.0,
            2021: 25.0,
            2024: 35.0
        }
    }
    
    # NFIS-II targets
    NFIS_TARGETS = {
        2025: {'ACC_OWNERSHIP': 70.0, 'USG_DIGITAL_PAYMENT': 45.0},
        2030: {'ACC_OWNERSHIP': 75.0, 'USG_DIGITAL_PAYMENT': 60.0}
    }
    
    def __init__(self, params: ForecastParameters = None):
        self.params = params or ForecastParameters()
        self.models = {}
        self.forecasts = {}
        
    def fit_linear_trend(self, indicator: str, data: pd.DataFrame) -> Dict:
        """Fit linear trend model with regularization - FIXED VERSION"""
        if indicator not in data.columns:
            raise ValueError(f"Indicator {indicator} not in data")
        
        valid_data = data[['year', indicator]].dropna()
        if len(valid_data) < 2:
            return None
        
        # Bayesian linear regression with weak priors
        X = valid_data['year'].values - 2000  # Center years
        y = valid_data[indicator].values
        
        # Ensure X is 2D for matrix operations
        X = X.reshape(-1, 1)
        
        # Add regularization (ridge regression)
        lambda_reg = 0.1
        X_centered = X - np.mean(X)
        
        # Fix the matrix dimension issue
        n_features = X_centered.shape[1]
        identity_matrix = np.eye(n_features)
        
        try:
            # Calculate coefficients using matrix algebra
            XTX = X_centered.T @ X_centered
            XTX_regularized = XTX + lambda_reg * identity_matrix
            
            # Check if matrix is invertible
            if np.linalg.matrix_rank(XTX_regularized) < n_features:
                # Use pseudoinverse if not invertible
                beta = np.linalg.pinv(XTX_regularized) @ X_centered.T @ y
            else:
                beta = np.linalg.inv(XTX_regularized) @ X_centered.T @ y
            
            alpha = np.mean(y) - beta * np.mean(X)
            
        except np.linalg.LinAlgError:
            # Fallback to simple linear regression
            print(f"⚠️ Matrix inversion failed for {indicator}, using simple regression")
            coeffs = np.polyfit(X.flatten(), y, 1)
            alpha = coeffs[1]
            beta = coeffs[0]
        
        # Forecast
        forecasts = {}
        for year in self.params.forecast_horizon:
            X_pred = year - 2000
            forecast = alpha + beta * X_pred
            
            # Handle scalar vs array
            if hasattr(forecast, '__len__'):
                forecast = float(forecast[0])
            else:
                forecast = float(forecast)
            
            # Apply bounds
            forecast = max(0, min(100, forecast))
            forecasts[year] = forecast
        
        return {
            'method': 'bayesian_linear',
            'alpha': float(alpha),
            'beta': float(beta[0]) if hasattr(beta, '__len__') else float(beta),
            'forecasts': forecasts,
            'r_squared': self._calculate_r_squared(X.flatten(), y, alpha, beta)
        }
    
    def fit_event_augmented_model(self, indicator: str, 
                                 historical_data: pd.DataFrame,
                                 event_matrix: pd.DataFrame) -> Dict:
        """Augment trend model with event impacts"""
        # Base trend
        trend_model = self.fit_linear_trend(indicator, historical_data)
        if trend_model is None:
            return None
        
        # Extract event impacts
        event_impacts = self._extract_event_impacts(indicator, event_matrix)
        
        # Apply event impacts with time decay
        augmented_forecasts = {}
        for year in self.params.forecast_horizon:
            base_value = trend_model['forecasts'][year]
            
            # Sum event impacts with exponential decay
            total_event_impact = 0
            for event in event_impacts:
                event_year = event.get('year', 2024)
                years_since_event = year - event_year
                if years_since_event >= 0:
                    impact = event.get('impact', 0)
                    decay = np.exp(-0.3 * years_since_event)  # 30% decay per year
                    total_event_impact += impact * decay
            
            augmented_value = base_value + total_event_impact
            augmented_forecasts[year] = max(0, min(100, augmented_value))
        
        return {
            'method': 'event_augmented',
            'base_model': trend_model,
            'event_impacts': event_impacts,
            'total_event_impact': sum(e.get('impact', 0) for e in event_impacts),
            'forecasts': augmented_forecasts
        }
    
    def generate_ensemble_forecast(self, indicator: str,
                                 historical_data: pd.DataFrame,
                                 event_matrix: pd.DataFrame) -> Dict:
        """Ensemble of multiple forecasting methods"""
        models = []
        weights = []
        
        # Linear trend model
        linear_model = self.fit_linear_trend(indicator, historical_data)
        if linear_model:
            models.append(linear_model)
            weights.append(0.4)  # 40% weight
        
        # Event augmented model
        event_model = self.fit_event_augmented_model(indicator, historical_data, event_matrix)
        if event_model:
            models.append(event_model)
            weights.append(0.6)  # 60% weight
        
        if not models:
            return None
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Weighted ensemble
        ensemble_forecasts = {}
        for year in self.params.forecast_horizon:
            weighted_sum = 0
            for model, weight in zip(models, weights):
                if year in model['forecasts']:
                    weighted_sum += model['forecasts'][year] * weight
            
            # Apply bounds
            ensemble_forecasts[year] = max(0, min(100, weighted_sum))
        
        return {
            'method': 'weighted_ensemble',
            'component_models': models,
            'weights': weights.tolist(),
            'forecasts': ensemble_forecasts
        }
    
    def generate_complete_forecasts(self, 
                                  historical_data: pd.DataFrame,
                                  event_matrix: pd.DataFrame,
                                  target_data: pd.DataFrame = None) -> Dict:
        """Generate forecasts for all key indicators"""
        
        indicators = ['ACC_OWNERSHIP', 'USG_DIGITAL_PAYMENT']
        results = {}
        
        for indicator in indicators:
            print(f"\n📊 Forecasting {indicator}...")
            
            # Check if we have enough historical data
            if indicator not in historical_data.columns:
                print(f"   ⚠️ {indicator} not in historical data, using benchmark")
                # Use historical benchmarks
                hist_benchmark = self.HISTORICAL_BENCHMARKS.get(indicator, {})
                if hist_benchmark:
                    # Create synthetic historical data
                    hist_df = pd.DataFrame({
                        'year': list(hist_benchmark.keys()),
                        indicator: list(hist_benchmark.values())
                    })
                    historical_data = historical_data.merge(hist_df, on='year', how='outer')
                else:
                    print(f"   ❌ No benchmark for {indicator}, skipping")
                    continue
            
            # Ensemble forecast (best practice)
            ensemble_result = self.generate_ensemble_forecast(
                indicator, historical_data, event_matrix
            )
            
            if ensemble_result:
                results[indicator] = {
                    'ensemble': ensemble_result,
                    'target_gap': self._calculate_target_gap(indicator, ensemble_result),
                    'growth_rate': self._calculate_growth_rate(ensemble_result)
                }
                print(f"   ✅ Forecast generated")
                for year, value in ensemble_result['forecasts'].items():
                    print(f"     {year}: {value:.1f}%")
            else:
                print(f"   ❌ Failed to generate forecast for {indicator}")
        
        self.forecasts = results
        return results
    
    def _extract_event_impacts(self, indicator: str, event_matrix: pd.DataFrame) -> List[Dict]:
        """Extract and format event impacts from matrix"""
        impacts = []
        
        if event_matrix is not None and not event_matrix.empty:
            # Try different column naming patterns
            impact_col_patterns = [
                f"{indicator}_impact",
                f"{indicator}_impact_pp",
                f"{indicator.lower()}_impact"
            ]
            
            impact_col = None
            for pattern in impact_col_patterns:
                if pattern in event_matrix.columns:
                    impact_col = pattern
                    break
            
            if impact_col:
                for _, row in event_matrix.iterrows():
                    try:
                        impact_value = row[impact_col]
                        if pd.isna(impact_value):
                            continue
                            
                        # Convert to float, handle string values
                        if isinstance(impact_value, str):
                            impact_value = float(impact_value.replace('pp', '').replace('%', ''))
                        
                        # Get event year
                        event_year = None
                        for year_col in ['event_year', 'year', 'event_date']:
                            if year_col in row and pd.notna(row[year_col]):
                                try:
                                    event_year = int(float(row[year_col]))
                                    break
                                except:
                                    continue
                        
                        if event_year is None:
                            event_year = 2024  # Default
                        
                        impacts.append({
                            'event': row.get('event_name', 'Unknown'),
                            'impact': float(impact_value),
                            'year': event_year,
                            'confidence': row.get('confidence', 'medium')
                        })
                    except Exception as e:
                        continue
        
        return impacts
    
    def _calculate_target_gap(self, indicator: str, forecast_result: Dict) -> Dict:
        """Calculate gap to NFIS-II targets"""
        gaps = {}
        
        for year in self.params.forecast_horizon:
            if year in self.NFIS_TARGETS and indicator in self.NFIS_TARGETS[year]:
                target = self.NFIS_TARGETS[year][indicator]
                forecast = forecast_result['forecasts'].get(year, 0)
                gap = target - forecast
                
                gaps[year] = {
                    'target': target,
                    'forecast': forecast,
                    'gap_pp': gap,
                    'gap_percent': (gap / target) * 100 if target > 0 else 0,
                    'required_annual_growth': max(0, gap / (year - 2024)) if year > 2024 else 0
                }
        
        return gaps
    
    def _calculate_r_squared(self, X: np.ndarray, y: np.ndarray, 
                           alpha: float, beta: float) -> float:
        """Calculate R-squared for regression model"""
        y_pred = alpha + beta * X
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _calculate_growth_rate(self, forecast_result: Dict) -> Dict:
        """Calculate implied growth rates from forecasts"""
        forecasts = forecast_result['forecasts']
        years = sorted(forecasts.keys())
        
        if len(years) < 2:
            return {}
        
        growth_rates = {}
        for i in range(1, len(years)):
            year1, year2 = years[i-1], years[i]
            value1, value2 = forecasts[year1], forecasts[year2]
            annual_growth = (value2 - value1) / (year2 - year1)
            growth_rates[f"{year1}-{year2}"] = annual_growth
        
        return growth_rates
    
    def save_results(self, output_dir: str):
        """Save all forecast results to files"""
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forecasts
        forecasts_df = self._convert_to_dataframe()
        if not forecasts_df.empty:
            forecasts_df.to_csv(f"{output_dir}/forecasts_2025_2027.csv", index=False)
            print(f"✅ Saved forecasts to {output_dir}/forecasts_2025_2027.csv")
        
        # Save model details
        model_details = {
            'parameters': self.params.__dict__,
            'timestamp': pd.Timestamp.now().isoformat(),
            'indicators_forecasted': list(self.forecasts.keys())
        }
        
        with open(f"{output_dir}/model_details.json", 'w') as f:
            json.dump(model_details, f, indent=2)
        
        # Save target gap analysis
        gap_analysis = {}
        for indicator, result in self.forecasts.items():
            if 'target_gap' in result:
                gap_analysis[indicator] = result['target_gap']
        
        if gap_analysis:
            with open(f"{output_dir}/target_gap_analysis.json", 'w') as f:
                json.dump(gap_analysis, f, indent=2)
        
        print(f"✅ All results saved to {output_dir}")
    
    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convert forecasts to DataFrame for export"""
        records = []
        
        for indicator, result in self.forecasts.items():
            if 'ensemble' in result:
                forecasts = result['ensemble']['forecasts']
                for year, value in forecasts.items():
                    records.append({
                        'indicator': indicator,
                        'year': year,
                        'forecast_value': value,
                        'model_type': result['ensemble']['method']
                    })
        
        return pd.DataFrame(records)
    
    def generate_final_report(self, scenario_analysis: Dict = None,
                            uncertainty_analysis: Dict = None) -> str:
        """Generate comprehensive final report"""
        from datetime import datetime
        
        report = f"""
# ETHIOPIA FINANCIAL INCLUSION FORECASTING REPORT
## Task 4: Forecasting Access and Usage (2025-2027)
### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY

### Key Forecasts (2027 Baseline)
"""
        
        # Add forecasts
        for indicator in ['ACC_OWNERSHIP', 'USG_DIGITAL_PAYMENT']:
            if indicator in self.forecasts:
                forecast_2027 = self.forecasts[indicator]['ensemble']['forecasts'].get(2027, 'N/A')
                if isinstance(forecast_2027, (int, float)):
                    report += f"- **{indicator.replace('_', ' ')}**: {forecast_2027:.1f}%\n"
                else:
                    report += f"- **{indicator.replace('_', ' ')}**: {forecast_2027}\n"
        
        report += f"""

### NFIS-II Target Analysis
"""
        
        # Add target gap analysis
        if 'ACC_OWNERSHIP' in self.forecasts and 'target_gap' in self.forecasts['ACC_OWNERSHIP']:
            gaps = self.forecasts['ACC_OWNERSHIP']['target_gap']
            for year, gap_info in gaps.items():
                report += f"- **{year}**: Target = {gap_info['target']}%, Forecast = {gap_info['forecast']:.1f}%, Gap = {gap_info['gap_pp']:.1f}pp\n"
        
        report += f"""

## METHODOLOGY
Used weighted ensemble approach combining:
1. Bayesian linear trend regression
2. Event-augmented forecasting
3. Scenario-based uncertainty quantification

## KEY INSIGHTS
"""
        
        # Add key insights
        insights = self._generate_insights()
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"
        
        report += f"""

## RECOMMENDATIONS
1. Focus on closing NFIS-II target gap through targeted interventions
2. Prioritize digital payment infrastructure expansion
3. Monitor event impacts and adjust forecasts quarterly
4. Invest in data collection for better forecasting accuracy

## LIMITATIONS
- Sparse historical data (only 5 Findex points)
- Event impact estimates based on comparable markets
- Assumes stable macroeconomic conditions

---
*Report generated by Ethiopia Financial Inclusion Forecasting System*
"""
        
        # Save report
        import os
        report_dir = "reports/task4"
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/final_forecasting_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")
        return report
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from forecasts"""
        insights = []
        
        # Account ownership insights
        if 'ACC_OWNERSHIP' in self.forecasts:
            acc_forecast = self.forecasts['ACC_OWNERSHIP']['ensemble']['forecasts'].get(2027, 0)
            insights.append(f"Account ownership expected to reach {acc_forecast:.1f}% by 2027")
        
        # Target gap insights
        if 'ACC_OWNERSHIP' in self.forecasts and 'target_gap' in self.forecasts['ACC_OWNERSHIP']:
            gap_2025 = self.forecasts['ACC_OWNERSHIP']['target_gap'].get(2025, {}).get('gap_pp', 0)
            if gap_2025 > 0:
                insights.append(f"NFIS-II 2025 target gap: {gap_2025:.1f}pp - acceleration needed")
        
        # Growth insights
        if 'ACC_OWNERSHIP' in self.forecasts and 'growth_rate' in self.forecasts['ACC_OWNERSHIP']:
            growth = self.forecasts['ACC_OWNERSHIP']['growth_rate']
            if growth:
                avg_growth = np.mean(list(growth.values()))
                insights.append(f"Average annual growth: {avg_growth:.2f}pp/year")
        
        return insights