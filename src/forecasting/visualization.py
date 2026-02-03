"""
Professional visualizations for forecasting results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ForecastVisualizer:
    """Create professional forecasting visualizations"""
    
    COLORS = {
        'pessimistic': '#FF6B6B',
        'baseline': '#4ECDC4',
        'optimistic': '#45B7D1',
        'historical': '#2E86AB',
        'target': '#A23B72'
    }
    
    def __init__(self, output_dir: str = 'reports/figures/task4'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comprehensive_dashboard(self, 
                                     forecast_results: Dict,
                                     scenario_analysis: Dict,
                                     uncertainty_analysis: Dict):
        """Create complete dashboard of visualizations"""
        
        print("📊 Generating professional visualizations...")
        
        # 1. Main forecast plot
        self._create_forecast_comparison_plot(forecast_results, scenario_analysis)
        
        # 2. Uncertainty fan chart
        self._create_uncertainty_fan_chart(scenario_analysis, uncertainty_analysis)
        
        # 3. Scenario comparison
        self._create_scenario_comparison_chart(scenario_analysis)
        
        # 4. Target gap analysis
        self._create_target_gap_chart(forecast_results)
        
        # 5. Uncertainty decomposition
        self._create_uncertainty_decomposition(uncertainty_analysis)
        
        # 6. Sensitivity analysis
        self._create_sensitivity_heatmap(uncertainty_analysis)
        
        print(f"✅ All visualizations saved to: {self.output_dir}")
    
    def _create_forecast_comparison_plot(self, forecast_results: Dict, 
                                       scenario_analysis: Dict):
        """Create main forecast comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        indicators = ['ACC_OWNERSHIP', 'USG_DIGITAL_PAYMENT']
        titles = ['Account Ownership Forecast', 'Digital Payment Forecast']
        
        for idx, (indicator, title) in enumerate(zip(indicators, titles)):
            ax = axes[idx]
            
            # Historical data
            historical_years = [2011, 2014, 2017, 2021, 2024]
            historical_values = [14, 22, 35, 46, 49] if indicator == 'ACC_OWNERSHIP' else [10, 18, 25, 35]
            
            if indicator == 'USG_DIGITAL_PAYMENT':
                historical_years = historical_years[1:]  # Start from 2014
            
            ax.plot(historical_years, historical_values, 'ko-', 
                   linewidth=2, markersize=8, label='Historical')
            
            # Forecast scenarios
            if indicator in scenario_analysis:
                scenarios = scenario_analysis[indicator]
                forecast_years = [2025, 2026, 2027]
                
                for scenario_name in ['pessimistic', 'baseline', 'optimistic']:
                    if scenario_name in scenarios:
                        values = [scenarios[scenario_name]['forecasts'].get(year, np.nan) 
                                 for year in forecast_years]
                        ax.plot(forecast_years, values, 'o-',
                               color=self.COLORS[scenario_name],
                               linewidth=2, markersize=6,
                               label=scenario_name.title())
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('% of Adults', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/forecast_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_uncertainty_fan_chart(self, scenario_analysis: Dict, 
                                    uncertainty_analysis: Dict):
        """Create uncertainty fan chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        indicator = 'ACC_OWNERSHIP'
        if indicator in scenario_analysis and indicator in uncertainty_analysis:
            
            # Extract data
            scenarios = scenario_analysis[indicator]
            uncertainty = uncertainty_analysis[indicator]
            
            # Prepare data for fan chart
            years = [2025, 2026, 2027]
            percentiles = [5, 25, 75, 95]
            
            # Get values for each percentile
            if 'monte_carlo' in uncertainty:
                mc_results = uncertainty['monte_carlo']
                
                # Create fan
                ax.fill_between(years, 
                               [mc_results[year]['p5'] for year in years],
                               [mc_results[year]['p95'] for year in years],
                               color=self.COLORS['baseline'], alpha=0.1,
                               label='90% Confidence Interval')
                
                ax.fill_between(years,
                               [mc_results[year]['p25'] for year in years],
                               [mc_results[year]['p75'] for year in years],
                               color=self.COLORS['baseline'], alpha=0.2,
                               label='50% Confidence Interval')
                
                # Plot mean
                mean_values = [mc_results[year]['mean_sim'] for year in years]
                ax.plot(years, mean_values, 'k-', linewidth=3, 
                       label='Mean Forecast', marker='o')
            
            ax.set_title('Account Ownership Forecast with Uncertainty Bands', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Account Ownership (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax.set_ylim(40, 80)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/uncertainty_fan_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scenario_comparison_chart(self, scenario_analysis: Dict):
        """Create scenario comparison chart"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        indicators = ['ACC_OWNERSHIP', 'USG_DIGITAL_PAYMENT']
        
        for idx, indicator in enumerate(indicators):
            ax = axes[idx]
            
            if indicator in scenario_analysis:
                scenarios = scenario_analysis[indicator]
                
                # Prepare data for 2027
                scenario_names = ['Pessimistic', 'Baseline', 'Optimistic']
                values_2027 = [scenarios[scen]['forecasts'].get(2027, 0) 
                              for scen in ['pessimistic', 'baseline', 'optimistic']]
                colors = [self.COLORS[scen] for scen in ['pessimistic', 'baseline', 'optimistic']]
                
                bars = ax.bar(scenario_names, values_2027, color=colors, alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}%', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
                
                ax.set_title(f'{indicator.replace("_", " ")} - 2027 Scenarios', 
                           fontsize=13, fontweight='bold')
                ax.set_ylabel('% of Adults', fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_target_gap_chart(self, forecast_results: Dict):
        """Create NFIS-II target gap chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'ACC_OWNERSHIP' in forecast_results and 'target_gap' in forecast_results['ACC_OWNERSHIP']:
            gaps = forecast_results['ACC_OWNERSHIP']['target_gap']
            
            years = list(gaps.keys())
            targets = [gaps[year]['target'] for year in years]
            forecasts = [gaps[year]['forecast'] for year in years]
            
            x = np.arange(len(years))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, targets, width, 
                          label='NFIS-II Target', color=self.COLORS['target'], alpha=0.8)
            bars2 = ax.bar(x + width/2, forecasts, width, 
                          label='Baseline Forecast', color=self.COLORS['baseline'], alpha=0.8)
            
            # Add gap labels
            for i, (target, forecast) in enumerate(zip(targets, forecasts)):
                gap = target - forecast
                ax.text(i, max(target, forecast) + 2, 
                       f'Gap: {gap:.1f}pp', ha='center', fontweight='bold')
            
            ax.set_title('NFIS-II Target Gap Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Account Ownership (%)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(years)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/target_gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_uncertainty_decomposition(self, uncertainty_analysis: Dict):
        """Create uncertainty decomposition pie chart"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if uncertainty_analysis:
            # Use first indicator's decomposition
            for indicator in uncertainty_analysis.keys():
                if 'uncertainty_decomposition' in uncertainty_analysis[indicator]:
                    decomposition = uncertainty_analysis[indicator]['uncertainty_decomposition']
                    
                    labels = list(decomposition.keys())
                    sizes = list(decomposition.values())
                    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                    
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                                     autopct='%1.1f%%', startangle=90)
                    
                    # Improve text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('Uncertainty Source Decomposition', 
                               fontsize=14, fontweight='bold')
                    
                    break  # Only plot first indicator
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/uncertainty_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sensitivity_heatmap(self, uncertainty_analysis: Dict):
        """Create sensitivity analysis heatmap"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if uncertainty_analysis:
            for indicator in uncertainty_analysis.keys():
                if 'sensitivity_analysis' in uncertainty_analysis[indicator]:
                    sensitivity = uncertainty_analysis[indicator]['sensitivity_analysis']
                    
                    # Prepare heatmap data
                    parameters = list(sensitivity.keys())
                    param_values = []
                    impacts = []
                    
                    for param in parameters:
                        values = list(sensitivity[param].keys())
                        impacts_row = [sensitivity[param][v]['impact_pp'] for v in values]
                        param_values.append(values)
                        impacts.append(impacts_row)
                    
                    # Create heatmap
                    if impacts:
                        im = ax.imshow(impacts, cmap='RdYlBu', aspect='auto')
                        
                        # Customize
                        ax.set_xticks(range(len(param_values[0])))
                        ax.set_yticks(range(len(parameters)))
                        ax.set_xticklabels(param_values[0] if param_values else [])
                        ax.set_yticklabels(parameters)
                        ax.set_title('Parameter Sensitivity Analysis', 
                                   fontsize=14, fontweight='bold')
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label('Impact (Percentage Points)', fontsize=11)
                        
                        # Add value annotations
                        for i in range(len(parameters)):
                            for j in range(len(param_values[0])):
                                text = ax.text(j, i, f'{impacts[i][j]:+.1f}',
                                             ha="center", va="center", 
                                             color="black", fontweight='bold')
                        
                        break  # Only plot first indicator
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()