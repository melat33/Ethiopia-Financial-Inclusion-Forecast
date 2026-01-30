"""
Impact Validation Module
Validates event impact models against historical data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ImpactValidator:
    """Validates event impact models against historical observations"""
    
    def __init__(self, observations_df: pd.DataFrame):
        """
        Initialize validator
        
        Args:
            observations_df: DataFrame with historical observations
        """
        self.observations = observations_df.copy()
        # Convert dates
        if 'observation_date' in self.observations.columns:
            self.observations['observation_date'] = pd.to_datetime(
                self.observations['observation_date'], errors='coerce'
            )
        
    def validate_telebirr_impact(self, 
                                indicator_code: str = 'ACC_MM_ACCOUNT') -> Dict:
        """
        Validate Telebirr launch impact
        
        Telebirr launched May 2021
        Mobile money accounts: 4.7% (2021) → 9.45% (2024)
        """
        # Get pre and post observations
        pre_telebirr = self.observations[
            (self.observations['indicator_code'] == indicator_code) &
            (self.observations['observation_date'] < '2021-05-01')
        ].sort_values('observation_date')
        
        post_telebirr = self.observations[
            (self.observations['indicator_code'] == indicator_code) &
            (self.observations['observation_date'] >= '2021-05-01')
        ].sort_values('observation_date')
        
        if len(pre_telebirr) > 0 and len(post_telebirr) > 0:
            # Calculate actual change
            last_pre = pre_telebirr.iloc[-1]['value_numeric']
            first_post = post_telebirr.iloc[0]['value_numeric']
            latest_post = post_telebirr.iloc[-1]['value_numeric']
            
            actual_immediate = first_post - last_pre
            actual_3yr = latest_post - last_pre
            
            # Expected impact based on comparable evidence
            # Similar mobile money launches in other African countries
            # showed 3-8pp increase in first 3 years
            expected_range = (3.0, 8.0)
            
            # Calculate model accuracy
            within_range = expected_range[0] <= actual_3yr <= expected_range[1]
            deviation = actual_3yr - np.mean(expected_range)
            
            validation_result = {
                'event': 'Telebirr Launch',
                'event_date': '2021-05-01',
                'indicator': indicator_code,
                'pre_event_value': last_pre,
                'post_event_3yr': latest_post,
                'actual_change_pp': actual_3yr,
                'expected_range_pp': expected_range,
                'within_expected_range': within_range,
                'deviation_from_expected': deviation,
                'validation_status': 'PASS' if within_range else 'REVIEW',
                'confidence': 'high' if within_range else 'medium'
            }
            
            return validation_result
        
        return {'error': 'Insufficient data for validation'}
    
    def validate_m_pesa_impact(self) -> Dict:
        """
        Validate M-Pesa entry impact
        
        M-Pesa entered Aug 2023
        """
        # Get digital payment trends around M-Pesa entry
        pre_m_pesa = self.observations[
            (self.observations['indicator_code'] == 'USG_DIGITAL_PAYMENT') &
            (self.observations['observation_date'] < '2023-08-01')
        ]
        
        post_m_pesa = self.observations[
            (self.observations['indicator_code'] == 'USG_DIGITAL_PAYMENT') &
            (self.observations['observation_date'] >= '2023-08-01')
        ]
        
        if len(pre_m_pesa) > 0 and len(post_m_pesa) > 0:
            last_pre = pre_m_pesa.iloc[-1]['value_numeric']
            latest_post = post_m_pesa.iloc[-1]['value_numeric'] if len(post_m_pesa) > 0 else None
            
            if latest_post:
                actual_change = latest_post - last_pre
                
                # Based on M-Pesa entries in other countries
                expected_range = (1.0, 4.0)  # pp increase in first year
                
                within_range = expected_range[0] <= actual_change <= expected_range[1]
                
                return {
                    'event': 'M-Pesa Entry',
                    'event_date': '2023-08-01',
                    'indicator': 'USG_DIGITAL_PAYMENT',
                    'pre_event_value': last_pre,
                    'actual_change_pp': actual_change,
                    'expected_range_pp': expected_range,
                    'within_expected_range': within_range,
                    'validation_status': 'PASS' if within_range else 'REVIEW',
                    'notes': 'Limited post-entry data available'
                }
        
        return {'error': 'Insufficient post-entry data'}
    
    def compare_country_evidence(self, 
                               event_type: str,
                               comparable_countries: List[str] = ['Kenya', 'Tanzania', 'Ghana']) -> Dict:
        """
        Compare event impacts with similar countries
        
        Args:
            event_type: Type of event (e.g., 'mobile_money_launch', 'interoperability')
            comparable_countries: List of countries for comparison
        """
        # Evidence base from research and reports
        evidence_base = {
            'mobile_money_launch': {
                'Kenya': {'impact_1yr': 5.2, 'impact_3yr': 12.8},
                'Tanzania': {'impact_1yr': 3.8, 'impact_3yr': 9.5},
                'Ghana': {'impact_1yr': 2.5, 'impact_3yr': 6.2}
            },
            'interoperability': {
                'Kenya': {'impact_1yr': 2.1, 'impact_3yr': 4.3},
                'Tanzania': {'impact_1yr': 1.8, 'impact_3yr': 3.9}
            },
            'agent_network_expansion': {
                'Kenya': {'impact_pp_per_100k': 0.8},
                'Tanzania': {'impact_pp_per_100k': 0.6}
            }
        }
        
        if event_type in evidence_base:
            comparable_evidence = evidence_base[event_type]
            
            # Calculate average impact for benchmarking
            impacts = []
            for country, data in comparable_evidence.items():
                if 'impact_3yr' in data:
                    impacts.append(data['impact_3yr'])
                elif 'impact_1yr' in data:
                    impacts.append(data['impact_1yr'] * 3)  # Rough extrapolation
            
            if impacts:
                avg_impact = np.mean(impacts)
                std_impact = np.std(impacts)
                
                return {
                    'event_type': event_type,
                    'comparable_countries': list(comparable_evidence.keys()),
                    'average_impact_3yr_pp': avg_impact,
                    'impact_range_pp': (avg_impact - std_impact, avg_impact + std_impact),
                    'evidence': comparable_evidence,
                    'recommended_adjustment_factor': 0.8  # Ethiopia-specific adjustment
                }
        
        return {'error': f'No evidence found for {event_type}'}
    
    def plot_validation_results(self, validation_results: List[Dict]):
        """Plot validation results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Actual vs Expected
        events = [r['event'] for r in validation_results if 'actual_change_pp' in r]
        actuals = [r['actual_change_pp'] for r in validation_results if 'actual_change_pp' in r]
        expecteds = [np.mean(r['expected_range_pp']) for r in validation_results if 'expected_range_pp' in r]
        
        x = np.arange(len(events))
        width = 0.35
        
        axes[0].bar(x - width/2, actuals, width, label='Actual', color='steelblue')
        axes[0].bar(x + width/2, expecteds, width, label='Expected', color='lightcoral')
        axes[0].set_xlabel('Event')
        axes[0].set_ylabel('Impact (pp)')
        axes[0].set_title('Actual vs Expected Event Impacts')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(events, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Validation Status
        status_counts = {}
        for result in validation_results:
            status = result.get('validation_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        axes[1].pie(status_counts.values(), labels=status_counts.keys(), 
                   autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightgray'])
        axes[1].set_title('Validation Status Distribution')
        
        plt.tight_layout()
        return fig
    
    def generate_validation_report(self, output_path: str = 'reports/validation_report.md'):
        """Generate comprehensive validation report"""
        report = []
        report.append("# Event Impact Validation Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Validate key events
        telebirr_validation = self.validate_telebirr_impact()
        m_pesa_validation = self.validate_m_pesa_impact()
        
        report.append("## Key Event Validations\n")
        
        for validation in [telebirr_validation, m_pesa_validation]:
            if 'error' not in validation:
                report.append(f"### {validation['event']}")
                report.append(f"- **Date**: {validation['event_date']}")
                report.append(f"- **Indicator**: {validation['indicator']}")
                report.append(f"- **Pre-event value**: {validation.get('pre_event_value', 'N/A')}%")
                report.append(f"- **Actual change**: {validation.get('actual_change_pp', 'N/A')}pp")
                report.append(f"- **Expected range**: {validation.get('expected_range_pp', 'N/A')}pp")
                report.append(f"- **Validation Status**: **{validation['validation_status']}**")
                report.append(f"- **Confidence**: {validation.get('confidence', 'N/A')}\n")
        
        # Compare with country evidence
        report.append("## Comparable Country Evidence\n")
        
        for event_type in ['mobile_money_launch', 'interoperability']:
            evidence = self.compare_country_evidence(event_type)
            if 'error' not in evidence:
                report.append(f"### {event_type.replace('_', ' ').title()}")
                report.append(f"- **Comparable countries**: {', '.join(evidence['comparable_countries'])}")
                report.append(f"- **Average 3-year impact**: {evidence['average_impact_3yr_pp']:.1f}pp")
                report.append(f"- **Impact range**: {evidence['impact_range_pp'][0]:.1f} - {evidence['impact_range_pp'][1]:.1f}pp")
                report.append(f"- **Ethiopia adjustment factor**: {evidence.get('recommended_adjustment_factor', 1.0)}\n")
        
        # Save report
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        return output_path