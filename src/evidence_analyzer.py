"""
Evidence Analyzer - Comparable country evidence analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class EvidenceAnalyzer:
    """Analyze and apply comparable country evidence"""
    
    def __init__(self):
        self.evidence_db = self._load_evidence_database()
    
    def _load_evidence_database(self) -> Dict:
        """Load evidence database"""
        return {
            'mobile_money_launch': {
                'Kenya (M-Pesa 2007)': {
                    'impact_1yr': 5.2,
                    'impact_3yr': 12.8,
                    'pre_launch': 26.0,
                    'post_3yr': 38.8,
                    'source': 'World Bank Findex',
                    'confidence': 'high'
                },
                'Tanzania (Vodacom 2008)': {
                    'impact_1yr': 3.8,
                    'impact_3yr': 9.5,
                    'pre_launch': 17.0,
                    'post_3yr': 26.5,
                    'source': 'GSMA Report',
                    'confidence': 'high'
                },
                'Ghana (MTN 2009)': {
                    'impact_1yr': 2.5,
                    'impact_3yr': 6.2,
                    'pre_launch': 29.0,
                    'post_3yr': 35.2,
                    'source': 'CGAP Study',
                    'confidence': 'medium'
                }
            },
            'market_entry_competition': {
                'Kenya (Airtel 2010)': {
                    'impact_1yr': 1.8,
                    'impact_3yr': 3.5,
                    'source': 'Central Bank of Kenya',
                    'confidence': 'medium'
                }
            },
            'regulatory_reform': {
                'India (PSP Licensing 2016)': {
                    'impact_2yr': 8.2,
                    'source': 'RBI Report',
                    'confidence': 'high'
                }
            }
        }
    
    def get_evidence_for_event(self, event_type: str, 
                              adjustment_factor: float = 0.8) -> Dict:
        """Get evidence for specific event type with Ethiopia adjustment"""
        if event_type not in self.evidence_db:
            return {'error': f'No evidence for {event_type}'}
        
        evidence = self.evidence_db[event_type]
        
        # Calculate statistics
        impacts_3yr = []
        for country, data in evidence.items():
            if 'impact_3yr' in data:
                impacts_3yr.append(data['impact_3yr'])
        
        if impacts_3yr:
            avg_impact = np.mean(impacts_3yr)
            std_impact = np.std(impacts_3yr)
            
            return {
                'event_type': event_type,
                'comparable_countries': list(evidence.keys()),
                'international_average_3yr': avg_impact,
                'international_std_3yr': std_impact,
                'ethiopia_adjusted_3yr': avg_impact * adjustment_factor,
                'confidence': evidence[list(evidence.keys())[0]]['confidence'],
                'adjustment_notes': f'Applied {adjustment_factor}x adjustment for Ethiopia context',
                'recommended_range': (
                    (avg_impact - std_impact) * adjustment_factor,
                    (avg_impact + std_impact) * adjustment_factor
                )
            }
        
        return {'error': 'Insufficient impact data'}
    
    def validate_estimate(self, event_type: str, estimate: float) -> Dict:
        """Validate impact estimate against evidence"""
        evidence = self.get_evidence_for_event(event_type)
        
        if 'error' in evidence:
            return {'validation': 'NO_EVIDENCE', 'confidence': 'low'}
        
        lower, upper = evidence['recommended_range']
        recommended = evidence['ethiopia_adjusted_3yr']
        
        if lower <= estimate <= upper:
            return {
                'validation': 'WITHIN_EXPECTED_RANGE',
                'confidence': 'high',
                'deviation_percent': ((estimate - recommended) / recommended) * 100
            }
        else:
            return {
                'validation': 'OUTSIDE_EXPECTED_RANGE',
                'confidence': 'medium',
                'deviation_percent': ((estimate - recommended) / recommended) * 100,
                'suggestion': f'Consider adjusting to {recommended:.1f}pp'
            }