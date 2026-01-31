"""
Evidence Analyzer - Comparable country evidence
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class EvidenceAnalyzer:
    """Analyze comparable country evidence for impact estimation"""
    
    # Evidence database from research
    COUNTRY_EVIDENCE = {
        'mobile_money_launch': {
            'Kenya': {'impact_1yr': 5.2, 'impact_3yr': 12.8, 'confidence': 'high'},
            'Tanzania': {'impact_1yr': 3.8, 'impact_3yr': 9.5, 'confidence': 'high'},
            'Ghana': {'impact_1yr': 2.5, 'impact_3yr': 6.2, 'confidence': 'medium'}
        },
        'interoperability': {
            'Kenya': {'impact_1yr': 2.1, 'impact_3yr': 4.3, 'confidence': 'medium'},
            'Tanzania': {'impact_1yr': 1.8, 'impact_3yr': 3.9, 'confidence': 'medium'},
            'India': {'impact_1yr': 3.5, 'impact_3yr': 8.2, 'confidence': 'high'}
        },
        'agent_network_expansion': {
            'Kenya': {'impact_per_10k_agents': 0.8, 'confidence': 'medium'},
            'Tanzania': {'impact_per_10k_agents': 0.6, 'confidence': 'medium'}
        }
    }
    
    def get_comparable_evidence(self, event_type: str, 
                               countries: List[str] = None) -> Dict:
        """Get evidence from comparable countries"""
        if event_type not in self.COUNTRY_EVIDENCE:
            return {'error': f'No evidence for {event_type}'}
        
        evidence = self.COUNTRY_EVIDENCE[event_type]
        
        if countries:
            evidence = {k: v for k, v in evidence.items() if k in countries}
        
        # Calculate averages
        impacts = []
        for country_data in evidence.values():
            if 'impact_3yr' in country_data:
                impacts.append(country_data['impact_3yr'])
            elif 'impact_1yr' in country_data:
                impacts.append(country_data['impact_1yr'] * 3)
        
        if impacts:
            avg_impact = np.mean(impacts)
            std_impact = np.std(impacts)
            
            return {
                'comparable_countries': list(evidence.keys()),
                'average_impact_3yr': avg_impact,
                'impact_range': (avg_impact - std_impact, avg_impact + std_impact),
                'country_evidence': evidence,
                'ethiopia_adjustment_factor': 0.8,  # Ethiopia-specific adjustment
                'recommended_impact': avg_impact * 0.8  # Adjusted for Ethiopia
            }
        
        return {'error': 'No impact data available'}
    
    def validate_impact_estimate(self, event_type: str, 
                                estimated_impact: float) -> Dict:
        """Validate impact estimate against comparable evidence"""
        evidence = self.get_comparable_evidence(event_type)
        
        if 'error' in evidence:
            return {'validation': 'NO_EVIDENCE', 'confidence': 'low'}
        
        lower, upper = evidence['impact_range']
        recommended = evidence['recommended_impact']
        
        if lower <= estimated_impact <= upper:
            return {
                'validation': 'WITHIN_RANGE',
                'confidence': 'high',
                'estimated': estimated_impact,
                'recommended_range': (lower, upper),
                'deviation': estimated_impact - recommended
            }
        else:
            return {
                'validation': 'OUTSIDE_RANGE',
                'confidence': 'medium',
                'estimated': estimated_impact,
                'recommended_range': (lower, upper),
                'deviation': estimated_impact - recommended,
                'suggestion': f'Consider adjusting to {recommended:.1f}pp'
            }