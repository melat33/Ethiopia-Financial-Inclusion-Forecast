"""
Evidence Analyzer for Ethiopia Financial Inclusion
Analyzes comparable country evidence for event impacts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class EvidenceAnalyzer:
    """
    Analyzes comparable country evidence for event impact estimation.
    
    This class provides evidence-based impact estimates for events
    where Ethiopian data is insufficient.
    """
    
    def __init__(self):
        """Initialize the evidence analyzer."""
        self.evidence_database = self._initialize_evidence_database()
        self.ethiopia_adjustment_factors = self._get_ethiopia_adjustment_factors()
    
    def _initialize_evidence_database(self) -> Dict:
        """Initialize the evidence database with international studies."""
        return {
            'mobile_money_launch': {
                'description': 'Impact of mobile money service launch',
                'studies': [
                    {
                        'country': 'Kenya',
                        'event': 'M-Pesa Launch (2007)',
                        'impact_3yr': 11.2,
                        'impact_5yr': 24.8,
                        'lag_months': 6,
                        'confidence': 'HIGH',
                        'source': 'GSMA, World Bank'
                    },
                    {
                        'country': 'Tanzania',
                        'event': 'M-Pesa Launch (2008)',
                        'impact_3yr': 8.7,
                        'impact_5yr': 18.9,
                        'lag_months': 9,
                        'confidence': 'HIGH',
                        'source': 'IMF, CGAP'
                    },
                    {
                        'country': 'Uganda',
                        'event': 'MTN Mobile Money (2009)',
                        'impact_3yr': 7.9,
                        'impact_5yr': 16.5,
                        'lag_months': 12,
                        'confidence': 'MEDIUM',
                        'source': 'World Bank'
                    }
                ],
                'summary_statistics': {
                    'average_3yr': 9.27,
                    'median_3yr': 8.70,
                    'std_3yr': 1.68,
                    'min_3yr': 7.90,
                    'max_3yr': 11.20
                }
            },
            'digital_id_implementation': {
                'description': 'Impact of national digital ID rollout',
                'studies': [
                    {
                        'country': 'India',
                        'event': 'Aadhaar Rollout',
                        'impact_2yr': 4.8,
                        'impact_5yr': 9.2,
                        'confidence': 'HIGH',
                        'source': 'World Bank ID4D'
                    },
                    {
                        'country': 'Pakistan',
                        'event': 'NADRA Implementation',
                        'impact_2yr': 3.2,
                        'impact_5yr': 6.1,
                        'confidence': 'MEDIUM',
                        'source': 'IMF'
                    }
                ]
            },
            'payment_interoperability': {
                'description': 'Impact of payment system interoperability',
                'studies': [
                    {
                        'country': 'Nigeria',
                        'event': 'NIP Implementation',
                        'impact_1yr': 2.1,
                        'impact_3yr': 4.3,
                        'confidence': 'HIGH',
                        'source': 'CBN, BIS'
                    }
                ]
            }
        }
    
    def _get_ethiopia_adjustment_factors(self) -> Dict[str, float]:
        """Get adjustment factors for Ethiopia context."""
        return {
            'market_maturity': 0.8,      # Less mature than Kenya
            'digital_literacy': 0.7,      # Lower digital literacy
            'population_density': 1.1,    # Higher population density
            'regulatory_environment': 0.9, # Evolving regulatory framework
            'infrastructure': 0.8,        # Developing infrastructure
            'financial_depth': 0.7        # Lower financial depth
        }
    
    def get_evidence_for_event(self, event_type: str, 
                              impact_horizon: str = '3yr') -> Dict:
        """
        Get evidence for a specific event type.
        
        Args:
            event_type: Type of event (e.g., 'mobile_money_launch')
            impact_horizon: Time horizon for impact ('1yr', '3yr', '5yr')
            
        Returns:
            Dictionary with evidence analysis
        """
        if event_type not in self.evidence_database:
            return {
                'error': f'No evidence found for event type: {event_type}',
                'available_events': list(self.evidence_database.keys())
            }
        
        event_data = self.evidence_database[event_type]
        studies = event_data.get('studies', [])
        
        if not studies:
            return {'error': 'No studies available for this event type'}
        
        # Extract impacts for the specified horizon
        impacts = []
        for study in studies:
            impact_key = f'impact_{impact_horizon}'
            if impact_key in study:
                impacts.append(study[impact_key])
        
        if not impacts:
            return {'error': f'No {impact_horizon} impact data available'}
        
        # Calculate statistics
        impacts_array = np.array(impacts)
        
        analysis = {
            'event_type': event_type,
            'impact_horizon': impact_horizon,
            'number_of_studies': len(studies),
            'countries_studied': [study['country'] for study in studies],
            'impact_statistics': {
                'mean': float(np.mean(impacts_array)),
                'median': float(np.median(impacts_array)),
                'std': float(np.std(impacts_array)),
                'min': float(np.min(impacts_array)),
                'max': float(np.max(impacts_array)),
                'q25': float(np.percentile(impacts_array, 25)),
                'q75': float(np.percentile(impacts_array, 75))
            },
            'ethiopia_adjusted': self._adjust_for_ethiopia(
                np.mean(impacts_array), event_type
            ),
            'recommended_range': self._calculate_recommended_range(impacts_array),
            'confidence_assessment': self._assess_confidence(studies),
            'studies_summary': self._summarize_studies(studies)
        }
        
        return analysis
    
    def _adjust_for_ethiopia(self, international_mean: float, 
                           event_type: str) -> float:
        """
        Adjust international evidence for Ethiopia context.
        
        Args:
            international_mean: Mean impact from international evidence
            event_type: Type of event
            
        Returns:
            Ethiopia-adjusted impact estimate
        """
        # Apply adjustment factors
        adjusted = international_mean
        
        for factor_name, factor_value in self.ethiopia_adjustment_factors.items():
            adjusted *= factor_value
        
        # Event-specific adjustments
        if event_type == 'mobile_money_launch':
            # Ethiopia launched later than comparator countries
            adjusted *= 0.9  # Later market entry
        
        return round(adjusted, 2)
    
    def _calculate_recommended_range(self, impacts: np.ndarray) -> Tuple[float, float]:
        """Calculate recommended impact range."""
        mean = np.mean(impacts)
        std = np.std(impacts)
        
        # Use interquartile range for robustness
        q25 = np.percentile(impacts, 25)
        q75 = np.percentile(impacts, 75)
        
        # Recommended range: 25th to 75th percentile
        lower_bound = q25 * 0.8  # Conservative lower bound
        upper_bound = q75 * 1.2  # Optimistic upper bound
        
        return (round(lower_bound, 2), round(upper_bound, 2))
    
    def _assess_confidence(self, studies: List[Dict]) -> str:
        """Assess overall confidence in evidence."""
        confidences = [study.get('confidence', 'LOW') for study in studies]
        
        high_count = confidences.count('HIGH')
        medium_count = confidences.count('MEDIUM')
        total = len(confidences)
        
        if high_count / total >= 0.7:
            return 'HIGH'
        elif (high_count + medium_count) / total >= 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _summarize_studies(self, studies: List[Dict]) -> List[Dict]:
        """Create summary of individual studies."""
        summaries = []
        
        for study in studies:
            summary = {
                'country': study['country'],
                'event': study['event'],
                'confidence': study.get('confidence', 'UNKNOWN'),
                'source': study.get('source', 'Unknown'),
                'key_finding': f"Impact: {study.get('impact_3yr', 'N/A')}pp over 3 years"
            }
            summaries.append(summary)
        
        return summaries
    
    def validate_estimate(self, event_type: str, estimate: float, 
                         horizon: str = '3yr') -> Dict:
        """
        Validate an impact estimate against evidence.
        
        Args:
            event_type: Type of event
            estimate: Impact estimate to validate
            horizon: Time horizon for comparison
            
        Returns:
            Validation results
        """
        evidence = self.get_evidence_for_event(event_type, horizon)
        
        if 'error' in evidence:
            return {'validation': 'UNABLE_TO_VALIDATE', 'reason': evidence['error']}
        
        stats = evidence['impact_statistics']
        recommended_range = evidence['recommended_range']
        
        # Check if estimate is within reasonable range
        if recommended_range[0] <= estimate <= recommended_range[1]:
            validation = 'WITHIN_EXPECTED_RANGE'
            confidence = 'HIGH'
        elif stats['min'] <= estimate <= stats['max']:
            validation = 'WITHIN_OBSERVED_RANGE'
            confidence = 'MEDIUM'
        else:
            validation = 'OUTSIDE_EXPECTED_RANGE'
            confidence = 'LOW'
        
        # Calculate deviation from evidence
        deviation = abs(estimate - stats['mean'])
        relative_deviation = (deviation / stats['mean']) * 100 if stats['mean'] > 0 else 100
        
        return {
            'validation': validation,
            'confidence': confidence,
            'estimate': estimate,
            'evidence_mean': stats['mean'],
            'deviation_pp': round(deviation, 2),
            'relative_deviation_percent': round(relative_deviation, 1),
            'recommended_range': recommended_range,
            'observed_range': (stats['min'], stats['max']),
            'ethiopia_adjusted': evidence['ethiopia_adjusted'],
            'recommendation': self._generate_validation_recommendation(
                validation, estimate, stats['mean'], recommended_range
            )
        }
    
    def _generate_validation_recommendation(self, validation: str, 
                                          estimate: float, evidence_mean: float,
                                          recommended_range: Tuple[float, float]) -> str:
        """Generate recommendation based on validation results."""
        if validation == 'WITHIN_EXPECTED_RANGE':
            return "Estimate is well-supported by evidence"
        
        elif validation == 'WITHIN_OBSERVED_RANGE':
            return "Estimate is plausible but at the edge of observed range"
        
        else:  # OUTSIDE_EXPECTED_RANGE
            if estimate < recommended_range[0]:
                adjustment = recommended_range[0] - estimate
                return f"Consider increasing estimate by {adjustment:.1f}pp"
            else:
                adjustment = estimate - recommended_range[1]
                return f"Consider decreasing estimate by {adjustment:.1f}pp"
    
    def generate_evidence_report(self, event_types: List[str] = None) -> Dict:
        """
        Generate comprehensive evidence report.
        
        Args:
            event_types: List of event types to include
            
        Returns:
            Comprehensive evidence report
        """
        if event_types is None:
            event_types = list(self.evidence_database.keys())
        
        report = {
            'evidence_summary': {},
            'methodology': {
                'adjustment_factors': self.ethiopia_adjustment_factors,
                'range_calculation': 'Interquartile range with conservative bounds',
                'confidence_assessment': 'Based on study quality and consistency'
            },
            'key_insights': []
        }
        
        for event_type in event_types:
            if event_type in self.evidence_database:
                evidence_3yr = self.get_evidence_for_event(event_type, '3yr')
                
                if 'error' not in evidence_3yr:
                    report['evidence_summary'][event_type] = {
                        'description': self.evidence_database[event_type]['description'],
                        'international_mean': evidence_3yr['impact_statistics']['mean'],
                        'ethiopia_adjusted': evidence_3yr['ethiopia_adjusted'],
                        'recommended_range': evidence_3yr['recommended_range'],
                        'confidence': evidence_3yr['confidence_assessment'],
                        'studies_count': evidence_3yr['number_of_studies']
                    }
                    
                    # Add key insight
                    insight = (
                        f"{event_type}: International evidence suggests "
                        f"{evidence_3yr['impact_statistics']['mean']:.1f}pp impact "
                        f"over 3 years, adjusted to {evidence_3yr['ethiopia_adjusted']:.1f}pp "
                        f"for Ethiopia context"
                    )
                    report['key_insights'].append(insight)
        
        return report