# src/impact_validator.py
"""
Robust validation module for event impact modeling.
Validates model predictions against historical data and comparable evidence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RobustImpactValidator:
    """
    Validates event impact models against historical data and comparable evidence.
    """

    def __init__(self, df: pd.DataFrame, ref_codes: pd.DataFrame = None):
        """
        Initialize the validator.
        
        Args:
            df: DataFrame with historical data
            ref_codes: DataFrame with reference codes
        """
        self.df = df.copy()
        self.ref_codes = ref_codes
        
        # Validation thresholds
        self.thresholds = {
            'good_match': 3.0,  # pp difference
            'acceptable_match': 5.0,  # pp difference
            'min_data_points': 3,  # for reliable validation
            'coverage_threshold': 0.3,  # minimum data coverage
        }
        
        # Comparable evidence database (simplified)
        self.evidence_db = {
            'mobile_money_launch': {
                'impact_range': (8, 15),  # percentage points
                'duration': '3yr',
                'confidence': 'high',
                'sources': ['GSMA', 'World Bank', 'IMF']
            },
            'national_id_launch': {
                'impact_range': (5, 12),
                'duration': '2yr',
                'confidence': 'medium',
                'sources': ['World Bank']
            },
            'digital_finance_strategy': {
                'impact_range': (3, 8),
                'duration': '5yr',
                'confidence': 'medium',
                'sources': ['AFI', 'IMF']
            }
        }
        
    def validate_telebirr_impact(self, association_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Telebirr launch impact model against historical data.
        
        Args:
            association_matrix: DataFrame with event impact estimates
            
        Returns:
            Dictionary with validation results
        """
        print("🔍 Validating Telebirr launch impact...")
        
        # Find Telebirr event in association matrix
        telebirr_event = association_matrix[
            association_matrix['event_name'].str.contains('telebirr', case=False, na=False)
        ]
        
        if telebirr_event.empty:
            return {
                'status': 'error',
                'message': 'Telebirr event not found in association matrix',
                'validation': None
            }
        
        # Get predicted impacts
        predicted_impacts = {}
        for col in telebirr_event.columns:
            if '_impact' in col and pd.notna(telebirr_event[col].iloc[0]):
                predicted_impacts[col.replace('_impact', '')] = telebirr_event[col].iloc[0]
        
        # Calculate actual changes from historical data
        actual_changes = self._calculate_actual_changes(
            event_name='Telebirr Launch',
            event_date='2021-05-01',
            indicators=list(predicted_impacts.keys())
        )
        
        # Compare predictions vs actuals
        validation_results = []
        total_difference = 0
        validated_count = 0
        
        for indicator, predicted_impact in predicted_impacts.items():
            if indicator in actual_changes:
                actual_change = actual_changes[indicator]['change_pp']
                difference = abs(predicted_impact - actual_change)
                
                validation_status = self._evaluate_match(
                    predicted_impact, 
                    actual_change, 
                    difference
                )
                
                validation_results.append({
                    'indicator': indicator,
                    'predicted_impact_pp': predicted_impact,
                    'actual_change_pp': actual_change,
                    'difference_pp': difference,
                    'status': validation_status,
                    'confidence': actual_changes[indicator]['confidence']
                })
                
                total_difference += difference
                validated_count += 1
        
        # Calculate overall validation score
        if validated_count > 0:
            avg_difference = total_difference / validated_count
            overall_status = self._determine_overall_status(avg_difference)
            
            validation_summary = {
                'status': overall_status,
                'avg_difference_pp': avg_difference,
                'validated_indicators': validated_count,
                'total_indicators': len(predicted_impacts),
                'validation_rate': validated_count / len(predicted_impacts)
            }
        else:
            validation_summary = {
                'status': 'no_data',
                'avg_difference_pp': None,
                'validated_indicators': 0,
                'total_indicators': len(predicted_impacts),
                'validation_rate': 0
            }
        
        return {
            'status': 'success',
            'validation': validation_summary,
            'detailed_results': validation_results,
            'recommendations': self._generate_validation_recommendations(
                validation_summary['status'],
                validation_summary['avg_difference_pp'] if validated_count > 0 else None,
                actual_changes,
                predicted_impacts
            )
        }
    
    def _calculate_actual_changes(self, event_name: str, event_date: str, 
                                indicators: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate actual changes from historical data for validation.
        
        Args:
            event_name: Name of the event
            event_date: Date of the event
            indicators: List of indicators to validate
            
        Returns:
            Dictionary with actual changes for each indicator
        """
        actual_changes = {}
        event_date = pd.to_datetime(event_date)
        
        for indicator in indicators:
            if indicator not in self.df.columns:
                continue
            
            # Get data before and after event
            mask_before = self.df['date'] < event_date
            mask_after = self.df['date'] > event_date
            
            before_data = self.df.loc[mask_before, indicator].dropna()
            after_data = self.df.loc[mask_after, indicator].dropna()
            
            # Check if we have enough data
            if len(before_data) >= self.thresholds['min_data_points'] and \
               len(after_data) >= self.thresholds['min_data_points']:
                
                before_avg = before_data.mean()
                after_avg = after_data.mean()
                change_pp = after_avg - before_avg
                
                # Calculate confidence based on data quality
                confidence = self._calculate_confidence(before_data, after_data)
                
                actual_changes[indicator] = {
                    'change_pp': change_pp,
                    'before_avg': before_avg,
                    'after_avg': after_avg,
                    'confidence': confidence,
                    'data_points': len(before_data) + len(after_data)
                }
        
        return actual_changes
    
    def _calculate_confidence(self, before_data: pd.Series, after_data: pd.Series) -> str:
        """
        Calculate confidence level based on data quality.
        
        Args:
            before_data: Data before event
            after_data: Data after event
            
        Returns:
            Confidence level ('high', 'medium', 'low')
        """
        n_before = len(before_data)
        n_after = len(after_data)
        
        # Calculate variance
        var_before = before_data.var()
        var_after = after_data.var()
        
        # Check data sufficiency
        if n_before >= 5 and n_after >= 5:
            if var_before < 10 and var_after < 10:
                return 'high'
            elif var_before < 20 and var_after < 20:
                return 'medium'
        
        return 'low'
    
    def _evaluate_match(self, predicted: float, actual: float, difference: float) -> str:
        """
        Evaluate how well predicted impact matches actual change.
        
        Args:
            predicted: Predicted impact in percentage points
            actual: Actual change in percentage points
            difference: Absolute difference
            
        Returns:
            Match status ('good', 'acceptable', 'poor')
        """
        if difference <= self.thresholds['good_match']:
            return 'good'
        elif difference <= self.thresholds['acceptable_match']:
            return 'acceptable'
        else:
            return 'poor'
    
    def _determine_overall_status(self, avg_difference: float) -> str:
        """
        Determine overall validation status.
        
        Args:
            avg_difference: Average difference across all indicators
            
        Returns:
            Overall status
        """
        if avg_difference <= self.thresholds['good_match']:
            return 'excellent_match'
        elif avg_difference <= self.thresholds['acceptable_match']:
            return 'acceptable_match'
        else:
            return 'needs_review'
    
    def _generate_validation_recommendations(self, status: str, difference: float,
                                           actual_changes: Dict, 
                                           predicted_impacts: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if status == 'excellent_match':
            recommendations.append(
                "✅ Model shows excellent match with historical data"
            )
            recommendations.append(
                "✅ Continue using current modeling approach"
            )
        elif status == 'acceptable_match':
            recommendations.append(
                f"⚠️ Model shows acceptable match (avg difference: {difference:.1f}pp)"
            )
            recommendations.append(
                "📊 Consider refining impact magnitude estimates"
            )
        elif status == 'needs_review':
            recommendations.append(
                f"🔍 Model needs review (avg difference: {difference:.1f}pp)"
            )
            recommendations.append(
                "📝 Review event-impact relationships and magnitude assumptions"
            )
            recommendations.append(
                "🌍 Compare with international evidence for calibration"
            )
        
        # Add specific recommendations for large discrepancies
        for indicator, predicted in predicted_impacts.items():
            if indicator in actual_changes:
                actual = actual_changes[indicator]['change_pp']
                if abs(predicted - actual) > 10:
                    recommendations.append(
                        f"📉 Review {indicator}: Predicted {predicted:.1f}pp vs Actual {actual:.1f}pp"
                    )
        
        return recommendations
    
    def validate_against_international_evidence(self, event_type: str, 
                                              predicted_impact: float) -> Dict[str, Any]:
        """
        Validate predicted impact against international evidence.
        
        Args:
            event_type: Type of event (e.g., 'mobile_money_launch')
            predicted_impact: Predicted impact in percentage points
            
        Returns:
            Dictionary with validation results
        """
        if event_type not in self.evidence_db:
            return {
                'status': 'error',
                'message': f'No international evidence found for {event_type}'
            }
        
        evidence = self.evidence_db[event_type]
        impact_range = evidence['impact_range']
        
        if impact_range[0] <= predicted_impact <= impact_range[1]:
            status = 'within_range'
            message = f'Predicted impact ({predicted_impact}pp) is within international range {impact_range}pp'
        elif predicted_impact < impact_range[0]:
            status = 'below_range'
            message = f'Predicted impact ({predicted_impact}pp) is below international range {impact_range}pp'
        else:
            status = 'above_range'
            message = f'Predicted impact ({predicted_impact}pp) is above international range {impact_range}pp'
        
        return {
            'status': status,
            'message': message,
            'predicted_impact': predicted_impact,
            'international_range': impact_range,
            'confidence': evidence['confidence'],
            'sources': evidence['sources'],
            'duration': evidence['duration']
        }
    
    def generate_comprehensive_validation_report(self, association_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for all events.
        
        Args:
            association_matrix: DataFrame with event impact estimates
            
        Returns:
            Comprehensive validation report
        """
        print("📊 Generating comprehensive validation report...")
        
        report = {
            'metadata': {
                'generated_date': datetime.now().isoformat(),
                'total_events': len(association_matrix),
                'total_impacts': association_matrix.filter(like='_impact').count().sum()
            },
            'summary': {},
            'event_validations': [],
            'recommendations': []
        }
        
        # Validate each event
        validation_status_counts = {
            'excellent_match': 0,
            'acceptable_match': 0,
            'needs_review': 0,
            'no_data': 0
        }
        
        total_differences = []
        
        for _, event in association_matrix.iterrows():
            event_name = event['event_name']
            event_type = event.get('event_type', 'unknown')
            
            # Skip if no impacts predicted
            impact_cols = [col for col in event.index if '_impact' in col and pd.notna(event[col])]
            if not impact_cols:
                validation_status_counts['no_data'] += 1
                continue
            
            # Calculate actual changes for this event
            actual_changes = self._calculate_actual_changes(
                event_name=event_name,
                event_date=event['event_date'],
                indicators=[col.replace('_impact', '') for col in impact_cols]
            )
            
            if not actual_changes:
                validation_status_counts['no_data'] += 1
                continue
            
            # Compare predictions vs actuals
            differences = []
            validated_indicators = []
            
            for impact_col in impact_cols:
                indicator = impact_col.replace('_impact', '')
                if indicator in actual_changes:
                    predicted = event[impact_col]
                    actual = actual_changes[indicator]['change_pp']
                    difference = abs(predicted - actual)
                    differences.append(difference)
                    validated_indicators.append(indicator)
            
            if differences:
                avg_difference = np.mean(differences)
                total_differences.append(avg_difference)
                
                event_status = self._determine_overall_status(avg_difference)
                validation_status_counts[event_status] += 1
                
                report['event_validations'].append({
                    'event_name': event_name,
                    'event_type': event_type,
                    'avg_difference_pp': avg_difference,
                    'status': event_status,
                    'validated_indicators': len(validated_indicators),
                    'total_indicators': len(impact_cols)
                })
            else:
                validation_status_counts['no_data'] += 1
        
        # Generate summary
        if total_differences:
            report['summary'] = {
                'overall_avg_difference_pp': np.mean(total_differences),
                'validation_status_counts': validation_status_counts,
                'validation_rate': sum(validation_status_counts.values()) / len(association_matrix),
                'total_validated_events': sum(validation_status_counts.values()) - validation_status_counts['no_data']
            }
        
        # Generate overall recommendations
        report['recommendations'] = self._generate_overall_recommendations(report['summary'])
        
        return report
    
    def _generate_overall_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on validation summary."""
        recommendations = []
        
        if 'overall_avg_difference_pp' in summary:
            avg_diff = summary['overall_avg_difference_pp']
            
            if avg_diff <= 3.0:
                recommendations.append("🎯 Excellent model accuracy overall")
                recommendations.append("✅ Model is well-calibrated to historical data")
            elif avg_diff <= 5.0:
                recommendations.append("📊 Good model accuracy overall")
                recommendations.append("⚡ Consider minor adjustments to impact magnitudes")
            else:
                recommendations.append("🔍 Model accuracy needs improvement")
                recommendations.append("📝 Review and recalibrate impact assumptions")
                recommendations.append("🌍 Use international evidence for cross-validation")
        
        # Check validation coverage
        if summary.get('validation_rate', 0) < 0.5:
            recommendations.append("📈 Increase validation coverage by including more historical events")
        
        # Check status distribution
        status_counts = summary.get('validation_status_counts', {})
        if status_counts.get('needs_review', 0) > status_counts.get('excellent_match', 0):
            recommendations.append("⚠️ High proportion of events need review")
            recommendations.append("🔄 Consider revising event-impact framework")
        
        return recommendations
    
    def save_validation_report(self, report: Dict[str, Any], filename: str = None):
        """
        Save validation report to file.
        
        Args:
            report: Validation report dictionary
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'validation_report_{timestamp}.json'
        
        # Create directory if it doesn't exist
        output_dir = Path('reports/validation')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Validation report saved to {filepath}")
        return filepath