# src/reference_code_integrator.py
"""
Reference code integration and validation module.
Ensures data consistency and compliance with defined schema.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ReferenceCodeIntegrator:
    """
    Integrates and validates data against reference codes.
    
    This class ensures data consistency and compliance with schema.
    """
    
    def __init__(self, df: pd.DataFrame, ref_codes: pd.DataFrame):
        """
        Initialize the integrator.
        
        Args:
            df: Main dataset
            ref_codes: Reference codes dataframe
        """
        self.df = df.copy()
        self.ref_codes = ref_codes.copy()
        
        # Extract reference code categories
        self.code_categories = self._extract_code_categories()
        
        # Create mapping for easier lookup
        self.field_code_mapping = self._create_field_code_mapping()
        
        # Validation statistics
        self.validation_stats = {
            'total_records': len(df),
            'validated_fields': 0,
            'invalid_values': 0,
            'corrected_values': 0
        }
        
    def _extract_code_categories(self) -> Dict[str, List[str]]:
        """
        Extract unique code categories from reference codes.
        
        Returns:
            Dictionary of code categories and their values
        """
        categories = {}
        
        if self.ref_codes is not None and not self.ref_codes.empty:
            print(f"📋 Reference codes structure:")
            print(f"   - Columns: {self.ref_codes.columns.tolist()}")
            print(f"   - Unique fields: {self.ref_codes['field'].unique().tolist()}")
            
            # Using 'field' as category, 'code' as values
            for field in self.ref_codes['field'].unique():
                if pd.notna(field):
                    valid_values = self.ref_codes[
                        self.ref_codes['field'] == field
                    ]['code'].dropna().astype(str).tolist()
                    categories[str(field)] = valid_values
        
        print(f"📊 Extracted {len(categories)} code categories")
        return categories
    
    def _create_field_code_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Create mapping between codes and their descriptions.
        
        Returns:
            Nested dictionary mapping field -> code -> description
        """
        mapping = {}
        
        if self.ref_codes is not None and not self.ref_codes.empty:
            for field in self.ref_codes['field'].unique():
                if pd.notna(field):
                    field_data = self.ref_codes[self.ref_codes['field'] == field]
                    field_mapping = {}
                    for _, row in field_data.iterrows():
                        code = str(row['code'])
                        description = str(row['description']) if pd.notna(row['description']) else ''
                        field_mapping[code] = description
                    mapping[str(field)] = field_mapping
        
        return mapping
    
    def validate_schema_compliance(self) -> Dict[str, Any]:
        """
        Validate dataset against schema requirements.
        
        Returns:
            Dictionary with validation results
        """
        print("🔍 Validating schema compliance...")
        
        validation_results = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'dataset_shape': self.df.shape,
                'columns_count': len(self.df.columns)
            },
            'column_validation': {},
            'data_quality': {},
            'recommendations': []
        }
        
        # Check which reference code fields exist in dataset
        available_fields = []
        missing_fields = []
        
        for field in self.ref_codes['field'].unique():
            if pd.notna(field):
                # Check if any column contains this field name
                matching_cols = [col for col in self.df.columns if str(field).lower() in str(col).lower()]
                if matching_cols:
                    available_fields.append(str(field))
                else:
                    missing_fields.append(str(field))
        
        validation_results['reference_code_coverage'] = {
            'available_fields': available_fields,
            'missing_fields': missing_fields,
            'coverage_rate': len(available_fields) / (len(available_fields) + len(missing_fields)) if (len(available_fields) + len(missing_fields)) > 0 else 0
        }
        
        # Validate required columns
        required_columns = ['date', 'indicator_code', 'value']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            validation_results['column_validation']['missing_columns'] = missing_columns
            validation_results['column_validation']['status'] = 'failed'
        else:
            validation_results['column_validation']['status'] = 'passed'
            validation_results['column_validation']['present_columns'] = required_columns
        
        # Validate data types
        type_validation = {}
        for col in self.df.columns:
            if col == 'date':
                # Check if date can be parsed
                try:
                    pd.to_datetime(self.df[col])
                    type_validation[col] = {'expected': 'datetime', 'actual': 'datetime', 'status': 'passed'}
                except:
                    type_validation[col] = {'expected': 'datetime', 'actual': str(self.df[col].dtype), 'status': 'failed'}
            elif 'value' in col.lower():
                # Check if numeric
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    type_validation[col] = {'expected': 'numeric', 'actual': 'numeric', 'status': 'passed'}
                else:
                    type_validation[col] = {'expected': 'numeric', 'actual': str(self.df[col].dtype), 'status': 'failed'}
            elif 'code' in col.lower():
                type_validation[col] = {'expected': 'string', 'actual': str(self.df[col].dtype), 'status': 'passed'}
        
        validation_results['data_types'] = type_validation
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        total_missing = missing_values.sum()
        missing_percentage = (total_missing / (self.df.shape[0] * self.df.shape[1])) * 100
        
        validation_results['data_quality']['missing_values'] = {
            'total_missing': int(total_missing),
            'missing_percentage': round(missing_percentage, 2),
            'by_column': missing_values[missing_values > 0].to_dict()
        }
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        validation_results['data_quality']['duplicates'] = {
            'count': int(duplicates),
            'percentage': round((duplicates / len(self.df)) * 100, 2)
        }
        
        # Generate recommendations
        recommendations = []
        
        if missing_columns:
            recommendations.append(f"❌ Add missing columns: {', '.join(missing_columns)}")
        
        if missing_fields:
            recommendations.append(f"📋 Consider adding columns for reference fields: {', '.join(missing_fields[:3])}...")
        
        failed_types = [col for col, info in type_validation.items() if info['status'] == 'failed']
        if failed_types:
            recommendations.append(f"🔄 Fix data types for: {', '.join(failed_types[:3])}")
        
        if missing_percentage > 5:
            recommendations.append(f"📊 High missing values ({missing_percentage:.1f}%) - consider imputation")
        
        if duplicates > 0:
            recommendations.append(f"🧹 Remove {duplicates} duplicate records")
        
        validation_results['recommendations'] = recommendations
        
        print(f"✅ Schema validation complete")
        print(f"   - Missing columns: {len(missing_columns)}")
        print(f"   - Data type issues: {len(failed_types)}")
        print(f"   - Missing values: {missing_percentage:.1f}%")
        print(f"   - Reference code coverage: {validation_results['reference_code_coverage']['coverage_rate']:.1%}")
        
        return validation_results
    
    def validate_against_reference_codes(self, column_name: str, 
                                       field_name: str = None) -> Dict[str, Any]:
        """
        Validate a column against reference codes.
        
        Args:
            column_name: Name of column to validate
            field_name: Specific field name to check against (e.g., 'record_type', 'pillar')
            
        Returns:
            Dictionary with validation results
        """
        print(f"🔍 Validating {column_name} against reference codes...")
        
        if column_name not in self.df.columns:
            return {
                'status': 'error',
                'message': f'Column {column_name} not found in dataset'
            }
        
        # Get valid values
        if field_name and field_name in self.code_categories:
            valid_values = self.code_categories[field_name]
            field = field_name
        else:
            # Try to infer field from column name patterns
            field, valid_values = self._infer_field_and_values(column_name)
        
        if not valid_values:
            return {
                'status': 'warning',
                'message': f'No reference codes found for {column_name}'
            }
        
        # Validate values
        column_values = self.df[column_name].dropna().astype(str).unique()
        invalid_values = []
        valid_values_found = []
        
        for value in column_values:
            if value not in valid_values:
                invalid_values.append(value)
            else:
                valid_values_found.append(value)
        
        # Calculate statistics
        total_values = len(column_values)
        invalid_count = len(invalid_values)
        valid_count = total_values - invalid_count
        compliance_rate = (valid_count / total_values * 100) if total_values > 0 else 0
        
        # Determine status
        if invalid_count == 0:
            status = 'perfect'
        elif compliance_rate >= 95:
            status = 'excellent'
        elif compliance_rate >= 90:
            status = 'good'
        elif compliance_rate >= 80:
            status = 'acceptable'
        else:
            status = 'poor'
        
        # Get descriptions for valid values
        value_descriptions = {}
        if field in self.field_code_mapping:
            for value in valid_values_found:
                if value in self.field_code_mapping[field]:
                    value_descriptions[value] = self.field_code_mapping[field][value]
        
        # Suggest corrections for invalid values
        correction_suggestions = {}
        if invalid_values:
            correction_suggestions = self._suggest_corrections(invalid_values, valid_values, field)
        
        result = {
            'column': column_name,
            'field': field,
            'status': status,
            'statistics': {
                'total_unique_values': total_values,
                'valid_values': valid_count,
                'invalid_values': invalid_count,
                'compliance_rate': round(compliance_rate, 2),
                'valid_values_list': valid_values_found[:10],  # First 10
                'invalid_values_list': invalid_values[:10]     # First 10
            },
            'value_descriptions': value_descriptions,
            'correction_suggestions': correction_suggestions,
            'recommendations': self._generate_validation_recommendations(
                status, invalid_count, compliance_rate, column_name, field
            )
        }
        
        # Update validation stats
        self.validation_stats['validated_fields'] += 1
        self.validation_stats['invalid_values'] += invalid_count
        
        print(f"✅ Validation complete: {status.upper()}")
        print(f"   - Compliance: {compliance_rate:.1f}%")
        print(f"   - Invalid values: {invalid_count}")
        print(f"   - Field: {field}")
        
        return result
    
    def _infer_field_and_values(self, column_name: str) -> Tuple[str, List[str]]:
        """
        Infer field and valid values based on column name patterns.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Tuple of (field_name, valid_values)
        """
        column_lower = column_name.lower()
        
        # Common patterns in your dataset
        if 'type' in column_lower:
            return 'record_type', self.code_categories.get('record_type', [])
        elif 'pillar' in column_lower:
            return 'pillar', self.code_categories.get('pillar', [])
        elif 'indicator' in column_lower and 'code' in column_lower:
            # Check if we have indicator codes in reference codes
            if 'indicator' in self.code_categories:
                return 'indicator', self.code_categories.get('indicator', [])
            else:
                # Extract from dataset if available
                if 'indicator_code' in self.df.columns:
                    return 'indicator_code', self.df['indicator_code'].dropna().astype(str).unique().tolist()
        
        return '', []
    
    def _suggest_corrections(self, invalid_values: List[str], 
                           valid_values: List[str],
                           field: str = '') -> Dict[str, str]:
        """
        Suggest corrections for invalid values.
        
        Args:
            invalid_values: List of invalid values
            valid_values: List of valid values
            field: Field name for context
            
        Returns:
            Dictionary mapping invalid values to suggested corrections
        """
        correction_map = {}
        
        for value in invalid_values:
            # Try exact match with different case
            for valid in valid_values:
                if str(value).strip().lower() == str(valid).strip().lower():
                    correction_map[value] = valid
                    break
            
            # If not found, try fuzzy matching
            if value not in correction_map:
                closest = self._find_closest_match(value, valid_values, field)
                if closest:
                    correction_map[value] = closest
        
        return correction_map
    
    def _find_closest_match(self, invalid_value: str, 
                          valid_values: List[str],
                          field: str = '') -> Optional[str]:
        """Find closest valid match for an invalid value."""
        invalid_str = str(invalid_value).lower().strip()
        
        for valid in valid_values:
            valid_str = str(valid).lower().strip()
            
            # Exact match ignoring case
            if invalid_str == valid_str:
                return valid
            
            # Contains match
            if invalid_str in valid_str or valid_str in invalid_str:
                return valid
            
            # Common variations for specific fields
            if field == 'record_type':
                variations = {
                    'observation': ['obs', 'observations', 'measured'],
                    'event': ['events', 'policy', 'launch'],
                    'impact_link': ['impact', 'link', 'relationship'],
                    'target': ['targets', 'goal', 'goals']
                }
                
                for base, variant_list in variations.items():
                    if invalid_str in variant_list and base in [v.lower() for v in valid_values]:
                        return base.title()
            
            elif field == 'pillar':
                variations = {
                    'access': ['acc', 'account', 'ownership'],
                    'usage': ['usg', 'use', 'utilization'],
                    'quality': ['qual', 'qlt', 'service']
                }
                
                for base, variant_list in variations.items():
                    if invalid_str in variant_list and base in [v.lower() for v in valid_values]:
                        return base.upper()
        
        return None
    
    def _generate_validation_recommendations(self, status: str, 
                                           invalid_count: int, 
                                           compliance_rate: float,
                                           column_name: str,
                                           field: str) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if status == 'perfect':
            recommendations.append(f"✅ {column_name}: Perfect compliance with {field} reference codes")
        elif status == 'excellent':
            recommendations.append(f"✅ {column_name}: Excellent compliance ({compliance_rate:.1f}%) with {field} codes")
            if invalid_count > 0:
                recommendations.append(f"   • Review {invalid_count} invalid value(s)")
        elif status == 'good':
            recommendations.append(f"⚠️ {column_name}: Good compliance ({compliance_rate:.1f}%) with {field} codes")
            recommendations.append(f"   • Address {invalid_count} invalid value(s)")
        elif status == 'acceptable':
            recommendations.append(f"⚠️ {column_name}: Acceptable compliance ({compliance_rate:.1f}%) with {field} codes")
            recommendations.append(f"   • Fix {invalid_count} invalid value(s) for better quality")
        else:  # poor
            recommendations.append(f"❌ {column_name}: Poor compliance ({compliance_rate:.1f}%) with {field} codes")
            recommendations.append(f"   • Major cleanup needed for {invalid_count} invalid value(s)")
            recommendations.append(f"   • Consider automated correction using mapping rules")
        
        return recommendations
    
    def get_code_description(self, field: str, code: str) -> str:
        """
        Get description for a specific code.
        
        Args:
            field: Field name (e.g., 'record_type', 'pillar')
            code: Code value
            
        Returns:
            Description of the code
        """
        if field in self.field_code_mapping:
            if code in self.field_code_mapping[field]:
                return self.field_code_mapping[field][code]
        
        return f"No description found for {field}.{code}"
    
    def list_available_codes(self, field: str = None) -> Dict[str, Any]:
        """
        List all available codes and their descriptions.
        
        Args:
            field: Specific field to list (optional)
            
        Returns:
            Dictionary with codes and descriptions
        """
        if field:
            if field in self.field_code_mapping:
                return {
                    'field': field,
                    'codes': self.field_code_mapping[field]
                }
            else:
                return {'error': f'Field {field} not found in reference codes'}
        else:
            return {
                'fields': list(self.field_code_mapping.keys()),
                'total_codes': sum(len(codes) for codes in self.field_code_mapping.values())
            }
    
    def apply_corrections(self, column_name: str, 
                        correction_map: Dict[str, str]) -> pd.DataFrame:
        """
        Apply corrections to a column based on correction map.
        
        Args:
            column_name: Column to correct
            correction_map: Dictionary mapping incorrect to correct values
            
        Returns:
            Corrected dataframe
        """
        if column_name not in self.df.columns:
            print(f"❌ Column {column_name} not found")
            return self.df
        
        # Count before correction
        before_count = len(self.df)
        
        # Apply corrections
        corrected_count = 0
        for wrong, correct in correction_map.items():
            mask = self.df[column_name].astype(str) == str(wrong)
            if mask.any():
                self.df.loc[mask, column_name] = correct
                corrected_count += mask.sum()
        
        # Update validation stats
        self.validation_stats['corrected_values'] += corrected_count
        
        print(f"✅ Applied {corrected_count} corrections to {column_name}")
        print(f"   - Before: {before_count} records")
        print(f"   - After: {len(self.df)} records")
        print(f"   - Corrections applied: {corrected_count}")
        
        return self.df
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        print("📊 Generating comprehensive validation report...")
        
        report = {
            'metadata': {
                'report_date': datetime.now().isoformat(),
                'dataset': {
                    'records': len(self.df),
                    'columns': len(self.df.columns),
                    'size_mb': self.df.memory_usage(deep=True).sum() / 1024**2
                },
                'reference_codes': {
                    'total_fields': len(self.code_categories),
                    'total_codes': sum(len(codes) for codes in self.code_categories.values())
                }
            },
            'schema_validation': self.validate_schema_compliance(),
            'code_validations': [],
            'summary_statistics': self.validation_stats,
            'overall_assessment': {},
            'action_items': []
        }
        
        # Validate key columns against reference codes
        key_columns = []
        
        # Identify columns that might need validation
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['type', 'pillar', 'code']):
                key_columns.append(col)
        
        # Validate each key column
        for col in key_columns[:10]:  # Limit to first 10 to avoid too much processing
            validation_result = self.validate_against_reference_codes(col)
            report['code_validations'].append(validation_result)
        
        # Calculate overall assessment
        total_compliance = 0
        valid_validations = 0
        
        for validation in report['code_validations']:
            if 'statistics' in validation:
                total_compliance += validation['statistics']['compliance_rate']
                valid_validations += 1
        
        overall_compliance = total_compliance / valid_validations if valid_validations > 0 else 0
        
        if overall_compliance >= 95:
            overall_status = 'Excellent'
            overall_color = 'green'
        elif overall_compliance >= 90:
            overall_status = 'Good'
            overall_color = 'yellow'
        elif overall_compliance >= 80:
            overall_status = 'Acceptable'
            overall_color = 'orange'
        else:
            overall_status = 'Needs Improvement'
            overall_color = 'red'
        
        report['overall_assessment'] = {
            'compliance_score': round(overall_compliance, 2),
            'status': overall_status,
            'color': overall_color,
            'validated_columns': valid_validations,
            'total_invalid_values': self.validation_stats['invalid_values'],
            'corrected_values': self.validation_stats['corrected_values']
        }
        
        # Generate action items
        action_items = []
        
        # From schema validation
        schema_recs = report['schema_validation'].get('recommendations', [])
        action_items.extend(schema_recs)
        
        # From code validations
        for validation in report['code_validations']:
            if validation['status'] in ['poor', 'acceptable']:
                recs = validation.get('recommendations', [])
                action_items.extend(recs)
        
        report['action_items'] = list(set(action_items))  # Remove duplicates
        
        print(f"✅ Comprehensive report generated")
        print(f"   - Overall compliance: {overall_compliance:.1f}% ({overall_status})")
        print(f"   - Validated columns: {valid_validations}")
        print(f"   - Action items: {len(report['action_items'])}")
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
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
    
    def get_validated_data(self) -> pd.DataFrame:
        """
        Get the validated (and potentially corrected) dataframe.
        
        Returns:
            Validated dataframe
        """
        return self.df.copy()