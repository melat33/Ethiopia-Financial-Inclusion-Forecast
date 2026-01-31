"""
Reference code integration from Task 1
"""
import pandas as pd
from typing import Dict, List

class ReferenceCodeIntegrator:
    def __init__(self, df: pd.DataFrame, ref_codes: pd.DataFrame):
        self.df = df
        self.ref_codes = ref_codes
        
    def validate_all_categories(self) -> Dict:
        """Validate all categorical columns against reference codes"""
        results = {}
        
        # Check each reference category
        for field in self.ref_codes['field'].unique():
            if field in self.df.columns:
                valid_codes = self.ref_codes[
                    self.ref_codes['field'] == field
                ]['code'].tolist()
                
                # Get actual values
                actual_values = self.df[field].dropna().unique()
                
                # Calculate compliance
                valid_count = sum(1 for v in actual_values if v in valid_codes)
                total_count = len(actual_values)
                compliance = (valid_count / total_count * 100) if total_count > 0 else 100
                
                results[field] = {
                    'valid_count': valid_count,
                    'total_count': total_count,
                    'compliance': compliance,
                    'valid_codes': valid_codes[:5]  # First 5 for display
                }
        
        # Calculate overall compliance
        if results:
            total_valid = sum(r['valid_count'] for r in results.values())
            total_all = sum(r['total_count'] for r in results.values())
            overall = (total_valid / total_all * 100) if total_all > 0 else 100
            
            results['overall_compliance'] = {
                'total': total_all,
                'valid': total_valid,
                'percentage': overall
            }
        
        return results
    
    def get_category_description(self, category: str) -> str:
        """Get description for a category"""
        desc = self.ref_codes[
            (self.ref_codes['field'] == 'category') & 
            (self.ref_codes['code'] == category)
        ]['description']
        
        return desc.iloc[0] if len(desc) > 0 else 'No description'