"""
Data quality assessment module
"""
import pandas as pd
import numpy as np
from typing import Dict

def assess_data_quality(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality assessment
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'basic_stats': {},
        'schema_compliance': {},
        'temporal_quality': {},
        'completeness': {}
    }
    
    # Basic statistics
    report['basic_stats'] = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'duplicate_records': df.duplicated().sum(),
        'duplicate_percent': (df.duplicated().sum() / len(df)) * 100
    }
    
    # Schema compliance
    report['schema_compliance'] = {
        'events_have_no_pillar': df[(df['record_type'] == 'event') & (df['pillar'].notna())].empty,
        'observations_have_pillar': df[(df['record_type'] == 'observation') & (df['pillar'].isna())].empty,
        'impact_links_have_parent': 'parent_id' in df.columns
    }
    
    # Temporal quality
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for date_col in date_columns:
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            valid_dates = dates.dropna()
            report['temporal_quality'][date_col] = {
                'valid_count': len(valid_dates),
                'invalid_count': len(dates) - len(valid_dates),
                'date_range': f"{valid_dates.min().date()} to {valid_dates.max().date()}" if len(valid_dates) > 0 else "No valid dates"
            }
    
    # Completeness
    missing = df.isnull().sum()
    report['completeness'] = {
        'columns_with_missing': (missing > 0).sum(),
        'total_missing_values': missing.sum(),
        'completeness_percent': (1 - missing.sum() / (len(df) * len(df.columns))) * 100
    }
    
    return report