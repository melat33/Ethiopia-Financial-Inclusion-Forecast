"""
Data loading and validation module
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

def load_and_validate_data(data_path: str, ref_codes_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load and validate the financial inclusion dataset
    
    Args:
        data_path: Path to main dataset CSV
        ref_codes_path: Path to reference codes CSV
        
    Returns:
        Tuple of (dataframe, reference_codes dataframe)
    """
    # Load main dataset
    df = pd.read_csv(data_path)
    
    # Load reference codes if provided
    ref_codes = None
    if ref_codes_path:
        ref_codes = pd.read_csv(ref_codes_path)
    
    # Basic validation
    required_columns = ['record_type', 'indicator_code', 'value_numeric']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df, ref_codes

def validate_schema_compliance(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that dataset follows unified schema rules
    
    Returns:
        Dictionary of validation results
    """
    validations = {
        'events_no_pillar': df[(df['record_type'] == 'event') & (df['pillar'].notna())].empty,
        'observations_have_pillar': df[(df['record_type'] == 'observation') & (df['pillar'].isna())].empty,
        'impact_links_have_parent': df[(df['record_type'] == 'impact_link') & (df['parent_id'].isna())].empty,
        'all_records_have_type': df['record_type'].notna().all()
    }
    
    return validations