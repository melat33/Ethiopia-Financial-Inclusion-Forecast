"""
Data loading and validation module for Ethiopia Financial Inclusion Project
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_validate_data(data_path, ref_codes_path=None):
    """
    Load and validate the financial inclusion dataset
    
    Parameters:
    -----------
    data_path : str
        Path to the main dataset (Excel or CSV file)
    ref_codes_path : str, optional
        Path to reference codes file
        
    Returns:
    --------
    Tuple of (dataframe, reference_codes dataframe)
    """
    
    # Check file extension and load accordingly
    file_extension = os.path.splitext(data_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(data_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Use CSV or Excel files.")
    
    print(f"✅ Loaded dataset from: {data_path}")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    
    # Load reference codes if provided
    ref_codes = None
    if ref_codes_path:
        ref_extension = os.path.splitext(ref_codes_path)[1].lower()
        
        if ref_extension == '.csv':
            ref_codes = pd.read_csv(ref_codes_path)
        elif ref_extension in ['.xlsx', '.xls']:
            ref_codes = pd.read_excel(ref_codes_path)
        else:
            print(f"⚠️ Warning: Could not load reference codes from {ref_codes_path}")
    
    # Basic validation
    validate_dataset(df)
    
    return df, ref_codes

def validate_dataset(df):
    """
    Perform basic validation on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to validate
    """
    print("\n🔍 Performing dataset validation...")
    
    # Check required columns
    required_columns = ['record_type', 'indicator_code', 'value_numeric', 'obs_date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ Warning: Missing required columns: {missing_columns}")
    else:
        print("✅ All required columns present")
    
    # Check data types
    if 'obs_date' in df.columns:
        try:
            # Try to convert to datetime
            df['obs_date'] = pd.to_datetime(df['obs_date'])
            print("✅ obs_date column converted to datetime")
        except Exception as e:
            print(f"⚠️ Warning: Could not convert obs_date to datetime: {e}")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ Warning: {duplicates} duplicate rows found")
    else:
        print("✅ No duplicate rows found")
    
    # Basic statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Total records: {len(df):,}")
    print(f"   - Unique indicators: {df['indicator_code'].nunique()}")
    print(f"   - Date range: {df['obs_date'].min().date()} to {df['obs_date'].max().date()}")
    
    # Record type distribution
    if 'record_type' in df.columns:
        record_counts = df['record_type'].value_counts()
        print("\n📋 Record Type Distribution:")
        for rt, count in record_counts.items():
            print(f"   - {rt}: {count:,}")
    
    return True