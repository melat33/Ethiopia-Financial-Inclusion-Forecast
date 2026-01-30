"""
Exploratory Data Analysis module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

def perform_comprehensive_eda(df: pd.DataFrame, ref_codes: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Perform comprehensive EDA on financial inclusion dataset
    
    Returns:
        Dictionary containing all EDA results
    """
    results = {
        'basic_stats': {},
        'record_type_summary': None,
        'missing_values_summary': None,
        'confidence_distribution': None,
        'temporal_coverage': {},
        'indicator_stats': None,
        'data_gaps': {},
        'key_insights': []
    }
    
    # Basic statistics
    results['basic_stats'] = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'unique_record_types': df['record_type'].nunique(),
        'unique_indicators': df['indicator_code'].nunique()
    }
    
    # Record type distribution
    results['record_type_summary'] = df['record_type'].value_counts()
    
    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    results['missing_values_summary'] = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_pct
    }).sort_values('missing_percent', ascending=False)
    
    # Confidence distribution
    if 'confidence' in df.columns:
        results['confidence_distribution'] = df['confidence'].value_counts()
    
    # Temporal analysis
    results['temporal_coverage'] = analyze_temporal_coverage(df)
    
    # Indicator statistics
    results['indicator_stats'] = analyze_indicators(df)
    
    # Data gap analysis
    results['data_gaps'] = identify_data_gaps(df)
    
    # Key insights
    results['key_insights'] = extract_key_insights(results)
    
    return results

def analyze_temporal_coverage(df: pd.DataFrame) -> Dict:
    """Analyze temporal coverage of dataset"""
    temporal = {}
    
    # Observations temporal coverage
    obs_df = df[df['record_type'] == 'observation']
    if not obs_df.empty and 'observation_date' in obs_df.columns:
        obs_df['year'] = pd.to_datetime(obs_df['observation_date'], errors='coerce').dt.year
        temporal['observation_years'] = sorted(obs_df['year'].dropna().unique().astype(int))
    
    # Events temporal coverage
    event_df = df[df['record_type'] == 'event']
    if not event_df.empty and 'event_date' in event_df.columns:
        event_df['year'] = pd.to_datetime(event_df['event_date'], errors='coerce').dt.year
        temporal['event_years'] = sorted(event_df['year'].dropna().unique().astype(int))
    
    # Identify gaps
    if 'observation_years' in temporal:
        years = temporal['observation_years']
        if len(years) > 1:
            all_years = list(range(min(years), max(years) + 1))
            temporal['temporal_gaps'] = [y for y in all_years if y not in years]
    
    return temporal

def analyze_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze key indicators"""
    observations = df[df['record_type'] == 'observation'].copy()
    
    if observations.empty:
        return pd.DataFrame()
    
    # Convert dates
    observations['obs_date'] = pd.to_datetime(observations['observation_date'], errors='coerce')
    
    # Group by indicator
    indicator_stats = observations.groupby('indicator_code').agg({
        'value_numeric': ['count', 'mean', 'min', 'max', 'std'],
        'obs_date': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    indicator_stats.columns = ['_'.join(col).strip() for col in indicator_stats.columns.values]
    
    return indicator_stats.sort_values('value_numeric_count', ascending=False)

def identify_data_gaps(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify critical data gaps"""
    gaps = {
        'temporal_gaps': [],
        'infrastructure_gaps': [],
        'usage_gaps': [],
        'disaggregation_gaps': [],
        'event_gaps': []
    }
    
    # Temporal gaps (based on Findex years)
    obs_df = df[df['record_type'] == 'observation']
    if 'observation_date' in obs_df.columns:
        obs_df['year'] = pd.to_datetime(obs_df['observation_date'], errors='coerce').dt.year
        findex_years = sorted(obs_df['year'].dropna().unique())
        
        if len(findex_years) > 1:
            all_possible = list(range(int(min(findex_years)), int(max(findex_years)) + 1))
            missing_years = [str(y) for y in all_possible if y not in findex_years]
            gaps['temporal_gaps'] = missing_years
    
    # Infrastructure gaps
    infra_indicators = ['agent', 'atm', 'pos', 'coverage', 'network']
    existing_infra = [code for code in df['indicator_code'].dropna().unique() 
                     if any(keyword in str(code).lower() for keyword in infra_indicators)]
    
    if len(existing_infra) < 3:  # Arbitrary threshold
        gaps['infrastructure_gaps'].append("Limited infrastructure metrics")
    
    # Usage gaps
    if 'USG_ACTIVE_MM_USERS' not in df['indicator_code'].values:
        gaps['usage_gaps'].append("Missing active vs registered user data")
    
    # Disaggregation gaps
    gender_indicators = [code for code in df['indicator_code'].dropna().unique() 
                        if any(keyword in str(code).lower() for keyword in ['male', 'female', 'gender'])]
    
    if len(gender_indicators) < 2:
        gaps['disaggregation_gaps'].append("Limited gender-disaggregated data")
    
    return gaps

def extract_key_insights(eda_results: Dict) -> List[str]:
    """Extract key insights from EDA results"""
    insights = []
    
    # Insight 1: Data sparsity
    total_records = eda_results['basic_stats']['total_records']
    if total_records < 100:
        insights.append(f"Limited dataset with only {total_records} records")
    
    # Insight 2: Temporal coverage
    temporal = eda_results['temporal_coverage']
    if 'observation_years' in temporal:
        obs_years = temporal['observation_years']
        if len(obs_years) <= 5:
            insights.append(f"Sparse temporal coverage: only {len(obs_years)} observation years")
    
    # Insight 3: Missing data
    missing_summary = eda_results['missing_values_summary']
    high_missing = missing_summary[missing_summary['missing_percent'] > 30]
    if len(high_missing) > 0:
        insights.append(f"{len(high_missing)} columns with >30% missing values")
    
    return insights