"""
Professional Data Loader for Ethiopia Financial Inclusion Forecasting
Handles real data formats from Task 1-3 outputs
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Professional data loader for financial inclusion forecasting"""
    
    def __init__(self, base_path: str = '.'):
        self.base_path = base_path
        self.data_cache = {}
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all required data for Task 4"""
        print("📂 LOADING TASK 4 DATA")
        print("="*50)
        
        # 1. Load enriched dataset from Task 1
        print("\n1️⃣ Loading enriched dataset...")
        historical_data = self._load_enriched_dataset()
        
        # 2. Load event matrix from Task 3
        print("\n2️⃣ Loading event impact matrix...")
        event_matrix = self._load_event_matrix()
        
        # 3. Load target data
        print("\n3️⃣ Loading target data...")
        target_data = self._load_target_data()
        
        # 4. Validate data
        print("\n4️⃣ Validating data...")
        self._validate_data(historical_data, event_matrix, target_data)
        
        print(f"\n✅ DATA LOADING COMPLETE")
        print(f"   • Historical points: {len(historical_data)}")
        print(f"   • Events: {len(event_matrix)}")
        print(f"   • Target years: {len(target_data)}")
        
        return historical_data, event_matrix, target_data
    
    def _load_enriched_dataset(self) -> pd.DataFrame:
        """Load and process the enriched dataset from Task 1"""
        
        # Try multiple possible locations
        possible_paths = [
            'data/processed/ethiopia_fi_enriched.csv',
            '../data/processed/ethiopia_fi_enriched.csv',
            './ethiopia_fi_enriched.csv',
            '../ethiopia_fi_enriched.csv'
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"   Found file: {path}")
                break
        
        if not file_path:
            raise FileNotFoundError(
                "❌ Could not find enriched dataset. "
                "Please ensure ethiopia_fi_enriched.csv exists in data/processed/"
            )
        
        try:
            # Load with flexible date parsing
            print(f"   Loading {file_path}...")
            df = pd.read_csv(file_path, low_memory=False)
            print(f"   Raw data shape: {df.shape}")
            
            # Process the data
            processed_data = self._process_enriched_data(df)
            return processed_data
            
        except Exception as e:
            raise Exception(f"❌ Failed to load enriched dataset: {str(e)}")
    
    def _process_enriched_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process enriched dataset into historical time series"""
        
        print("   Processing data...")
        
        # Filter for observation records only
        if 'record_type' in df.columns:
            obs_df = df[df['record_type'] == 'observation'].copy()
            print(f"   Observation records: {len(obs_df)}")
        else:
            obs_df = df.copy()
            print("   ⚠️ No record_type column - using all records")
        
        # Extract year from observation_date
        if 'observation_date' in obs_df.columns:
            print("   Parsing observation dates...")
            obs_df['year'] = self._extract_year(obs_df['observation_date'])
        else:
            # Try other date columns
            for date_col in ['collection_date', 'event_date', 'date']:
                if date_col in obs_df.columns:
                    print(f"   Using {date_col} for year extraction...")
                    obs_df['year'] = self._extract_year(obs_df[date_col])
                    break
        
        if 'year' not in obs_df.columns:
            raise ValueError("❌ Could not extract year from any date column")
        
        # Filter valid years (2000-2030)
        obs_df = obs_df[(obs_df['year'] >= 2000) & (obs_df['year'] <= 2030)]
        
        # Get unique indicators
        if 'indicator_code' in obs_df.columns:
            indicators = obs_df['indicator_code'].unique()
            print(f"   Found {len(indicators)} unique indicators")
            
            # Create historical DataFrame
            historical_data = pd.DataFrame()
            
            # Start with year column
            all_years = sorted(obs_df['year'].unique())
            historical_data['year'] = all_years
            
            # Process key indicators
            key_indicators = {
                'ACC_OWNERSHIP': 'Account Ownership',
                'USG_DIGITAL_PAYMENT': 'Digital Payments',
                'ACC_MM_ACCOUNT': 'Mobile Money Accounts',
                'GEN_GAP_ACC': 'Gender Gap',
                'INF_AGENT_DENSITY': 'Agent Density'
            }
            
            for indicator_code, display_name in key_indicators.items():
                print(f"   Processing {indicator_code}...")
                
                # Filter for this indicator
                indicator_data = obs_df[obs_df['indicator_code'] == indicator_code].copy()
                
                if not indicator_data.empty:
                    # Clean numeric values
                    indicator_data['value_numeric'] = pd.to_numeric(
                        indicator_data['value_numeric'], errors='coerce'
                    )
                    
                    # Group by year
                    yearly_data = indicator_data.groupby('year').agg({
                        'value_numeric': 'mean'
                    }).reset_index()
                    
                    yearly_data = yearly_data.rename(
                        columns={'value_numeric': indicator_code}
                    )
                    
                    # Merge with historical data
                    historical_data = historical_data.merge(
                        yearly_data, on='year', how='left'
                    )
            
            print(f"   Processed historical data shape: {historical_data.shape}")
            
            # Sort by year
            historical_data = historical_data.sort_values('year').reset_index(drop=True)
            
            return historical_data
        
        else:
            raise ValueError("❌ No indicator_code column found in data")
    
    def _extract_year(self, date_series: pd.Series) -> pd.Series:
        """Extract year from date column with robust parsing"""
        
        # Try multiple parsing strategies
        year_series = pd.Series(index=date_series.index, dtype='float64')
        
        # Strategy 1: Direct parsing
        try:
            parsed_dates = pd.to_datetime(date_series, errors='coerce', format='mixed')
            year_series = parsed_dates.dt.year
            if year_series.notna().any():
                return year_series
        except:
            pass
        
        # Strategy 2: Extract year from string
        try:
            # Look for 4-digit year patterns
            year_pattern = date_series.astype(str).str.extract(r'(\d{4})')
            if year_pattern.notna().any():
                return year_pattern[0].astype(float)
        except:
            pass
        
        # Strategy 3: Handle common date formats
        date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d']
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(date_series, errors='coerce', format=fmt)
                if parsed.notna().any():
                    return parsed.dt.year
            except:
                continue
        
        # If all fails, return NaN
        return pd.Series([np.nan] * len(date_series), index=date_series.index)
    
    def _load_event_matrix(self) -> pd.DataFrame:
        """Load event impact matrix from Task 3"""
        
        possible_paths = [
            'models/task3/event_indicator_association_matrix.csv',
            '../models/task3/event_indicator_association_matrix.csv',
            'event_indicator_association_matrix.csv'
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"   Found event matrix: {path}")
                break
        
        if not file_path:
            print("   ⚠️ Event matrix not found - creating synthetic data")
            return self._create_synthetic_event_matrix()
        
        try:
            event_matrix = pd.read_csv(file_path)
            print(f"   Event matrix shape: {event_matrix.shape}")
            
            # Clean event matrix
            event_matrix = self._clean_event_matrix(event_matrix)
            
            return event_matrix
            
        except Exception as e:
            print(f"   ⚠️ Failed to load event matrix: {e}")
            return self._create_synthetic_event_matrix()
    
    def _clean_event_matrix(self, event_matrix: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize event matrix"""
        
        # Ensure required columns
        required_impact_cols = ['ACC_OWNERSHIP_impact', 'USG_DIGITAL_PAYMENT_impact']
        
        for col in required_impact_cols:
            if col not in event_matrix.columns:
                print(f"   ⚠️ Adding missing column: {col}")
                event_matrix[col] = 0.0
        
        # Clean impact values
        for col in event_matrix.columns:
            if 'impact' in col:
                # Convert to numeric, handle string values like "2.0pp"
                event_matrix[col] = event_matrix[col].astype(str).str.replace('pp', '', regex=False)
                event_matrix[col] = pd.to_numeric(event_matrix[col], errors='coerce')
                event_matrix[col] = event_matrix[col].fillna(0.0)
        
        return event_matrix
    
    def _create_synthetic_event_matrix(self) -> pd.DataFrame:
        """Create synthetic event matrix for testing"""
        
        events = [
            {
                'event_name': 'Telebirr Launch',
                'event_year': 2021,
                'ACC_OWNERSHIP_impact': 2.0,
                'USG_DIGITAL_PAYMENT_impact': 3.0,
                'confidence': 'high'
            },
            {
                'event_name': 'M-Pesa Entry',
                'event_year': 2023,
                'ACC_OWNERSHIP_impact': 1.5,
                'USG_DIGITAL_PAYMENT_impact': 2.5,
                'confidence': 'medium'
            },
            {
                'event_name': 'QR System Launch',
                'event_year': 2023,
                'ACC_OWNERSHIP_impact': 0.8,
                'USG_DIGITAL_PAYMENT_impact': 1.5,
                'confidence': 'medium'
            }
        ]
        
        return pd.DataFrame(events)
    
    def _load_target_data(self) -> pd.DataFrame:
        """Load NFIS-II and other target data"""
        
        print("   Loading NFIS-II targets...")
        
        target_data = pd.DataFrame({
            'year': [2025, 2030],
            'ACC_OWNERSHIP': [70.0, 75.0],
            'USG_DIGITAL_PAYMENT': [45.0, 60.0],
            'source': ['NFIS-II 2025 Target', 'NFIS-II 2030 Target'],
            'pillar': ['ACCESS', 'USAGE']
        })
        
        return target_data
    
    def _validate_data(self, historical_data: pd.DataFrame, 
                      event_matrix: pd.DataFrame, 
                      target_data: pd.DataFrame):
        """Validate loaded data"""
        
        print("\n📊 DATA VALIDATION REPORT")
        print("-" * 30)
        
        # Historical data validation
        print("\nHistorical Data:")
        print(f"   Years: {historical_data['year'].min()} to {historical_data['year'].max()}")
        
        for indicator in ['ACC_OWNERSHIP', 'USG_DIGITAL_PAYMENT']:
            if indicator in historical_data.columns:
                values = historical_data[indicator].dropna()
                if len(values) > 0:
                    print(f"   {indicator}: {len(values)} data points")
                    print(f"     Range: {values.min():.1f}% to {values.max():.1f}%")
                else:
                    print(f"   ⚠️ {indicator}: No data points")
            else:
                print(f"   ⚠️ {indicator}: Column missing")
        
        # Event matrix validation
        print(f"\nEvent Matrix: {len(event_matrix)} events")
        if 'ACC_OWNERSHIP_impact' in event_matrix.columns:
            total_impact = event_matrix['ACC_OWNERSHIP_impact'].sum()
            print(f"   Total account ownership impact: {total_impact:.1f}pp")
        
        # Target validation
        print(f"\nTarget Data: {len(target_data)} target years")
        print(f"   Target years: {list(target_data['year'].values)}")
        
        print("\n✅ Validation complete")
    
    def get_data_summary(self) -> Dict:
        """Get summary of loaded data"""
        
        summary = {
            'historical_years': [],
            'indicators': [],
            'event_count': 0,
            'target_years': []
        }
        
        if 'historical_data' in self.data_cache:
            hist = self.data_cache['historical_data']
            summary['historical_years'] = sorted(hist['year'].unique().tolist())
            summary['indicators'] = [col for col in hist.columns if col != 'year']
        
        if 'event_matrix' in self.data_cache:
            summary['event_count'] = len(self.data_cache['event_matrix'])
        
        if 'target_data' in self.data_cache:
            summary['target_years'] = self.data_cache['target_data']['year'].tolist()
        
        return summary


# Convenience function for notebook use
def load_task_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function for loading Task 4 data"""
    loader = DataLoader()
    historical_data, event_matrix, target_data = loader.load_all_data()
    
    # Cache the data
    loader.data_cache = {
        'historical_data': historical_data,
        'event_matrix': event_matrix,
        'target_data': target_data
    }
    
    return historical_data, event_matrix, target_data


def create_data_diagnostic_report(data_path: str = None) -> pd.DataFrame:
    """Create diagnostic report for data files"""
    
    report = []
    
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, nrows=1000)  # Read first 1000 rows for diagnostics
            
            report.append({
                'file': os.path.basename(data_path),
                'rows': len(df),
                'columns': len(df.columns),
                'date_columns': [col for col in df.columns if 'date' in col.lower()],
                'indicator_columns': [col for col in df.columns if 'indicator' in col.lower()],
                'sample_dates': df.iloc[0:3][[col for col in df.columns if 'date' in col.lower()][0]].tolist() if any('date' in col.lower() for col in df.columns) else []
            })
        except Exception as e:
            report.append({
                'file': os.path.basename(data_path),
                'error': str(e)
            })
    
    return pd.DataFrame(report)