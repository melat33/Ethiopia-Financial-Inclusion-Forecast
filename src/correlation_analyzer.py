"""Correlation analysis for financial inclusion indicators"""
import pandas as pd
import numpy as np
from scipy import stats

class CorrelationAnalyzer:
    def __init__(self, df):
        self.df = df
        self.prepare_pivot_data()
    
    def prepare_pivot_data(self):
        """Prepare pivot table for correlation analysis"""
        obs_df = self.df[self.df['record_type'] == 'observation'].copy()
        obs_df['year'] = pd.to_datetime(obs_df['observation_date']).dt.year
        
        # Create pivot table
        self.pivot_df = obs_df.pivot_table(
            index='year',
            columns='indicator_code',
            values='value_numeric',
            aggfunc='mean'
        ).dropna(how='all', axis=1).dropna(how='all', axis=0)
    
    def get_correlation_matrix(self):
        """Get correlation matrix for all indicators"""
        return self.pivot_df.corr()
    
    def find_strongest_correlations(self, target_indicator, n=10):
        """Find strongest correlations with target indicator"""
        if target_indicator not in self.pivot_df.columns:
            return []
        
        correlations = {}
        corr_matrix = self.get_correlation_matrix()
        
        if target_indicator in corr_matrix.index:
            target_corrs = corr_matrix[target_indicator]
            for indicator, corr_value in target_corrs.items():
                if indicator != target_indicator and not pd.isna(corr_value):
                    correlations[indicator] = corr_value
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_corrs[:n]