"""
Visualization utilities for EDA
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict
import os

def create_eda_visualizations(df: pd.DataFrame, save_path: str = None) -> Dict[str, str]:
    """
    Create standard EDA visualizations
    
    Returns:
        Dictionary mapping figure names to file paths
    """
    figures = {}
    
    # Ensure save directory exists
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Record type distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    record_counts = df['record_type'].value_counts()
    record_counts.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax1.set_title('Record Type Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    if save_path:
        fig1_path = os.path.join(save_path, 'record_type_distribution.png')
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        figures['record_type_distribution'] = fig1_path
    
    # 2. Confidence distribution
    if 'confidence' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        conf_counts = df['confidence'].value_counts()
        conf_counts.plot(kind='bar', ax=ax2, color=['#2E86AB', '#F18F01', '#C73E1D'])
        ax2.set_title('Data Confidence Levels', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        
        if save_path:
            fig2_path = os.path.join(save_path, 'confidence_distribution.png')
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            figures['confidence_distribution'] = fig2_path
    
    plt.close('all')  # Close all figures to free memory
    
    return figures