"""
Chart Components
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import config

def create_progress_chart():
    """Create progress vs target chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current (2024)',
        x=['Account Ownership', 'Digital Payments'],
        y=[49, 35],
        marker_color=[config.COLORS['primary'], config.COLORS['secondary']],
        text=['49%', '35%'],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='NFIS-II 2025 Target',
        x=['Account Ownership', 'Digital Payments'],
        y=[70, 45],
        marker_color=['rgba(46, 134, 171, 0.3)', 'rgba(78, 205, 196, 0.3)'],
        text=['70%', '45%'],
        textposition='auto',
    ))
    
    fig.update_layout(
        barmode='overlay',
        height=config.CHART_HEIGHT,
        showlegend=True,
        plot_bgcolor=config.CHART_BG_COLOR,
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_trend_chart(historical_data, selected_indicators):
    """Create historical trend chart"""
    fig = go.Figure()
    
    colors = {
        'Account Ownership': config.COLORS['primary'],
        'Digital Payments': config.COLORS['secondary']
    }
    
    for indicator in selected_indicators:
        fig.add_trace(go.Scatter(
            x=historical_data['year'],
            y=historical_data[indicator],
            name=indicator,
            mode='lines+markers',
            line=dict(color=colors.get(indicator, config.COLORS['primary']), width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        height=500,
        title="Historical Trends",
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        plot_bgcolor=config.CHART_BG_COLOR,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_forecast_chart(forecast_data):
    """Create forecast visualization"""
    if forecast_data.empty:
        return create_sample_forecast_chart()
    
    # Pivot for plotting
    pivot_df = forecast_data.pivot(index='year', columns='indicator', values='forecast_value')
    
    fig = go.Figure()
    
    colors = {
        'ACC_OWNERSHIP': config.COLORS['primary'],
        'USG_DIGITAL_PAYMENT': config.COLORS['secondary']
    }
    
    for column in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[column],
            name=column.replace('_', ' '),
            mode='lines+markers',
            line=dict(color=colors.get(column, config.COLORS['primary']), width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        height=500,
        title="Financial Inclusion Forecasts 2025-2027",
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        plot_bgcolor=config.CHART_BG_COLOR,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_sample_forecast_chart():
    """Create sample forecast chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[2025, 2026, 2027],
        y=[54.2, 56.9, 59.7],
        name='Account Ownership',
        mode='lines+markers',
        line=dict(color=config.COLORS['primary'], width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=[2025, 2026, 2027],
        y=[36.4, 38.8, 41.2],
        name='Digital Payments',
        mode='lines+markers',
        line=dict(color=config.COLORS['secondary'], width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        height=500,
        title="Sample Forecasts 2025-2027",
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        plot_bgcolor=config.CHART_BG_COLOR,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_scenario_chart():
    """Create scenario comparison chart"""
    scenario_data = pd.DataFrame({
        'Scenario': ['Optimistic', 'Baseline', 'Pessimistic'],
        'Account Ownership 2027': [67.5, 62.5, 56.6],
        'Digital Payments 2027': [53.1, 45.4, 36.5]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Account Ownership',
        x=scenario_data['Scenario'],
        y=scenario_data['Account Ownership 2027'],
        marker_color=config.COLORS['primary'],
        text=scenario_data['Account Ownership 2027'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Digital Payments',
        x=scenario_data['Scenario'],
        y=scenario_data['Digital Payments 2027'],
        marker_color=config.COLORS['secondary'],
        text=scenario_data['Digital Payments 2027'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto'
    ))
    
    fig.update_layout(
        height=400,
        barmode='group',
        plot_bgcolor=config.CHART_BG_COLOR,
        title="2027 Forecast by Scenario",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_channel_chart(channel_data):
    """Create channel comparison chart"""
    fig = px.line(
        channel_data.melt(id_vars=['Year'], var_name='Channel', value_name='Usage'),
        x='Year',
        y='Usage',
        color='Channel',
        markers=True,
        title="Channel Adoption Over Time"
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor=config.CHART_BG_COLOR,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig