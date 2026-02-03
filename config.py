"""
Dashboard Configuration Settings
"""

# App Configuration
APP_NAME = "Ethiopia Financial Inclusion Dashboard"
APP_VERSION = "1.0.0"

# Color Scheme
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#4ECDC4",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "danger": "#FF6B6B",
    "dark": "#2C3E50",
    "light": "#F8F9FA"
}

# Chart Settings
CHART_HEIGHT = 400
CHART_BG_COLOR = "white"

# Data Settings
FORECAST_YEARS = [2025, 2026, 2027]
SCENARIOS = ["Optimistic", "Baseline", "Pessimistic"]

# Target Values
TARGETS = {
    "account_ownership": {
        "current": 49.0,
        "2025_target": 70.0,
        "unit": "%"
    },
    "digital_payments": {
        "current": 35.0,
        "2025_target": 45.0,
        "unit": "%"
    }
}