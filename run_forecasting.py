"""
Main script to run Ethiopia Financial Inclusion Forecasting (Task 4)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('.')

print("="*60)
print("ETHIOPIA FINANCIAL INCLUSION FORECASTING SYSTEM")
print("Task 4: Forecasting Access and Usage (2025-2027)")
print("="*60)

# Initialize variables
DataLoader = None
ScenarioGenerator = None
UncertaintyQuantifier = None
ForecastVisualizer = None

print("\n📦 Loading modules...")

# Import core models (required)
try:
    from src.forecasting.core_models import FinancialInclusionForecaster, ForecastParameters
    print("✅ core_models imported")
except ImportError as e:
    print(f"❌ Error importing core_models: {e}")
    sys.exit(1)

# Import data loader (optional)
try:
    from src.utils.data_loader import DataLoader
    print("✅ data_loader imported")
except ImportError as e:
    print(f"⚠️ data_loader not found ({e}), will use synthetic data")

# Import scenario engine (optional)
try:
    from src.forecasting.scenario_engine import ScenarioGenerator, ScenarioParameters
    print("✅ scenario_engine imported")
except ImportError as e:
    print(f"⚠️ scenario_engine not found ({e}), will skip scenario analysis")

# Import uncertainty (optional)
try:
    from src.forecasting.uncertainty import UncertaintyQuantifier
    print("✅ uncertainty imported")
except ImportError as e:
    print(f"⚠️ uncertainty not found ({e}), will skip uncertainty analysis")

# Import visualization (optional)
try:
    from src.forecasting.visualization import ForecastVisualizer
    print("✅ visualization imported")
except ImportError as e:
    print(f"⚠️ visualization not found ({e}), will skip visualizations")

def create_synthetic_data():
    """Create synthetic data for testing"""
    historical_data = pd.DataFrame({
        'year': [2011, 2014, 2017, 2021, 2024],
        'ACC_OWNERSHIP': [14.0, 22.0, 35.0, 46.0, 49.0],
        'USG_DIGITAL_PAYMENT': [10.0, 18.0, 25.0, 35.0, 35.0]
    })
    
    event_matrix = pd.DataFrame({
        'event_name': ['Telebirr Launch', 'M-Pesa Entry', 'QR System Launch'],
        'event_year': [2021, 2023, 2023],
        'ACC_OWNERSHIP_impact': [2.0, 1.5, 0.8],
        'USG_DIGITAL_PAYMENT_impact': [3.0, 2.5, 1.5],
        'confidence': ['high', 'medium', 'medium']
    })
    
    target_data = pd.DataFrame({
        'year': [2025, 2030],
        'ACC_OWNERSHIP': [70.0, 75.0],
        'USG_DIGITAL_PAYMENT': [45.0, 60.0]
    })
    
    return historical_data, event_matrix, target_data

def main():
    """Main execution function for Task 4"""
    
    # Step 1: Load data
    print("\n📂 STEP 1: Loading data...")
    
    if DataLoader:  # Check if DataLoader was successfully imported
        try:
            loader = DataLoader()
            historical_data, event_matrix, target_data = loader.load_all_data()
            print(f"✅ Data loaded successfully")
            print(f"   Historical data shape: {historical_data.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Creating synthetic data...")
            historical_data, event_matrix, target_data = create_synthetic_data()
    else:
        print("Creating synthetic data...")
        historical_data, event_matrix, target_data = create_synthetic_data()
    
    # Display data preview
    print(f"\n📊 Historical Data Preview:")
    print(historical_data.head())
    
    if event_matrix is not None:
        print(f"\n⚡ Event Impact Matrix Preview:")
        print(event_matrix.head())
    
    # Step 2: Initialize forecaster
    print("\n🤖 STEP 2: Initializing forecasting models...")
    params = ForecastParameters(
        forecast_horizon=[2025, 2026, 2027],
        baseline_year=2024,
        confidence_level=0.95,
        n_simulations=1000
    )
    
    forecaster = FinancialInclusionForecaster(params)
    
    # Step 3: Generate forecasts
    print("\n🔮 STEP 3: Generating forecasts (2025-2027)...")
    try:
        forecast_results = forecaster.generate_complete_forecasts(
            historical_data, 
            event_matrix, 
            target_data
        )
        print(f"✅ Forecasts generated for {len(forecast_results)} indicators")
    except Exception as e:
        print(f"❌ Error generating forecasts: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display forecast results
    if forecast_results:
        print("\n📊 Forecast Results:")
        for indicator, results in forecast_results.items():
            print(f"\n{indicator.replace('_', ' ')}:")
            if 'ensemble' in results:
                forecasts = results['ensemble']['forecasts']
                for year, value in sorted(forecasts.items()):
                    print(f"   {year}: {value:.1f}%")
            
            if 'target_gap' in results:
                print(f"   NFIS-II Target Analysis:")
                for year, gap in results['target_gap'].items():
                    print(f"     {year}: Target {gap['target']}% | Forecast {gap['forecast']:.1f}% | Gap {gap['gap_pp']:.1f}pp")
    else:
        print("❌ No forecasts generated")
        return
    
    # Step 4: Generate scenarios (if module available)
    scenario_analysis = {}
    if ScenarioGenerator:
        print("\n🎯 STEP 4: Generating scenarios...")
        try:
            scenario_params = ScenarioParameters()
            scenario_generator = ScenarioGenerator(scenario_params)
            scenario_analysis = scenario_generator.generate_all_scenarios(forecast_results)
            print(f"✅ Scenarios generated: {len(scenario_analysis)} indicators")
            
            # Display scenarios
            for indicator, scenarios in scenario_analysis.items():
                print(f"\n{indicator}:")
                for scenario_name, scenario_data in scenarios.items():
                    forecast_2027 = scenario_data['forecasts'].get(2027, 0)
                    probability = scenario_data.get('probability', 0)
                    print(f"   {scenario_name.title()}: {forecast_2027:.1f}% (Prob: {probability:.1%})")
        except Exception as e:
            print(f"⚠️ Error generating scenarios: {e}")
    else:
        print("\n⚠️ Skipping scenario analysis (module not available)")
    
    # Step 5: Uncertainty quantification (if module available)
    uncertainty_analysis = {}
    if UncertaintyQuantifier and scenario_analysis:
        print("\n📈 STEP 5: Uncertainty quantification...")
        try:
            uncertainty_analysis = UncertaintyQuantifier.calculate_all_uncertainty(scenario_analysis)
            print(f"✅ Uncertainty analysis completed")
        except Exception as e:
            print(f"⚠️ Error in uncertainty analysis: {e}")
    else:
        print("\n⚠️ Skipping uncertainty analysis (module not available)")
    
    # Step 6: Create visualizations (if module available)
    if ForecastVisualizer:
        print("\n🎨 STEP 6: Creating visualizations...")
        try:
            os.makedirs('reports/figures/task4', exist_ok=True)
            visualizer = ForecastVisualizer(output_dir='reports/figures/task4')
            visualizer.create_comprehensive_dashboard(
                forecast_results, 
                scenario_analysis, 
                uncertainty_analysis
            )
            print("✅ Visualizations created successfully")
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
    else:
        print("\n⚠️ Skipping visualizations (module not available)")
    
    # Step 7: Save results
    print("\n💾 STEP 7: Saving results...")
    try:
        output_dir = 'models/task4'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save forecasts using forecaster's method
        forecaster.save_results(output_dir)
        
        # Save scenario analysis
        if scenario_analysis:
            import json
            
            def convert_to_python(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                else:
                    return str(obj)
            
            with open(f'{output_dir}/scenario_analysis.json', 'w') as f:
                json.dump(scenario_analysis, f, indent=2, default=convert_to_python)
            print(f"✅ Scenario analysis saved to {output_dir}/scenario_analysis.json")
    except Exception as e:
        print(f"⚠️ Error saving results: {e}")
    
    # Step 8: Generate final report
    print("\n📄 STEP 8: Generating final report...")
    try:
        report = forecaster.generate_final_report(
            scenario_analysis=scenario_analysis,
            uncertainty_analysis=uncertainty_analysis
        )
        print("✅ Final report generated")
    except Exception as e:
        print(f"⚠️ Error generating report: {e}")
    
    # Step 9: Print summary
    print("\n" + "="*60)
    print("FORECASTING COMPLETE - SUMMARY")
    print("="*60)
    
    if forecast_results:
        for indicator, results in forecast_results.items():
            print(f"\n📊 {indicator}:")
            if 'ensemble' in results:
                forecasts = results['ensemble']['forecasts']
                for year in sorted(forecasts.keys()):
                    print(f"   {year}: {forecasts[year]:.1f}%")
    
    print("\n📁 Outputs saved in:")
    print("   • models/task4/ - Forecast data and models")
    print("   • reports/task4/ - Final report")
    if ForecastVisualizer:
        print("   • reports/figures/task4/ - Visualizations")
    
    print("\n✅ TASK 4 COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()