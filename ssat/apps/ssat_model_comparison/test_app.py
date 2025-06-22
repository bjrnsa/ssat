#!/usr/bin/env python3
"""
Test script for the SSAT Model Comparison Dashboard

This script performs basic functionality tests to ensure the application
works correctly.
"""

import sys
import traceback
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        import panel as pn
        print("✅ Panel imported successfully")
        
        import panel_material_ui as pmui
        print("✅ Panel Material UI imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ Seaborn imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_app_creation():
    """Test that the app can be created without errors."""
    print("\n🏗️  Testing app creation...")
    
    try:
        from app import SSATModelComparisonApp, create_app
        print("✅ App module imported successfully")
        
        # Test class instantiation
        app_instance = SSATModelComparisonApp()
        print("✅ App class instantiated successfully")
        
        # Test app creation function
        app = create_app()
        print("✅ App created successfully")
        
        return True, app_instance
        
    except Exception as e:
        print(f"❌ App creation error: {e}")
        traceback.print_exc()
        return False, None

def test_data_generation():
    """Test that sample data generation works."""
    print("\n📊 Testing data generation...")
    
    try:
        from app import SSATModelComparisonApp
        app = SSATModelComparisonApp()
        
        # Test sample data
        data = app.sample_data
        print(f"✅ Sample data created: {len(data)} matches")
        print(f"✅ Teams: {data['home_team'].nunique()} unique teams")
        print(f"✅ Columns: {list(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_model_simulation():
    """Test that model training simulation works."""
    print("\n🧠 Testing model simulation...")
    
    try:
        from app import SSATModelComparisonApp
        app = SSATModelComparisonApp()
        
        # Set up some models
        app.model_select.value = ['Bradley-Terry', 'GSSD']
        app._update_model_results()
        
        print(f"✅ Model results generated: {len(app.model_results)} models")
        for model, results in app.model_results.items():
            print(f"  {model}: accuracy={results['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model simulation error: {e}")
        return False

def test_visualization():
    """Test that visualization generation works."""
    print("\n📈 Testing visualization...")
    
    try:
        from app import SSATModelComparisonApp
        app = SSATModelComparisonApp()
        
        # Set up models and generate results
        app.model_select.value = ['Bradley-Terry', 'GSSD']
        app._update_model_results()
        
        # Test performance plot
        perf_fig = app.create_performance_comparison_plot()
        print("✅ Performance comparison plot created")
        
        # Test prediction plot
        pred_fig = app._create_prediction_comparison_plot()
        print("✅ Prediction comparison plot created")
        
        # Test data summary
        summary = app._create_data_summary_table()
        print("✅ Data summary table created")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_config():
    """Test that configuration file works."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from config import APP_CONFIG, MODEL_CONFIG, DATA_CONFIG
        print("✅ Configuration imported successfully")
        print(f"✅ App title: {APP_CONFIG['title']}")
        print(f"✅ Frequentist models: {len(MODEL_CONFIG['frequentist_models'])}")
        print(f"✅ Sample teams: {len(DATA_CONFIG['teams'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🧪 SSAT Model Comparison Dashboard Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("App Creation", test_app_creation),
        ("Data Generation", test_data_generation),
        ("Model Simulation", test_model_simulation),
        ("Visualization", test_visualization),
        ("Configuration", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if test_name == "App Creation":
                success, _ = test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n📋 Test Summary")
    print("-" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The dashboard should work correctly.")
        print("\n💡 Next steps:")
        print("1. Run: python demo.py")
        print("2. Or: python app.py")
        print("3. Or: panel serve app.py --show")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ recommended)")
        print("3. Verify panel and panel-material-ui versions")

if __name__ == "__main__":
    run_all_tests()
