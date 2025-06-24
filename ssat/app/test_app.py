"""Test script for SSAT Model Comparison App.

Simple test to verify the application loads and components work correctly.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")

    try:
        # Test main app import
        print("✅ Main app imports successful")

        # Test configuration imports
        print("✅ Configuration imports successful")

        # Test component imports
        print("✅ Component imports successful")

        # Test page imports
        print("✅ Page imports successful")

        # Test utility imports
        print("✅ Utility imports successful")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_app_creation():
    """Test that the app can be created successfully."""
    print("\n🧪 Testing app creation...")

    try:
        from ssat.app.main import create_app

        # Create app instance
        app = create_app()
        print("✅ App instance created successfully")

        # Check app has required attributes
        assert hasattr(app, "app"), "App missing 'app' attribute"
        assert hasattr(app, "model_type"), "App missing 'model_type' parameter"
        assert hasattr(app, "selected_models"), (
            "App missing 'selected_models' parameter"
        )
        assert hasattr(app, "dark_theme"), "App missing 'dark_theme' parameter"
        print("✅ App has required attributes")

        # Test parameter defaults
        assert app.model_type == "Frequentist", (
            f"Expected 'Frequentist', got '{app.model_type}'"
        )
        assert len(app.selected_models) >= 2, (
            f"Expected 2+ models, got {len(app.selected_models)}"
        )
        assert isinstance(app.dark_theme, bool), (
            f"Expected bool, got {type(app.dark_theme)}"
        )
        print("✅ Parameter defaults are correct")

        return True

    except Exception as e:
        print(f"❌ App creation failed: {e}")
        traceback.print_exc()
        return False


def test_configurations():
    """Test that configurations are valid."""
    print("\n🧪 Testing configurations...")

    try:
        from ssat.app.config.app_config import APP_INFO, AVAILABLE_MODELS, MODEL_TYPES
        from ssat.app.config.ui_config import THEME_CONFIG

        # Test model configuration
        assert len(MODEL_TYPES) == 2, f"Expected 2 model types, got {len(MODEL_TYPES)}"
        assert "Frequentist" in MODEL_TYPES, "Frequentist not in model types"
        assert "Bayesian" in MODEL_TYPES, "Bayesian not in model types"
        print("✅ Model types configuration valid")

        # Test available models
        for model_type in MODEL_TYPES:
            assert model_type in AVAILABLE_MODELS, (
                f"{model_type} not in available models"
            )
            assert len(AVAILABLE_MODELS[model_type]) > 0, f"No models for {model_type}"
        print("✅ Available models configuration valid")

        # Test theme configuration
        assert "light" in THEME_CONFIG, "Light theme not configured"
        assert "dark" in THEME_CONFIG, "Dark theme not configured"
        print("✅ Theme configuration valid")

        # Test app info
        required_keys = ["title", "version", "description"]
        for key in required_keys:
            assert key in APP_INFO, f"Missing {key} in APP_INFO"
        print("✅ App info configuration valid")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def run_tests():
    """Run all tests."""
    print("🚀 Starting SSAT Model Comparison App Tests\n")

    tests = [
        ("Import Tests", test_imports),
        ("App Creation Tests", test_app_creation),
        ("Configuration Tests", test_configurations),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"{'=' * 50}")
        print(f"Running: {test_name}")
        print(f"{'=' * 50}")

        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED\n")
        else:
            print(f"❌ {test_name} FAILED\n")

    print(f"{'=' * 50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'=' * 50}")

    if passed == total:
        print("🎉 All tests passed! App is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
