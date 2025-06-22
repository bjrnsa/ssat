#!/usr/bin/env python3
"""
Demo script for the SSAT Model Comparison Dashboard

This script demonstrates how to run the interactive model comparison
dashboard in different modes.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import panel as pn
    from app import create_app
    
    def main():
        """Main demo function."""
        print("🚀 SSAT Model Comparison Dashboard Demo")
        print("=" * 50)
        print()
        
        # Enable Panel extensions
        pn.extension()
        
        # Create the application
        print("📊 Creating interactive dashboard...")
        app = create_app()
        
        # Choose serving method
        print("\nChoose how to run the dashboard:")
        print("1. Open in browser (default)")
        print("2. Show in notebook/panel serve")
        print("3. Just create app object")
        
        choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
        
        if choice == "1":
            print("\n🌐 Opening dashboard in browser...")
            print("Dashboard will be available at: http://localhost:5007")
            print("Press Ctrl+C to stop the server")
            app.show(port=5007, autoreload=True)
            
        elif choice == "2":
            print("\n📱 Making app servable...")
            app.servable()
            print("Use 'panel serve demo.py --show' to run")
            
        elif choice == "3":
            print("\n✅ App created successfully!")
            print("App object available as 'app' variable")
            return app
            
        else:
            print("❌ Invalid choice. Exiting.")
            
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("\n💡 Please install required packages:")
    print("pip install panel panel-material-ui pandas numpy matplotlib seaborn")
    print("\nOr use:")
    print("pip install -r requirements.txt")
    
except Exception as e:
    print(f"❌ Error creating dashboard: {e}")
    print("\n🔧 Troubleshooting tips:")
    print("1. Ensure all dependencies are installed")
    print("2. Check that port 5007 is available")
    print("3. Verify Python version compatibility (3.8+)")
