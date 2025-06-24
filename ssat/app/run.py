#!/usr/bin/env python3
"""Run script for SSAT Model Comparison App.

Simple script to launch the application with default settings.
"""

import argparse
import sys


def panel_app_cli():
    """Main entry point for running the app."""
    parser = argparse.ArgumentParser(
        description="SSAT Model Comparison Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run with defaults (port 5007)
  python run.py --port 8080        # Run on custom port
  python run.py --no-browser       # Don't open browser automatically
  python run.py --no-autoreload    # Disable autoreload for production
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5007,
        help="Port to run the server on (default: 5007)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)",
    )

    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )

    parser.add_argument(
        "--no-autoreload",
        action="store_true",
        help="Disable autoreload (for production)",
    )

    parser.add_argument(
        "--test", action="store_true", help="Run tests instead of starting the app"
    )

    args = parser.parse_args()

    if args.test:
        print("Running app tests...")
        try:
            from ssat.app.test_app import run_tests

            success = run_tests()
            sys.exit(0 if success else 1)
        except ImportError as e:
            print(f"Error importing test module: {e}")
            sys.exit(1)

    # Import and run the main app
    try:
        from ssat.app.main import serve_app

        print("🚀 Starting SSAT Model Comparison Dashboard...")
        print(f"📡 Server: http://{args.host}:{args.port}")
        print("🎨 Features: Material UI, Dark/Light themes, Responsive design")
        print("🧠 Models: 12+ Frequentist and Bayesian models available")
        print("📊 Data: Real handball match data from multiple leagues")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)

        serve_app(
            port=args.port, show=not args.no_browser, autoreload=not args.no_autoreload
        )

    except ImportError as e:
        print(f"Error importing app: {e}")
        print(
            "Make sure you're running from the correct directory and have all dependencies installed."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    panel_app_cli()
