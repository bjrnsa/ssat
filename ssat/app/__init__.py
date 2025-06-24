"""SSAT Model Comparison App.

A clean, modular web application for comparing statistical sports models
from the SSAT (Statistical Sports Analysis Toolkit) library.
"""

__version__ = "0.1.0"


def main():
    """Entry point for the SSAT Model Comparison App.
    
    This function is called when running 'app' from the command line
    after installation via pip/uv.
    """
    from ssat.app.main import serve_app
    serve_app()
