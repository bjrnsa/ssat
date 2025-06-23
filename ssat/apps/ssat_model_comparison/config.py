"""
Configuration settings for the SSAT Model Comparison Dashboard
"""

# Import SSAT model classes
from ssat.frequentist import BradleyTerry, GSSD, Poisson as FrequentistPoisson, TOOR, ZSD, PRP
from ssat.bayesian import Poisson as BayesianPoisson, NegBinom, Skellam, SkellamZero

# Application settings
APP_CONFIG = {
    'title': 'SSAT Model Comparison Dashboard',
    'port': 5007,
    'theme': 'dark',  # 'light' or 'dark'
    'sidebar_width': 350,
    'header_color': '#2E7D32',
    'autoreload': True,
    'sizing_mode': 'stretch_width'  # Default sizing mode for components
}

# Model configurations with direct class references
MODEL_CLASSES = {
    'Frequentist': {
        'Bradley-Terry': BradleyTerry,
        'GSSD': GSSD,
        'Poisson': FrequentistPoisson,
        'TOOR': TOOR,
        'ZSD': ZSD,
        'PRP': PRP
    },
    'Bayesian': {
        'Poisson': BayesianPoisson,
        'NegBinom': NegBinom,
        'Skellam': Skellam,
        'SkellamZero': SkellamZero
    }
}

# Model lists for UI (derived from MODEL_CLASSES)
MODELS = {
    model_type: list(models.keys()) 
    for model_type, models in MODEL_CLASSES.items()
}

# Legacy model configuration (kept for backward compatibility)
MODEL_CONFIG = {
    'frequentist_models': MODELS['Frequentist'],
    'bayesian_models': MODELS['Bayesian'],
    'default_frequentist': ['Bradley-Terry', 'GSSD'],
    'default_bayesian': ['Poisson', 'Skellam']
}

# Data generation settings
DATA_CONFIG = {
    'n_matches': 200,
    'teams': [
        'Aalborg', 'Skjern', 'GOG', 'Kolding', 
        'Bjerringbro-Silkeborg', 'TTH Holstebro', 
        'Skanderborg', 'Fredericia', 'Mors-Thy', 'KIF Kolding'
    ],
    'leagues': ['Danish Handball League'],
    'seasons': [2024],
    'home_goals_lambda': 28,
    'away_goals_lambda': 26,
    'random_seed': 42
}

# Visualization settings
VIZ_CONFIG = {
    'figure_dpi': 100,
    'figure_format': 'svg',  # SVG format for better responsive scaling
    'figure_fixed_aspect': False,  # Allow width/height to scale independently
    'color_palette': 'husl',
    'style': 'default',
    'plot_sizes': {
        'performance': (15, 6),  # Wider for better responsive layout
        'predictions': (16, 8),  # Even wider for complex 4-subplot layout
        'summary': (12, 6)  # Increased from original
    }
}

# Performance simulation parameters
PERFORMANCE_CONFIG = {
    'bradley_terry': {'accuracy': 0.65, 'mae': 4.2, 'll': -150, 'std': 0.02},
    'gssd': {'accuracy': 0.68, 'mae': 3.8, 'll': -145, 'std': 0.02},
    'poisson_freq': {'accuracy': 0.63, 'mae': 4.5, 'll': -155, 'std': 0.02},
    'poisson_bayes': {'accuracy': 0.66, 'mae': 4.0, 'll': -148, 'std': 0.02},
    'skellam': {'accuracy': 0.67, 'mae': 3.9, 'll': -147, 'std': 0.02},
    'default': {'accuracy': 0.60, 'mae': 4.8, 'll': -160, 'std': 0.05}
}

# UI Text and labels
UI_TEXT = {
    'status_initial': "Select models and click 'Train Models' to begin comparison.",
    'status_training': "Training models...",
    'status_complete': "✅ Training completed! Trained {n} models.",
    'status_predicting': "Generating predictions...",
    'status_predicted': "✅ Predictions generated! Check the visualizations below.",
    'status_export': "📁 Export functionality would save results to CSV/Excel files.",
    'no_performance_data': 'Train models to see performance comparison',
    'no_prediction_data': 'Generate predictions to see comparison',
    'need_multiple_models': 'Need 2+ models for correlation analysis'
}
