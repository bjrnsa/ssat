"""Application Configuration for SSAT Model Comparison App.

This module contains application-wide settings, model configurations,
and other non-UI related configuration options.
"""

# Application Information
APP_INFO = {
    "title": "SSAT Model Comparison Dashboard",
    "subtitle": "Statistical Sports Analysis Toolkit",
    "version": "2.0.0",
    "description": "Compare machine learning models to predict sports match outcomes",
    "author": "SSAT Team",
}

# Server Configuration
SERVER_CONFIG = {
    "port": 5007,
    "host": "localhost",
    "autoreload": True,
    "allow_websocket_origin": ["localhost:5007"],
}

# Model Configuration with Real SSAT Imports
from ssat.frequentist import BradleyTerry, GSSD, TOOR, ZSD, PRP
from ssat.bayesian import Poisson, NegBinom, Skellam, SkellamZero, PoissonDecay, SkellamDecay

MODEL_TYPES = ["Frequentist", "Bayesian"]

# Model class mappings for instantiation
MODEL_CLASSES = {
    "Frequentist": {
        "Bradley-Terry": BradleyTerry,
        "GSSD": GSSD,
        "TOOR": TOOR,
        "ZSD": ZSD,
        "PRP": PRP,
    },
    "Bayesian": {
        "Poisson": Poisson,
        "NegBinom": NegBinom,
        "Skellam": Skellam,
        "SkellamZero": SkellamZero,
        "PoissonDecay": PoissonDecay,
        "SkellamDecay": SkellamDecay,
    },
}

AVAILABLE_MODELS = {
    "Frequentist": [
        "Bradley-Terry",
        "GSSD",
        "TOOR",
        "ZSD",
        "PRP",
    ],
    "Bayesian": [
        "Poisson",
        "NegBinom",
        "Skellam",
        "SkellamZero",
        "PoissonDecay",
        "SkellamDecay",
    ],
}

# Default Model Selections
DEFAULT_MODELS = {
    "Frequentist": ["Bradley-Terry", "GSSD"],
    "Bayesian": ["Poisson", "Skellam"],
}

# Model Descriptions (for documentation)
MODEL_DESCRIPTIONS = {
    "Frequentist": {
        "Bradley-Terry": "Paired comparison model using logistic regression for team rankings",
        "GSSD": "Generalized Scores Standard Deviation model with team offensive/defensive ratings",
        "Poisson": "Classical Poisson model for goal scoring events",
        "TOOR": "Team Offense-Offense Rating model focusing on offensive performance",
        "ZSD": "Zero-Score Distribution model for low-scoring sports",
        "PRP": "Possession-based Rating Process model for possession-heavy sports",
    },
    "Bayesian": {
        "Poisson": "Bayesian Poisson model with MCMC sampling for uncertainty quantification",
        "NegBinom": "Negative Binomial model for overdispersed goal scoring",
        "Skellam": "Direct goal difference modeling using Skellam distribution",
        "SkellamZero": "Zero-inflated Skellam model for sports with frequent draws",
        "PoissonDecay": "Poisson model with time decay for recent match weighting",
        "SkellamDecay": "Skellam model with time decay for recent match weighting",
    },
}

# Dynamic Data Configuration Functions
def get_available_leagues():
    """Get available leagues from real data."""
    try:
        from ssat.data import get_sample_handball_match_data
        data = get_sample_handball_match_data()
        return sorted(data['league'].unique().tolist())
    except Exception:
        # Fallback to hardcoded list if data loading fails
        return [
            "European Championship",
            "Liga ASOBAL", 
            "Starligue",
            "Herre Handbold Ligaen",
            "Kvindeligaen Women",
            "Handbollsligan Women",
            "EHF Euro Cup",
        ]

def get_available_seasons():
    """Get available seasons from real data."""
    try:
        from ssat.data import get_sample_handball_match_data
        data = get_sample_handball_match_data()
        return sorted(data['season'].unique().tolist())
    except Exception:
        # Fallback to hardcoded list if data loading fails
        return [2024, 2025, 2026]

# Data Configuration
DATA_CONFIG = {
    "sample_leagues": get_available_leagues(),
    "sample_seasons": get_available_seasons(),
    "train_split_range": (50.0, 90.0),
    "default_train_split": 80.0,
}

# Status Messages
STATUS_MESSAGES = {
    "initial": "👋 Welcome! Select models and configure data filters to begin comparison.",
    "loading_data": "📊 Loading data...",
    "data_loaded": "✅ Data loaded successfully! {n_matches} matches available.",
    "training": "🎓 Training models... This may take a few moments.",
    "training_complete": "✅ Training completed! {n_models} models trained successfully.",
    "predicting": "🔮 Generating predictions...",
    "predictions_complete": "✅ Predictions generated! Check the Results tab for analysis.",
    "export_complete": "💾 Results exported successfully!",
    "error": "❌ Error: {error_message}",
    "warning": "⚠️ Warning: {warning_message}",
}

# Performance Metrics Configuration
METRICS_CONFIG = {
    "primary_metrics": ["accuracy", "mae", "log_likelihood"],
    "secondary_metrics": ["brier", "log_loss", "rps", "calibration", "ignorance"],
    "metric_descriptions": {
        "accuracy": "Percentage of correct outcome predictions (higher is better)",
        "mae": "Mean Absolute Error for goal spread predictions (lower is better)",
        "log_likelihood": "Model fit quality (less negative is better)",
        "brier": "Brier score for probability predictions (lower is better)",
        "log_loss": "Logarithmic loss for predictions (lower is better)",
        "rps": "Ranked Probability Score (lower is better)",
        "calibration": "Calibration error for probabilities (lower is better)",
        "ignorance": "Ignorance score for predictions (lower is better)",
    },
}

# Export Configuration
EXPORT_CONFIG = {
    "formats": ["CSV", "Excel", "JSON"],
    "default_format": "CSV",
    "filename_template": "ssat_results_{timestamp}",
}
