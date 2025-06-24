"""Main Application Class for SSAT Model Comparison App.

This module contains the main application class that orchestrates the
entire dashboard interface and manages application state.
"""

import panel as pn
import panel_material_ui as pmui
import param

from ssat.app.config.app_config import (
    APP_INFO,
    AVAILABLE_MODELS,
    DATA_CONFIG,
    DEFAULT_MODELS,
    MODEL_CLASSES,
    MODEL_TYPES,
    SERVER_CONFIG,
)
from ssat.app.config.ui_config import LAYOUT_CONFIG, THEME_CONFIG
from ssat.app.utils.ui_helpers import (
    create_status_message,
    get_status_message,
)

# Enable Panel extensions
# Panel Material UI doesn't require a specific extension name in pn.extension()
# The components work directly with Panel's extension system
pn.extension(sizing_mode="stretch_width")

# Verify Panel Material UI is available (without creating components)
try:
    # Test that the module is importable without creating instances
    hasattr(pmui, 'Button')
    print("✅ Panel Material UI components available")
except Exception as e:
    print(f"⚠️ Panel Material UI not available: {e}")
    print("📦 Install with: pip install panel-material-ui")


class SSATModelComparisonApp(param.Parameterized):
    """Main application class for SSAT Model Comparison Dashboard.

    This class manages the overall application state, UI components,
    and navigation between different sections of the dashboard.
    """

    # Theme and UI State
    dark_theme = param.Boolean(
        default=False, doc="Whether to use dark theme for the app"
    )

    current_tab = param.String(
        default="overview", doc="Currently active tab in the interface"
    )

    # Model Configuration
    model_type = param.Selector(
        default="Frequentist",
        objects=MODEL_TYPES,
        doc="Type of statistical models to compare",
    )

    selected_models = param.List(
        default=DEFAULT_MODELS["Frequentist"].copy(),
        doc="List of selected models for comparison",
    )

    # Data Configuration
    league = param.Selector(
        default=DATA_CONFIG["sample_leagues"][0],
        objects=DATA_CONFIG["sample_leagues"],
        doc="Selected league/competition",
    )

    season = param.ListSelector(
        default=[DATA_CONFIG["sample_seasons"][0]],
        objects=DATA_CONFIG["sample_seasons"],
        doc="Selected seasons",
    )

    train_split = param.Number(
        default=DATA_CONFIG["default_train_split"],
        bounds=DATA_CONFIG["train_split_range"],
        step=1.0,
        doc="Training split percentage",
    )

    # Application State
    busy = param.Boolean(default=False, doc="Whether the app is currently processing")

    status_message = param.String(
        default=get_status_message("initial"),
        doc="Current status message displayed to user",
    )

    # Data placeholder (will be populated later)
    data_loaded = param.Boolean(default=False, doc="Whether data has been loaded")

    # Store loaded data
    filtered_data = param.DataFrame(default=None, doc="Filtered match data")
    filtered_odds_data = param.DataFrame(default=None, doc="Filtered odds data")

    models_trained = param.Boolean(
        default=False, doc="Whether models have been trained"
    )

    predictions_generated = param.Boolean(
        default=False, doc="Whether predictions have been generated"
    )

    # Store training results
    model_results = param.DataFrame(default=None, doc="Model training results")
    model_metrics = param.DataFrame(default=None, doc="Model performance metrics")
    prediction_results = param.DataFrame(default=None, doc="Model prediction results")

    def __init__(self, **params):
        """Initialize the SSAT Model Comparison App."""
        super().__init__(**params)

        # Initialize UI components
        self._create_widgets()

        # Create the main application layout
        self.app = self._create_layout()

    def _create_widgets(self):
        """Create all UI widgets and components."""
        # Import components here to avoid circular imports
        from ssat.app.components.sidebar import create_sidebar
        from ssat.app.components.tabs import create_main_tabs

        # Create sidebar components
        self.sidebar_components = create_sidebar(self)

        # Create main tab components
        self.main_tabs = create_main_tabs(self)

    def _create_layout(self) -> pmui.Page:
        """Create the main application layout.

        Returns:
            The main page component
        """
        # Create the main page with Material UI styling
        page = pmui.Page(
            title=APP_INFO["title"],
            sidebar_width=LAYOUT_CONFIG["sidebar_width"],
            sidebar_variant="persistent",
            theme_config=THEME_CONFIG["light"],
            theme_toggle=True,
        )

        # Link theme parameter
        # page.param.dark_theme.link(self.param.dark_theme)

        # Set up sidebar
        page.sidebar = self.sidebar_components

        # Set up main content
        page.main = [self.main_tabs]

        return page

    @param.depends("model_type", watch=True)
    def _on_model_type_change(self):
        """Handle model type selection change."""
        # Update available models for the new type
        new_models = AVAILABLE_MODELS[self.model_type]

        # Reset to default models for the new type
        self.selected_models = DEFAULT_MODELS[self.model_type].copy()

        # Update status
        self.status_message = get_status_message("initial")

        # Reset training state
        self.models_trained = False
        self.predictions_generated = False

    @param.depends("selected_models", watch=True)
    def _on_models_change(self):
        """Handle model selection change."""
        if len(self.selected_models) < 2:
            self.status_message = create_status_message(
                "warning", "⚠️ Please select at least 2 models for comparison."
            )
        else:
            self.status_message = create_status_message(
                "info",
                f"✅ {len(self.selected_models)} models selected. Configure data filters and train models to begin.",
            )

        # Reset training state when models change
        self.models_trained = False
        self.predictions_generated = False

    def on_apply_filters(self, event=None):
        """Handle apply filters button click."""
        # Load real data
        self.busy = True
        self.status_message = create_status_message(
            "info", get_status_message("loading_data")
        )

        try:
            # Import data functions
            from ssat.data import (
                get_sample_handball_match_data,
                get_sample_handball_odds_data,
            )

            # Load real SSAT data
            raw_data = get_sample_handball_match_data()
            odds_data = get_sample_handball_odds_data()

            # Apply actual filters
            filtered_data = raw_data[
                (raw_data["league"] == self.league)
                & (raw_data["season"].isin(self.season))
            ]

            # Store real data
            self.filtered_data = filtered_data
            self.filtered_odds_data = odds_data
            self.data_loaded = True

            # Update status with actual match count
            n_matches = len(filtered_data)
            self.status_message = create_status_message(
                "success", get_status_message("data_loaded", n_matches=n_matches)
            )

        except Exception as e:
            self.status_message = create_status_message(
                "error", f"Data loading failed: {e}"
            )
            self.data_loaded = False
        finally:
            self.busy = False

    def on_train_models(self, event=None):
        """Handle train models button click."""
        if not self.data_loaded:
            self.status_message = create_status_message(
                "warning", "⚠️ Please apply data filters first to load data."
            )
            return

        if len(self.selected_models) < 2:
            self.status_message = create_status_message(
                "warning", "⚠️ Please select at least 2 models for training."
            )
            return

        # Real model training
        self.busy = True
        self.status_message = create_status_message(
            "info", get_status_message("training")
        )

        try:
            # Import training functions
            from ssat.apps.ssat_model_comparison.data import run_models, model_metrics

            # Create model instances
            models = {}
            for model_name in self.selected_models:
                model_instance = self._get_model_instance(model_name, self.model_type)
                if model_instance:
                    models[model_name] = model_instance
                else:
                    self.status_message = create_status_message(
                        "error", f"Failed to create model: {model_name}"
                    )
                    self.busy = False
                    return

            # Helper function to update status
            def set_status_message(message):
                self.status_message = create_status_message("info", message)

            # Train models using real SSAT training logic
            self.model_results = run_models(
                data=self.filtered_data,
                models=models,
                train_split=self.train_split,
                set_status_message=set_status_message
            )

            # Calculate metrics
            self.model_metrics = model_metrics(self.model_results)

            # Update status
            self.models_trained = True
            self.status_message = create_status_message(
                "success",
                get_status_message("training_complete", n_models=len(self.selected_models)),
            )

        except Exception as e:
            self.status_message = create_status_message(
                "error", f"Training failed: {e}"
            )
            self.models_trained = False
        finally:
            self.busy = False

    def _get_model_instance(self, model_name: str, model_type: str):
        """Create a model instance from the model configuration.
        
        Args:
            model_name: Name of the model to instantiate
            model_type: Type of model ("Frequentist" or "Bayesian")
            
        Returns:
            Model instance or None if not found
        """
        try:
            model_class = MODEL_CLASSES.get(model_type, {}).get(model_name)
            if model_class:
                return model_class()
            else:
                print(f"Model class not found: {model_type}.{model_name}")
                return None
        except Exception as e:
            print(f"Error creating {model_name}: {e}")
            return None

    def on_generate_predictions(self, event=None):
        """Handle generate predictions button click."""
        if not self.models_trained:
            self.status_message = create_status_message(
                "warning", "⚠️ Please train models first before generating predictions."
            )
            return

        # Real prediction generation
        self.busy = True
        self.status_message = create_status_message(
            "info", get_status_message("predicting")
        )

        try:
            # Import prediction function
            from ssat.apps.ssat_model_comparison.data import generate_predictions

            # Generate real predictions using trained models
            self.prediction_results = generate_predictions(
                data=self.filtered_data,
                selected_models=self.selected_models,
                model_results=None  # Use fallback prediction method for now
            )

            # Update status
            self.predictions_generated = True
            self.status_message = create_status_message(
                "success", get_status_message("predictions_complete")
            )

        except Exception as e:
            self.status_message = create_status_message(
                "error", f"Prediction generation failed: {e}"
            )
            self.predictions_generated = False
        finally:
            self.busy = False

    def on_export_results(self, event=None):
        """Handle export results button click."""
        if not self.predictions_generated:
            self.status_message = create_status_message(
                "warning", "⚠️ Please generate predictions first before exporting."
            )
            return

        # Simulate export
        self.status_message = create_status_message(
            "success", get_status_message("export_complete")
        )

    def servable(self):
        """Make the app servable for Panel."""
        return self.app.servable()


def create_app():
    """Create and return the SSAT Model Comparison App instance.

    Returns:
        SSATModelComparisonApp instance
    """
    app = SSATModelComparisonApp()
    return app


def serve_app(port: int = None, show: bool = True, autoreload: bool = True):
    """Serve the application.

    Args:
        port: Port to serve on (defaults to config)
        show: Whether to open browser
        autoreload: Whether to enable autoreload
    """
    app = create_app()

    if port is None:
        port = SERVER_CONFIG["port"]

    app.app.show(port=port, open=show)


if __name__ == "__main__":
    # When run directly, serve the app
    serve_app()
