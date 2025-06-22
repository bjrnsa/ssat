#!/usr/bin/env python3
"""
SSAT Interactive Model Comparison Dashboard

This application provides an interactive interface for comparing different statistical models
from the SSAT (Statistical Sports Analysis Toolkit) library. Users can compare Bayesian
and Frequentist models side-by-side, analyze team performance, and visualize predictions.

Key Features:
- Model selection and comparison
- Interactive parameter tuning
- Team performance visualization
- Real-time prediction updates
- Export capabilities
"""

import os
import panel as pn
import panel_material_ui as pmui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import warnings
import param
from ssat.apps.ssat_model_comparison.plots import create_performance_comparison_plot, _create_prediction_comparison_plot
from ssat.apps.ssat_model_comparison.config import MODELS, MODEL_CLASSES
from ssat.apps.ssat_model_comparison.data import train_models, generate_predictions

# Import SSAT library functions
from ssat.data import get_sample_handball_match_data

# Import Panel Graphic Walker for data exploration
from panel_gwalker import GraphicWalker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette("husl")

# Panel extension with responsive sizing
pn.extension('tabulator', sizing_mode="stretch_width")

class SSATModelComparisonApp(param.Parameterized):
    """Interactive Model Comparison Dashboard for SSAT library."""
    
    # The core dataset driving the app
    data = param.DataFrame(
        default=pd.DataFrame(),
        allow_None=True,
        doc="Match data containing home_team, away_team, home_goals, away_goals columns"
    )
    
    # Core Selection Parameters independent of data
    model_type = param.Selector(
        default='Frequentist', 
        objects=['Frequentist', 'Bayesian'],
        doc="Type of statistical models to compare"
    )
    
    selected_models = param.ListSelector(
        default=['Bradley-Terry', 'GSSD'],
        objects=MODELS["Frequentist"],
        doc="List of selected models for comparison"
    )

    train_split = param.Number(
        default=80.0,
        bounds=(50.0, 90.0),
        step=1.0,
        doc="Training split percentage"
    )
    
    # Core Selection Parameters dependent on data
    league = param.Selector(
        doc="Selected league/competition"
    )
    
    season = param.Selector(
        default=2026,
        objects=list(range(2020, 2031)),
        doc="Selected season"
    )
    
    # Status parameters

    busy = param.Boolean(
        default=False,
        doc="Whether the app is currently processing (training/predicting)"
    )
    
    status_message = param.String(
        default="<p><em>Load data and select models to begin comparison.</em></p>",
        doc="Current status message displayed to user"
    )

    dark_theme = param.Boolean(
        default=False,
        doc="Whether to use dark theme for the app",
        allow_refs=True,
    )
    
    # Result Parameters

    model_results = param.Dict(
        default={},
        doc="Dictionary to store results of trained models"
    )

    comparison_data = param.DataFrame(
        default=None,
        doc="DataFrame containing comparison results for selected models"
    )
    
    def __init__(self, **params):
        """Initialize the SSAT Model Comparison App."""
        # Initialize available models
        self.available_models = MODELS
        
        # If no data is provided, load sample data
        if 'data' not in params or params['data'].empty:
            params['data'] = self._load_sample_data()
        
        super().__init__(**params)
        
        
        self._setup_parameter_options()
        self._create_widgets()
        self.app = self._create_layout()
    
    def _safe_data_access(self, column = None):
        """Safely access data with proper type checking."""
        try:
            # Check if data exists and has content
            if not hasattr(self, 'data') or self.data is None:
                return None
            
            # Convert to pandas DataFrame if needed
            if hasattr(self.data, '__len__'):
                if len(self.data) == 0:
                    return None
            else:
                return None
                
            if column:
                if hasattr(self.data, 'columns') and column in self.data.columns:
                    return self.data[column]
                else:
                    return None
            return self.data
        except (AttributeError, TypeError, KeyError):
            return None
    
    def _setup_parameter_options(self):
        """Set up parameter options based on loaded data."""
        # Check if data is available and not empty
        data = self._safe_data_access()
        if data is None:
            # Set default options when no data is available
            self.param.league.objects = ['All']
            self.param.season.objects = list(range(2020, 2030))
            self.league = 'All'
            self.season = 2026
            return
            
        # Update league options and current value
        league_column = self._safe_data_access('league')
        if league_column is not None:
            available_leagues = sorted(league_column.unique())
            self.param.league.objects = available_leagues
            if available_leagues:
                # Set the current value, not the default
                self.league = available_leagues[0]
        else:
            self.param.league.objects = ['All']
            self.league = 'All'
        
        # Update season options and current value
        season_column = self._safe_data_access('season')
        if season_column is not None:
            available_seasons = sorted(season_column.unique())
            if available_seasons:
                min_season = int(min(available_seasons))
                max_season = int(max(available_seasons))
                self.param.season.objects = list(range(min_season, max_season+1))
                # Set the current value to the latest season
                self.season = max_season
        else:
            self.param.season.objects = list(range(2020, 2030))
            self.season = 2026
    
    def _load_sample_data(self) -> pd.DataFrame:
        """Load sample handball data from SSAT library."""
        # Load real handball data from SSAT library
        df = get_sample_handball_match_data()
        
        # Add derived columns for compatibility with existing code
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        df['spread'] = df['goal_diff']
        
        return df
    
    def _create_widgets(self):
        """Create all the UI widgets driven by parameters."""
        self._model_type_select = pmui.Select.from_param(
            self.param.model_type,
            sizing_mode="stretch_width"
        )
        
        self._model_select = pmui.MultiSelect.from_param(
            self.param.selected_models,
            height=120,
            sizing_mode="stretch_width"
        )
        
        self._league_select = pmui.Select.from_param(
            self.param.league,
            sizing_mode="stretch_width"
        )
        
        self._season_select = pmui.Select.from_param(
            self.param.season,
            sizing_mode="stretch_width"
        )
        
        self._train_split_slider = pmui.FloatSlider.from_param(
            self.param.train_split,
            sizing_mode="stretch_width"
        )
        
        self._train_button = pmui.Button(
            label="Train Models",
            variant="contained",
            color="primary",
            icon="school",
            sizing_mode="stretch_width",
            disabled=self.param.busy,
            on_click=self._on_train_models,
        )
        
        self._predict_button = pmui.Button(
            label="Predict Results",
            variant="contained", 
            color="secondary",
            icon="analytics",
            sizing_mode="stretch_width",
            disabled=self.param.busy,
            on_click=self._on_generate_predictions,
        )
        
        self._export_button = pmui.Button(
            label="Export Results",
            variant="outlined",
            color="primary",
            icon="download",
            sizing_mode="stretch_width",
            disabled=self.param.busy,
            on_click=self._on_export_results,
        )
        
        self._status_text = pn.pane.HTML(
            self.param.status_message,
            margin=(10, 0),
            sizing_mode="stretch_width"
        )
    
    @param.depends('data', watch=True)
    def _reset_app_state(self, *events):
        """Handle data parameter changes - triggers full app reset."""
        # Reset all app state
        self.model_results = {}
        self.comparison_data = None
        
        # Set up parameter options based on new data
        self._setup_parameter_options()
        
        # Update league select widget options and value
        if hasattr(self, '_league_select'):
            self._league_select.options = self.param.league.objects
            self._league_select.value = self.league
        
        # Update season select widget options and value
        if hasattr(self, '_season_select'):
            # Get available seasons from new data
            season_column = self._safe_data_access('season')
            if season_column is not None:
                available_seasons = sorted(season_column.unique())
                self._season_select.options = available_seasons
                self._season_select.value = self.season
            else:
                self._season_select.options = [self.season]
                self._season_select.value = self.season
        
        # Reset status message
        self.status_message = "<p><em>Data updated. Load new data and select models to begin comparison.</em></p>"
        
    
    @param.depends('model_type', watch=True)
    def _on_model_type_change(self, *events):
        """Handle model type selection change."""
        self.param.selected_models.objects = self.available_models[self.model_type]
        self.selected_models = self.param.selected_models.objects[0:2]
    
    def _on_train_models(self, event):
        """Handle model training using parameter-driven logic."""
        # Set busy state and update status
        with self.param.update(busy=True, status_message = "<p><strong>Training models...</strong></p>"):
            self._update_model_results()
        
        selected_count = len(self.selected_models) if isinstance(self.selected_models, list) else 0
        self.status_message = f"<p><strong>✅ Training completed!</strong> Trained {selected_count} models.</p>"
        
    def _set_status_message(self, value):
        self.status_message = value

    def _update_model_results(self):
        """Train real SSAT models and store results."""
        # Get data safely
        data = self.data
        model_type = self.model_type
        models = {key: self._get_model_instance(key, model_type) for key in self.selected_models}
        
        model_results = train_models(
            data=data,
            models=models,
            train_split=self.train_split,
            model_type=self.model_type,
            set_status_message=self._set_status_message
        )

        self.model_results = model_results
    
    
    def _get_model_instance(self, model_name: str, model_type: str):
        """Get an instance of the specified model using the configuration."""
        try:
            # Get the model class from the configuration
            model_class = MODEL_CLASSES.get(model_type, {}).get(model_name)
            
            if model_class is None:
                print(f"Warning: Unknown model {model_name} for type {model_type}")
                return None
            
            # Create and return an instance of the model
            return model_class()
            
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            return None
    
    def _on_generate_predictions(self, event):
        """Handle prediction generation using parameter-driven logic."""
        # Set busy state and update status
        with self.param.update(busy=True, status_message = "<p><strong>Generating predictions...</strong></p>"):
            self._generate_sample_predictions()
        
        self.status_message = "<p><strong>✅ Predictions generated!</strong> Check the visualizations below.</p>"
    
    def _generate_sample_predictions(self):
        """Generate sample predictions for comparison."""
        # Create sample prediction data
        data = self.data
        selected_models = self.selected_models if isinstance(self.selected_models, list) else []

        predictions = generate_predictions(data, selected_models)        
        
        self.comparison_data = predictions
    
    def _on_export_results(self, event):
        """Handle results export."""
        self.status_message = "<p><strong>📁 Export functionality would save results to CSV/Excel files.</strong></p>"
    
    @param.depends("model_results", watch=True)
    def _reset_comparison_data(self):
        self.comparison_data = None
    
    def _create_data_summary_table(self) -> pn.pane.HTML:
        """Create data summary table."""
        data = self._safe_data_access()
        
        if data is not None:
            try:
                home_team_col = self._safe_data_access('home_team')
                home_goals_col = self._safe_data_access('home_goals')
                away_goals_col = self._safe_data_access('away_goals')
                goal_diff_col = self._safe_data_access('goal_diff')
                
                summary_stats = {
                    'Total Matches': len(data),
                    'Unique Teams': home_team_col.nunique() if home_team_col is not None else 'N/A',
                    'Average Home Goals': f"{home_goals_col.mean():.1f}" if home_goals_col is not None else 'N/A',
                    'Average Away Goals': f"{away_goals_col.mean():.1f}" if away_goals_col is not None else 'N/A',
                    'Home Win Rate': f"{(goal_diff_col > 0).mean():.1%}" if goal_diff_col is not None else 'N/A',
                    'Draw Rate': f"{(goal_diff_col == 0).mean():.1%}" if goal_diff_col is not None else 'N/A',
                    'Away Win Rate': f"{(goal_diff_col < 0).mean():.1%}" if goal_diff_col is not None else 'N/A'
                }
            except Exception:
                summary_stats = {
                    'Total Matches': 'N/A',
                    'Unique Teams': 'N/A',
                    'Average Home Goals': 'N/A',
                    'Average Away Goals': 'N/A',
                    'Home Win Rate': 'N/A',
                    'Draw Rate': 'N/A',
                    'Away Win Rate': 'N/A'
                }
        else:
            summary_stats = {
                'Total Matches': 'N/A',
                'Unique Teams': 'N/A',
                'Average Home Goals': 'N/A',
                'Average Away Goals': 'N/A',
                'Home Win Rate': 'N/A',
                'Draw Rate': 'N/A',
                'Away Win Rate': 'N/A'
            }
        
        html_content = "<h4>Dataset Summary</h4><table style='border-collapse: collapse; width: 100%;'>"
        for key, value in summary_stats.items():
            html_content += f"<tr><td style='border: 1px solid #ddd; padding: 8px; font-weight: bold;'>{key}</td>"
            html_content += f"<td style='border: 1px solid #ddd; padding: 8px;'>{value}</td></tr>"
        html_content += "</table>"
        
        return pn.pane.HTML(html_content)
    
    def _create_layout(self) -> pmui.Page:
        """Create the main application layout."""
        # Create Page with Material UI styling
        page = pmui.Page(
            title="SSAT: Model Comparison Dashboard",
            sidebar_width=350,
            sidebar_variant="persistent",
            theme_config={
                'palette': {
                    'primary': {
                        'main': '#2E7D32'  # Green header color
                    },
                }
            },
            theme_toggle=True,  # Allow users to switch between light/dark themes
        )

        self.dark_theme = page.param.dark_theme
        
        # Create introduction card for the sidebar
        intro_card = pn.pane.HTML("""
            <div style="text-align: center; margin-bottom: 15px;">
                <h3>🏆 Welcome to SSAT</h3>
                <p style="margin: 10px 0; font-size: 14px; line-height: 1.4;">
                    <strong>Sports Statistics Analysis Toolkit</strong><br>
                    Compare machine learning models to predict sports match outcomes
                </p>
            </div>
            """)
        
        # Sidebar controls
        controls_card = pmui.Card(
            pn.pane.HTML("<h3>Model Configuration</h3>"),
            self._model_type_select,
            self._model_select,
            pn.Spacer(height=10),
            pn.pane.HTML("<h4>Data Filters</h4>"),
            self._league_select,
            self._season_select,
            pn.Spacer(height=10),
            pn.pane.HTML("<h4>Training Parameters</h4>"),
            self._train_split_slider,
            pn.Spacer(height=15),
            self._train_button,
            self._predict_button,
            self._export_button,
            title="Controls",
            margin=10,
            sizing_mode="stretch_width"
        )
        
        # Set sidebar content
        page.sidebar = [intro_card, controls_card]
        
        # Main content tabs
        performance_plot = pn.pane.Matplotlib(
            pn.bind(create_performance_comparison_plot, self.param.model_results, dark_theme=self.param.dark_theme),
            tight=True,
            format='svg',
            dpi=100,
            sizing_mode="stretch_width",
            fixed_aspect=False,
            loading=self.param.busy,
        )
        
        prediction_plot = pn.pane.Matplotlib(
            pn.bind(_create_prediction_comparison_plot, self.param.comparison_data, dark_theme=self.param.dark_theme),
            tight=True,
            format='svg',
            dpi=100,
            sizing_mode="stretch_width",
            fixed_aspect=False,
            loading=self.param.busy,
        )
        
        # Create main content cards
        status_card = pmui.Card(
            self._status_text,
            title="Status",
            margin=10,
            sizing_mode="stretch_width"
        )
        
        data_summary_card = pmui.Card(
            self._create_data_summary_table(),
            title="Data Overview",
            margin=10,
            sizing_mode="stretch_width"
        )
        
        # Performance tab with explanatory text
        performance_explanation = pn.pane.HTML("""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
            <h3>📊 How to Interpret Model Performance</h3>
            <p><strong>Understanding the metrics:</strong></p>
            <ul>
                <li><strong>Accuracy:</strong> Percentage of correct predictions (higher is better, aim for >60%)</li>
                <li><strong>MAE (Mean Absolute Error):</strong> Average prediction error (lower is better, <1.5 is good)</li>
                <li><strong>Log-Likelihood:</strong> Model fit quality (less negative is better, closer to 0 indicates better fit)</li>
            </ul>
            <p><strong>What to look for:</strong></p>
            <ul>
                <li>Models with <em>high accuracy</em> and <em>low MAE</em> make reliable predictions</li>
                <li>Compare multiple metrics - a model might excel in one area but lag in another</li>
                <li>Bayesian models often have better uncertainty quantification than Frequentist models</li>
            </ul>
            <p><strong>Learn more:</strong> 
                <a href="https://en.wikipedia.org/wiki/Accuracy_and_precision" target="_blank">Accuracy vs Precision</a> | 
                <a href="https://en.wikipedia.org/wiki/Mean_absolute_error" target="_blank">Mean Absolute Error</a> | 
                <a href="https://en.wikipedia.org/wiki/Likelihood_function" target="_blank">Likelihood Functions</a>
            </p>
        </div>
        """)
        
        performance_card = pmui.Card(
            pn.Column(performance_plot, performance_explanation),
            title="Model Performance Comparison",
            margin=10,
            sizing_mode="stretch_width"
        )
        
        # Predictions tab with explanatory text
        predictions_explanation = pn.pane.HTML("""
        <div style="padding: 15px; background-color: #f0f8ff; border-radius: 8px; margin-bottom: 20px;">
            <h3>🔮 How to Interpret Predictions</h3>
            <p><strong>Understanding the visualizations:</strong></p>
            <ul>
                <li><strong>Win Probability Heatmap:</strong> Shows each model's confidence in home team victory (darker = higher probability)</li>
                <li><strong>Goal Spread Predictions:</strong> Expected goal difference (positive = home team favored)</li>
                <li><strong>Model Agreement:</strong> When models agree, predictions are more reliable</li>
            </ul>
            <p><strong>Making decisions:</strong></p>
            <ul>
                <li>Look for <em>consensus</em> across models for more confident predictions</li>
                <li>High win probabilities (>70%) suggest clear favorites</li>
                <li>Large goal spreads indicate expected blowouts</li>
                <li>Disagreement between models suggests uncertain outcomes</li>
            </ul>
            <p><strong>Sports Analytics Resources:</strong> 
                <a href="https://en.wikipedia.org/wiki/Sports_analytics" target="_blank">Sports Analytics Overview</a> | 
                <a href="https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/" target="_blank">FiveThirtyEight Methodology</a> | 
                <a href="https://www.pinnacle.com/en/betting-articles/educational/what-are-betting-odds" target="_blank">Understanding Odds</a>
            </p>
        </div>
        """)
        
        predictions_card = pmui.Card(
            pn.Column(prediction_plot, predictions_explanation),
            title="Prediction Analysis",
            margin=10,
            sizing_mode="stretch_width"
        )
        
        main_tabs = pmui.Tabs(
            ("Overview", pn.Column(status_card, data_summary_card)),
            ("Performance", performance_card),
            ("Predictions", predictions_card),
            ("Source Data", self._create_data_exploration_tab()),
            ("Documentation", self._create_documentation_tab()),
        )
        
        page.main = [main_tabs]
        
        return page
    
    def _create_documentation_tab(self) -> pmui.Card:
        """Create documentation tab content."""
        doc_content = """
        ## SSAT Model Comparison Dashboard
        
        This interactive dashboard allows you to compare different statistical models from the SSAT library for sports match prediction.
        
        ### Available Models
        
        **Frequentist Models:**
        - **Bradley-Terry**: Paired comparison model using logistic regression
        - **GSSD**: Generalized Scores Standard Deviation model with team offensive/defensive ratings
        - **Poisson**: Classical Poisson model for goal scoring
        - **TOOR**: Team Offense-Offense Rating model
        - **ZSD**: Zero-Score Distribution model
        - **PRP**: Possession-based Rating Process model
        
        **Bayesian Models:**
        - **Poisson**: Bayesian Poisson model with MCMC sampling
        - **NegBinom**: Negative Binomial model for overdispersed scoring
        - **Skellam**: Direct goal difference modeling
        - **SkellamZero**: Zero-inflated Skellam for frequent draws
        
        ### How to Use
        
        1. **Select Model Type**: Choose between Frequentist and Bayesian approaches
        2. **Choose Models**: Select 2 or more models to compare
        3. **Configure Data**: Set league, season, and training split parameters
        4. **Train Models**: Click "Train Models" to fit selected models to the data
        5. **Generate Predictions**: Create predictions for comparison analysis
        6. **Analyze Results**: View performance metrics and prediction comparisons
        
        ### Performance Metrics
        
        - **Accuracy**: Percentage of correct win/loss/draw predictions
        - **MAE**: Mean Absolute Error for goal spread predictions
        - **Log-Likelihood**: Model fit quality (higher is better)
        
        ### Prediction Analysis
        
        - **Probability Heatmaps**: Visualize win probabilities across matches
        - **Spread Predictions**: Compare predicted goal differences
        - **Model Agreement**: Correlation analysis between model predictions
        
        ### Export Options
        
        Results can be exported to CSV or Excel formats for further analysis.
        """
        
        return pmui.Card(
            pn.pane.Markdown(doc_content),
            title="Documentation",
            margin=10,
            sizing_mode="stretch_width"
        )

    @pn.depends('dark_theme')
    def _appearance(self)->str:
        """Toggle between light and dark themes."""
        if self.dark_theme:
            return "dark"
        return "light"
    
    def _create_data_exploration_tab(self) -> pmui.Card:
        """Create data exploration tab with Panel Graphic Walker."""
        # Create GraphicWalker for interactive data exploration
        walker = GraphicWalker(
            object=self.param.data,
            renderer='explorer',  # Use explorer renderer for full functionality
            theme_key='g2',  # Use g2 theme to match Material UI design
            appearance=self._appearance,
            height=600,  # Set appropriate height
            sizing_mode="stretch_both"
        )
        
        # Create description text
        description = """
        ### Interactive Data Exploration
        
        Use the [Graphic Walker](https://github.com/panel-extensions/panel-graphic-walker) interface below to explore the source data interactively:
        
        - **Data Tab**: View and browse the raw data table
        - **Visualization Tab**: Create custom visualizations by dragging and dropping fields
        """
        
        description_pane = pn.pane.Markdown(description, sizing_mode="stretch_width")
        
        return pmui.Card(
            description_pane,
            walker,
            title="Source Data",
            margin=10,
            sizing_mode="stretch_width"
        )

def create_app():
    """Create and return the SSAT Model Comparison App."""
    app = SSATModelComparisonApp()
    return app.app

if pn.state.served:
    # Create and serve the app
    app = SSATModelComparisonApp()
    app.app.servable()
if __name__ == "__main__":
    # For standalone running, create and show the app
    # Set Bokeh WebSocket origin for serving
    os.environ.setdefault('BOKEH_ALLOW_WS_ORIGIN', 'localhost:5007')
    app = create_app()
    app.show(port=5007, autoreload=True)
