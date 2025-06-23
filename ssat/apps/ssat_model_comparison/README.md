# SSAT Interactive Model Comparison Dashboard

An interactive Panel application for comparing statistical models from the SSAT (Statistical Sports Analysis Toolkit) library. This dashboard provides a user-friendly interface for model selection, training, and performance comparison with rich visualizations using **real handball match data** from the SSAT library.

## Features

### 🎯 Core Functionality
- **Data-Agnostic Design**: Works with any dataset conforming to the required schema
- **Runtime Data Switching**: Change datasets at runtime with automatic app reset
- **Real Data Integration**: Uses actual handball match data via `ssat.data.get_sample_handball_match_data()` by default
- **Interactive Data Exploration**: Explore data with Panel Graphic Walker (Tableau-like interface)
- **Model Selection**: Choose from 10+ Frequentist and Bayesian models
- **Interactive Training**: Configure parameters and train models with real-time feedback
- **Performance Comparison**: Side-by-side analysis of accuracy, MAE, and log-likelihood
- **Prediction Analysis**: Visualize win probabilities and goal spread predictions
- **Model Agreement**: Correlation analysis between different model predictions

### 📊 Data Requirements & Flexibility
- **Required Schema**: `home_team`, `away_team`, `home_goals`, `away_goals` columns
- **Optional Columns**: `league`, `season`, `goal_diff`, `spread` (auto-calculated if missing)
- **Dataset Independence**: Automatically adapts to new team names, leagues, and seasons
- **Reset Behavior**: When data changes, all options, defaults, and widget values recalculate
- **Safe Fallbacks**: Gracefully handles missing columns or incomplete data

### 🏗️ Architecture
- **Parameter-Driven Design**: Built using `param.Parameterized` for robust state management
- **Reactive UI**: All widgets are controlled by parameters and update automatically
- **Busy State Management**: Global "busy" parameter disables widgets during processing
- **Bidirectional Binding**: Changes to widgets update parameters and vice versa
- **Event-Driven Logic**: Parameter watchers handle state changes seamlessly

### 📊 Data & Visualizations
- **Real match data**: 1,449 handball matches from multiple leagues and seasons
- **125 unique teams** across 7 different competitions
- **Multi-league support**: European Championship, Liga ASOBAL, Starligue, and more
- **Interactive data explorer**: Drag-and-drop interface for creating custom visualizations
- **Tableau-like experience**: Professional data exploration without coding
- Performance metrics bar charts with detailed statistics
- Probability heatmaps for match outcome predictions
- Scatter plots for predicted goal spreads
- Correlation matrices for model agreement analysis
- Interactive parameter tuning with real-time updates

### 🎨 UI Components
- Material UI design with Panel Material UI components
- **Source Data Tab**: Interactive data exploration with Panel Graphic Walker
- **Icon-Enhanced Buttons**: Train (🎓), Predict (📊), Export (💾) with Material UI icons
- Responsive layout with sidebar controls and tabbed main content
- **Responsive sizing**: Components automatically stretch to fit available width
- **Adaptive layout**: Optimized for different screen sizes and viewports
- **Clean Progress Feedback**: Built-in template progress indicators (no redundant progress bars)
- Export functionality for results
- Comprehensive documentation tab

### 📱 Responsive Design Features
- **Stretch width sizing**: All components automatically fit the available width
- **Flexible layouts**: Cards and containers adapt to screen size
- **Sidebar optimization**: Fixed-width sidebar with responsive content
- **Mobile-friendly**: Interface works well on tablets and smaller screens
- **Consistent spacing**: Proper margins and padding throughout the interface

## Installation & Setup

### Prerequisites
```bash
# Install required packages
pip install panel panel-material-ui panel-graphic-walker pandas numpy matplotlib seaborn ssat

# Or use uv for faster installation
uv pip install panel panel-material-ui panel-graphic-walker pandas numpy matplotlib seaborn ssat
```

**Key Dependencies:**
- `panel>=1.4.0`: Web application framework
- `panel-material-ui>=1.0.0`: Material Design components
- `panel-graphic-walker>=0.5.0`: Interactive data exploration (Tableau-like interface)
- `ssat>=0.0.3`: Statistical Sports Analysis Toolkit with real data
- `pandas`, `numpy`: Data manipulation and analysis
- `matplotlib`, `seaborn`: Visualization libraries

### Running the Application

#### Option 1: Standalone Mode
```bash
cd /path/to/panel-material-ui/examples/apps/ssat_model_comparison/
# Set environment variable for WebSocket origin (automatically set in code)
export BOKEH_ALLOW_WS_ORIGIN=localhost:5007
python app.py
```
The app will be available at `http://localhost:5007`

#### Option 2: Panel Serve
```bash
export BOKEH_ALLOW_WS_ORIGIN=localhost:5007
panel serve app.py --port 5007 --allow-websocket-origin=localhost:5007 --autoreload
```

#### Option 3: Jupyter Integration
```python
import panel as pn
from app import create_app

pn.extension()
app = create_app()
app.servable()
```

## Usage Guide

### 1. Using Custom Data

The app is designed to work with any sports dataset that has the required schema. Here's how to use your own data:

#### Required Data Schema
```python
import pandas as pd

# Your data must have these columns
custom_data = pd.DataFrame({
    'home_team': ['Team A', 'Team B', ...],
    'away_team': ['Team B', 'Team C', ...], 
    'home_goals': [25, 30, ...],  # or points, scores, etc.
    'away_goals': [23, 28, ...],  # or points, scores, etc.
})

# Optional columns (will be auto-calculated if missing)
# 'league': ['League 1', 'Premier League', ...]
# 'season': [2024, 2023, ...]
# 'goal_diff': home_goals - away_goals
# 'spread': home_goals - away_goals
```

#### Loading Custom Data
```python
from ssat.apps.ssat_model_comparison.app import SSATModelComparisonApp

# Create app with your data
app = SSATModelComparisonApp(data=custom_data)

# Or change data at runtime (triggers full reset)
app.data = new_custom_data
```

#### Data-Agnostic Features
- **Automatic Reset**: When data changes, all options and widgets recalculate
- **Dynamic Options**: League and season dropdowns update based on your data
- **Team Adaptation**: Works with any team names or number of teams
- **Flexible Sports**: Works with football, basketball, hockey, handball, etc.
- **Safe Fallbacks**: Handles missing optional columns gracefully

### 2. Model Configuration
- **Model Type**: Select between "Frequentist" or "Bayesian" approaches
- **Model Selection**: Choose 2 or more models from the available options
- **Data Filters**: Configure league and season (using sample data)
- **Training Split**: Adjust the percentage of data used for training

### 2. Model Training
- Click "Train Models" to fit selected models to the data
- Monitor progress with the built-in progress indicator
- View training status updates in real-time

### 3. Analysis & Comparison
- **Overview Tab**: Dataset summary and training status
- **Performance Tab**: Model accuracy, MAE, and log-likelihood comparison
- **Predictions Tab**: Win probability analysis and prediction correlation
- **Source Data Tab**: Interactive data exploration with Graphic Walker
- **Documentation Tab**: Detailed usage instructions and model descriptions

### 4. Interpreting Results

#### Performance Metrics
- **Accuracy**: Percentage of correct outcome predictions (higher = better)
- **MAE (Mean Absolute Error)**: Average prediction error in goals (lower = better)
- **Log-Likelihood**: Model fit quality (less negative = better)

#### Prediction Analysis
- **Home Win Probabilities**: Heatmap showing predicted home team win chances
- **Predicted Spreads**: Scatter plot of expected goal differences
- **Average Probabilities**: Bar chart comparing win/draw/loss rates by model
- **Model Correlation**: How closely different models agree on predictions

### 5. Interactive Data Exploration

The **Source Data** tab provides a powerful, Tableau-like interface for exploring the handball match data:

#### Key Features
- **Drag-and-Drop Interface**: Create visualizations by dragging fields to different areas
- **Multiple Chart Types**: Automatically suggests appropriate visualizations based on data types
- **Interactive Filtering**: Filter data by any field to focus on specific subsets
- **Statistical Profiling**: Get detailed statistical summaries of all data columns
- **Export Capabilities**: Save your custom visualizations and specifications

#### How to Use the Data Explorer
1. **Switch to Data Tab**: View the raw data table with sorting and filtering
2. **Create Visualizations**: Switch to Vis tab and drag fields to create charts
3. **Explore Patterns**: Use the interactive interface to discover insights
4. **Profile Data**: Click on the profiler to get statistical summaries

This is the same real data used by the model training and prediction functionality.

## Data Source

The dashboard now uses **real handball match data** from the SSAT library:

```python
from ssat.data import get_sample_handball_match_data

# Load actual match data
df = get_sample_handball_match_data()
print(f"Data shape: {df.shape}")  # (1449, 10)
print(f"Leagues: {df['league'].unique()}")
# ['European Championship', 'Liga ASOBAL', 'Starligue', ...]
```

### Data Overview
- **1,449 real handball matches** from international competitions
- **125 unique teams** from multiple countries and leagues
- **7 different leagues**: European Championship, Liga ASOBAL, Starligue, Handbollsligan Women, etc.
- **3 seasons** of data: 2024, 2025, 2026
- **Complete match information**: Goals, teams, dates, results, and computed statistics

### Available Leagues
- European Championship (International)
- Liga ASOBAL (Spain)
- Starligue (France) 
- Herre Handbold Ligaen (Denmark)
- Kvindeligaen Women (Denmark)
- Handbollsligan Women (Sweden)
- EHF Euro Cup (European)

## Available Models

### Frequentist Models
| Model | Description | Best For |
|-------|-------------|----------|
| **Bradley-Terry** | Paired comparison with logistic regression | Team rankings, win probabilities |
| **GSSD** | Generalized Scores Standard Deviation | Detailed performance analysis |
| **Poisson** | Classical goal-scoring model | Traditional sports modeling |
| **TOOR** | Team Offense-Offense Rating | Offensive performance focus |
| **ZSD** | Zero-Score Distribution | Low-scoring sports |
| **PRP** | Possession-based Rating Process | Possession-heavy sports |

### Bayesian Models
| Model | Description | Best For |
|-------|-------------|----------|
| **Poisson** | Bayesian goal-scoring with MCMC | Uncertainty quantification |
| **NegBinom** | Overdispersed goal modeling | High-variance scoring |
| **Skellam** | Direct goal difference modeling | Spread betting analysis |
| **SkellamZero** | Zero-inflated for frequent draws | Sports with many draws |

## Technical Architecture

### Framework Stack
- **Panel**: Web application framework
- **Panel Material UI**: Modern Material Design components
- **Matplotlib**: Statistical visualizations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Seaborn**: Enhanced statistical plotting

### Application Structure
```
app.py
├── SSATModelComparisonApp (Main Application Class)
│   ├── Widget Creation & Management
│   ├── Model Training Simulation
│   ├── Visualization Generation
│   └── Layout Management
├── Data Management
│   ├── Sample Data Generation
│   ├── Model Results Storage
│   └── Prediction Comparison
└── UI Components
    ├── Sidebar Controls
    ├── Main Content Tabs
    ├── Interactive Plots
    └── Status Updates
```

### Parameter-Driven Architecture

The application is built using `param.Parameterized` for robust, reactive state management:

#### Core Parameters
```python
class SSATModelComparisonApp(param.Parameterized):
    # UI state parameters
    model_type = param.Selector(default='Frequentist', objects=['Frequentist', 'Bayesian'])
    selected_models = param.List(default=['Bradley-Terry', 'GSSD'])
    league = param.Selector(default='European Championship')
    season = param.Integer(default=2026)
    train_split = param.Number(default=80.0, bounds=(50.0, 90.0))
    
    # Process state parameters  
    busy = param.Boolean(default=False, doc="Global busy state for UI control")
    models_trained = param.Boolean(default=False)
    predictions_generated = param.Boolean(default=False)
    status_message = param.String(default="Select models and click 'Train Models'")
```

#### Key Benefits
- **Centralized State**: All application state is managed through parameters
- **Automatic UI Updates**: Widgets automatically reflect parameter changes
- **Busy State Control**: Single `busy` parameter disables all widgets during processing
- **Parameter Validation**: Built-in type checking and validation
- **Event-Driven Logic**: Parameter watchers handle state transitions

#### Parameter Watchers
```python
@param.depends('model_type', watch=True)
def _on_model_type_change(self, *events):
    """Update available models when model type changes"""
    self.model_select.options = self.available_models[self.model_type]

@param.depends('busy', watch=True) 
def _on_busy_change(self, *events):
    """Disable/enable all widgets based on busy state"""
    disabled = bool(self.busy)
    self.train_button.disabled = disabled
    # ... disable all other widgets
```

#### Widget-Parameter Binding
```python
# Bidirectional binding between widgets and parameters
self.model_type_select.link(self, value='model_type')
self.train_split_slider.link(self, value='train_split')
```

## Customization Options

### Adding New Models
To add support for additional SSAT models:

1. Update the `available_models` dictionary in `__init__`:
```python
self.available_models = {
    'Frequentist': ['Bradley-Terry', 'GSSD', 'YourNewModel'],
    'Bayesian': ['Poisson', 'Skellam', 'YourNewBayesianModel']
}
```

2. Add model characteristics in `_simulate_model_training`:
```python
elif model_name == 'YourNewModel':
    accuracy = 0.70 + np.random.normal(0, 0.02)
    mae = 3.5 + np.random.normal(0, 0.3)
    log_likelihood = -140 + np.random.normal(0, 10)
```

### Styling Customization
The app uses Material UI theming which can be customized:
```python
template = pn.template.MaterialTemplate(
    title="Your Custom Title",
    header_background='#YOUR_COLOR',
    theme=pn.template.LightTheme,  # or DarkTheme
)
```

### Data Integration
To use real SSAT data instead of simulated data:
```python
def _load_sample_data(self) -> pd.DataFrame:
    # Replace with actual SSAT data loading
    return pd.read_parquet("your_actual_data.parquet")
```

## Development Notes

### Performance Considerations
- Model training is currently simulated for demonstration
- Real SSAT integration would require longer processing times
- Consider implementing background tasks for actual model training
- Progress indicators provide user feedback during long operations

### Future Enhancements
- Real-time data integration
- Advanced model hyperparameter tuning
- Export to multiple formats (PDF, JSON, etc.)
- Model ensemble creation and comparison
- Integration with betting odds APIs
- Team-specific analysis views

## Troubleshooting

### Common Issues
1. **Port Already in Use**: Change the port in the run command
2. **Import Errors**: Ensure all dependencies are installed
3. **Visualization Issues**: Check matplotlib backend configuration
4. **Memory Usage**: Large datasets may require optimization

### Support
For issues specific to:
- **Panel**: Check [Panel documentation](https://panel.holoviz.org/)
- **Panel Material UI**: See [Panel Material UI docs](https://panel-material-ui.holoviz.org/)
- **SSAT Library**: Refer to SSAT documentation and examples

## License

This application is part of the Panel Material UI examples and follows the same licensing terms.
