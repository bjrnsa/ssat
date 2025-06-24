# SSAT Model Comparison Dashboard 2.0

A clean, modern web application for comparing statistical sports models from the SSAT (Statistical Sports Analysis Toolkit) library.

## ✨ Features

### 🎨 **Modern UI/UX**
- **Material Design** with Panel Material UI components
- **Dark/Light themes** with toggle support
- **Responsive design** that works on desktop, tablet, and mobile
- **Clean parameter-driven architecture** using `param.Parameterized`
- **Modular component design** for maintainability

### 🧠 **Model Analysis**
- **12+ Statistical Models** - Both Frequentist and Bayesian approaches
- **Side-by-side comparison** with comprehensive metrics
- **Real-time training** with progress feedback
- **Interactive predictions** and model agreement analysis

### 📊 **Data Integration**
- **Real sports data** from SSAT library (1,449 handball matches)
- **Multiple leagues** and seasons (European Championship, Liga ASOBAL, etc.)
- **Interactive data exploration** (placeholder for Panel Graphic Walker)
- **Flexible data filtering** by league, season, and training split

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the SSAT package installed with dependencies
pip install panel panel-material-ui

# Or using uv (recommended)
uv add panel panel-material-ui
```

**Note:** Panel Material UI components are automatically available once the library is imported. No special extension configuration is required.

### Running the App

#### Option 1: Using the run script
```bash
cd /path/to/ssat/
python -m ssat.app.run
```

#### Option 2: Direct execution
```bash
cd /path/to/ssat/
python -m ssat.app.main
```

#### Option 3: Custom configuration
```bash
python -m ssat.app.run --port 8080 --no-browser
```

### Testing
```bash
python -m ssat.app.run --test
```

## 📁 Project Structure

```
ssat/app/
├── main.py                 # Main application class and entry point
├── run.py                  # Convenient run script with CLI options
├── test_app.py            # Test suite for the application
├── README.md              # This file
│
├── config/                # Configuration modules
│   ├── ui_config.py       # UI themes, colors, layouts
│   └── app_config.py      # App settings, models, data
│
├── components/            # Reusable UI components
│   ├── sidebar.py         # Sidebar with controls and filters
│   ├── tabs.py            # Main tab interface
│   └── cards.py           # Reusable card components
│
├── pages/                 # Individual page implementations
│   ├── overview.py        # Overview and welcome page
│   ├── models.py          # Model selection and info
│   ├── results.py         # Performance and predictions
│   ├── data.py            # Data exploration page
│   └── docs.py            # Documentation and help
│
└── utils/                 # Utility functions
    └── ui_helpers.py      # UI helper functions and themes
```

## 🎯 Usage Guide

### 1. **Model Selection**
- Choose between **Frequentist** (classical statistics) or **Bayesian** (uncertainty quantification)
- Select **2-4 models** for comparison (more models = more complex visualization)
- View model descriptions and recommendations in the **Models** tab

### 2. **Data Configuration** 
- Select from **7 handball leagues** with real match data
- Filter by **season(s)** (2024-2026 available)
- Adjust **training split** percentage (50-90%)
- Click **"Apply Filters"** to load the filtered dataset

### 3. **Analysis Workflow**
- **Train Models**: Fit selected models to your filtered dataset
- **Generate Predictions**: Create predictions for analysis and comparison
- **View Results**: Analyze performance metrics in the **Results** tab
- **Explore Data**: Use interactive tools in the **Data Explorer** tab

### 4. **Interpretation**
- **Higher accuracy** and **lower MAE** indicate better models
- **Model consensus** suggests more reliable predictions
- Use the **Documentation** tab for detailed guidance

## 🧠 Available Models

### Frequentist Models
- **Bradley-Terry**: Paired comparison with logistic regression
- **GSSD**: Generalized Scores Standard Deviation model
- **Poisson**: Classical goal-scoring model
- **TOOR**: Team Offense-Offense Rating model
- **ZSD**: Zero-Score Distribution model
- **PRP**: Possession-based Rating Process model

### Bayesian Models
- **Poisson**: Bayesian goal-scoring with MCMC
- **NegBinom**: Overdispersed goal modeling
- **Skellam**: Direct goal difference modeling
- **SkellamZero**: Zero-inflated for frequent draws
- **PoissonDecay**: Time-weighted Poisson model
- **SkellamDecay**: Time-weighted Skellam model

## 🔧 Technical Details

### **Architecture**
- **Framework**: Panel + Panel Material UI
- **Backend**: SSAT Statistical Models Library
- **Styling**: Material Design with custom theming
- **State Management**: Parameter-driven reactive architecture
- **Responsive**: CSS Grid and Flexbox with Panel sizing modes
- **Extension Setup**: Panel Material UI components auto-register (no `pn.extension("material")` needed)

### **Data Sources**
- **Real handball data**: 1,449 matches from multiple leagues
- **European competitions**: Championship, Liga ASOBAL, Starligue
- **Nordic leagues**: Danish and Swedish national competitions
- **Time span**: 2024-2026 seasons

### **Performance**
- **Modular loading**: Components loaded on-demand
- **Efficient updates**: Parameter-driven reactive updates
- **Responsive UI**: Adapts to different screen sizes
- **Clean separation**: UI and business logic separated

## 🆚 Improvements Over Original

### **Code Quality**
- ✅ **Modular architecture** - Components in separate files
- ✅ **Clean separation of concerns** - Config, UI, logic separated
- ✅ **Consistent naming** and documentation
- ✅ **Reusable components** for maintainability
- ✅ **Comprehensive tests** for reliability

### **User Experience**
- ✅ **Responsive design** - Works on all devices
- ✅ **Better navigation** - Clear tab structure
- ✅ **Improved feedback** - Status messages and progress
- ✅ **Comprehensive help** - Documentation tab with guidance
- ✅ **Theme consistency** - Unified Material Design

### **Technical Architecture**
- ✅ **Parameter-driven design** - Centralized state management
- ✅ **Reduced complexity** - No circular imports or deep nesting
- ✅ **Easy testing** - Modular components easy to test
- ✅ **Future-ready** - Easy to extend with new features

## 🔮 Future Enhancements

This initial version focuses on **UI/UX improvements** and **clean architecture**. Future phases will add:

- **Phase 2**: Real SSAT model integration and data loading
- **Phase 3**: Advanced visualization with Panel Graphic Walker
- **Phase 4**: Export functionality and reporting
- **Phase 5**: Advanced features (ensemble models, real-time data, etc.)

## 🤝 Contributing

The new architecture makes it easy to contribute:

1. **Add new models**: Update `config/app_config.py`
2. **Add new visualizations**: Create components in `components/`
3. **Add new pages**: Create modules in `pages/`
4. **Modify themes**: Update `config/ui_config.py`
5. **Add utilities**: Extend `utils/ui_helpers.py`

## 📄 License

Part of the SSAT (Statistical Sports Analysis Toolkit) project.