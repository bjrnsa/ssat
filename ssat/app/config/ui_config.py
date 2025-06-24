"""UI Configuration for SSAT Model Comparison App.

This module contains all UI-related configuration including themes,
colors, layouts, and styling options.
"""

# Application Theme Configuration
THEME_CONFIG = {
    "light": {
        "palette": {
            "primary": {"main": "#2E7D32"},  # Green
            "secondary": {"main": "#1976D2"},  # Blue
            "error": {"main": "#D32F2F"},  # Red
            "warning": {"main": "#F57C00"},  # Orange
            "success": {"main": "#388E3C"},  # Green
        },
        "background": {
            "default": "#FFFFFF",
            "paper": "#F5F5F5",
        },
    },
    "dark": {
        "palette": {
            "primary": {"main": "#4CAF50"},  # Light Green
            "secondary": {"main": "#2196F3"},  # Light Blue
            "error": {"main": "#F44336"},  # Light Red
            "warning": {"main": "#FF9800"},  # Light Orange
            "success": {"main": "#4CAF50"},  # Light Green
        },
        "background": {
            "default": "#121212",
            "paper": "#1E1E1E",
        },
    },
}

# Layout Configuration
LAYOUT_CONFIG = {
    "sidebar_width": 350,
    "max_content_width": 1200,
    "card_spacing": 15,
    "component_height": {
        "select": 40,
        "multiselect": 120,
        "slider": 40,
        "button": 45,
    },
}

# Responsive Breakpoints
BREAKPOINTS = {
    "xs": 0,
    "sm": 600,
    "md": 960,
    "lg": 1280,
    "xl": 1920,
}

# Icon Configuration
ICONS = {
    "model_type": "category",
    "models": "psychology",
    "filters": "filter_alt",
    "training": "school",
    "predictions": "analytics",
    "export": "download",
    "overview": "dashboard",
    "results": "insights",
    "data": "table_view",
    "documentation": "help",
    "theme": "dark_mode",
}

# Tab Configuration
TAB_CONFIG = [
    {"label": "Overview", "key": "overview", "icon": ICONS["overview"]},
    {"label": "Models", "key": "models", "icon": ICONS["models"]},
    {"label": "Results", "key": "results", "icon": ICONS["results"]},
    {"label": "Data Explorer", "key": "data", "icon": ICONS["data"]},
    {"label": "Documentation", "key": "docs", "icon": ICONS["documentation"]},
]

# Status Message Styles
STATUS_STYLES = {
    "default": {"color": "#666666", "background": "#F5F5F5"},
    "success": {"color": "#2E7D32", "background": "#E8F5E8"},
    "warning": {"color": "#F57C00", "background": "#FFF3E0"},
    "error": {"color": "#D32F2F", "background": "#FFEBEE"},
    "info": {"color": "#1976D2", "background": "#E3F2FD"},
}

# Component Sizing Modes
SIZING_MODES = {
    "default": "stretch_width",
    "fixed": "fixed",
    "stretch_both": "stretch_both",
    "scale_width": "scale_width",
    "scale_height": "scale_height",
    "scale_both": "scale_both",
}

# Responsive Grid Configuration
RESPONSIVE_CONFIG = {
    "desktop_columns": 2,
    "tablet_columns": 1,
    "mobile_columns": 1,
    "grid_gap": "15px",
    "card_min_width": "300px",
    "card_max_width": "600px",
}
