"""UI Helper Functions for SSAT Model Comparison App.

This module contains utility functions for UI operations, styling,
and common interface patterns.
"""

from datetime import datetime
from typing import Dict, List, Optional

import panel as pn

from ssat.app.config.app_config import STATUS_MESSAGES
from ssat.app.config.ui_config import SIZING_MODES, STATUS_STYLES


def create_status_message(
    message_type: str = "default", text: str = "", **kwargs
) -> str:
    """Create a styled status message.

    Args:
        message_type: Type of message ('default', 'success', 'warning', 'error', 'info')
        text: The message text
        **kwargs: Additional format arguments for the text

    Returns:
        HTML-formatted status message
    """
    if text and kwargs:
        text = text.format(**kwargs)

    style = STATUS_STYLES.get(message_type, STATUS_STYLES["default"])

    return f"""
    <div style="
        padding: 12px 16px;
        border-radius: 6px;
        background-color: {style["background"]};
        color: {style["color"]};
        border-left: 4px solid {style["color"]};
        margin: 8px 0;
        font-size: 14px;
        line-height: 1.4;
    ">
        {text}
    </div>
    """


def get_status_message(key: str, **kwargs) -> str:
    """Get a predefined status message.

    Args:
        key: The status message key
        **kwargs: Format arguments for the message

    Returns:
        Formatted status message
    """
    template = STATUS_MESSAGES.get(key, STATUS_MESSAGES["initial"])
    if kwargs:
        template = template.format(**kwargs)
    return template


def create_info_card(
    title: str, content: str, icon: Optional[str] = None, color: str = "primary"
) -> str:
    """Create an informational card with title and content.

    Args:
        title: Card title
        content: Card content (can be HTML)
        icon: Optional Material icon name
        color: Card color theme

    Returns:
        HTML-formatted info card
    """
    icon_html = (
        f'<span class="material-icons" style="margin-right: 8px;">{icon}</span>'
        if icon
        else ""
    )

    return f"""
    <div style="
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        border-left: 4px solid #2E7D32;
    ">
        <h4 style="
            margin: 0 0 8px 0;
            color: #2E7D32;
            font-size: 16px;
            font-weight: 600;
        ">
            {icon_html}{title}
        </h4>
        <div style="
            color: #333;
            font-size: 14px;
            line-height: 1.5;
        ">
            {content}
        </div>
    </div>
    """


def create_welcome_message() -> str:
    """Create the welcome message for the app.

    Returns:
        HTML-formatted welcome message
    """
    return """
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #2E7D32; margin-bottom: 10px; font-size: 24px;">
            🏆 Welcome to SSAT Model Comparison Dashboard
        </h2>
        <p style="color: #555; font-size: 16px; line-height: 1.6; margin: 0;">
            <strong>Statistical Sports Analysis Toolkit</strong><br>
            Compare and analyze machine learning models for sports match outcome predictions
        </p>
    </div>
    """


def create_model_info_section(model_type: str, models: List[str]) -> str:
    """Create an informational section about selected models.

    Args:
        model_type: Type of models ('Frequentist' or 'Bayesian')
        models: List of selected model names

    Returns:
        HTML-formatted model info section
    """
    model_list = ", ".join(models) if models else "None selected"

    return f"""
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h4 style="color: #1976D2; margin: 0 0 8px 0;">
            📊 Current Configuration
        </h4>
        <p style="margin: 5px 0; color: #333;">
            <strong>Model Type:</strong> {model_type}
        </p>
        <p style="margin: 5px 0; color: #333;">
            <strong>Selected Models:</strong> {model_list}
        </p>
    </div>
    """


def create_metrics_explanation() -> str:
    """Create an explanation of performance metrics.

    Returns:
        HTML-formatted metrics explanation
    """
    return """
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
        <h4 style="color: #2E7D32; margin: 0 0 12px 0;">
            📈 Understanding Performance Metrics
        </h4>
        <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
            <li><strong>Accuracy:</strong> Percentage of correct predictions (higher is better, aim for >60%)</li>
            <li><strong>MAE:</strong> Mean Absolute Error for predictions (lower is better, <1.5 is good)</li>
            <li><strong>Log-Likelihood:</strong> Model fit quality (less negative is better)</li>
        </ul>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 13px;">
            💡 <em>Look for models with high accuracy and low MAE for reliable predictions</em>
        </p>
    </div>
    """


def create_prediction_explanation() -> str:
    """Create an explanation of prediction analysis.

    Returns:
        HTML-formatted prediction explanation
    """
    return """
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0;">
        <h4 style="color: #1976D2; margin: 0 0 12px 0;">
            🔮 Understanding Predictions
        </h4>
        <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
            <li><strong>Win Probabilities:</strong> Model confidence in each outcome (0-100%)</li>
            <li><strong>Goal Spreads:</strong> Expected goal difference (positive = home team favored)</li>
            <li><strong>Model Agreement:</strong> When models agree, predictions are more reliable</li>
        </ul>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 13px;">
            💡 <em>High consensus across models suggests more confident predictions</em>
        </p>
    </div>
    """


def get_sizing_mode(component_type: str = "default") -> str:
    """Get the appropriate sizing mode for a component.

    Args:
        component_type: Type of component

    Returns:
        Panel sizing mode string
    """
    return SIZING_MODES.get(component_type, SIZING_MODES["default"])


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format a timestamp for display.

    Args:
        timestamp: Timestamp to format (defaults to now)

    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d_%H-%M-%S")


def create_placeholder_content(title: str, message: str, icon: str = "info") -> str:
    """Create placeholder content for empty states.

    Args:
        title: Placeholder title
        message: Placeholder message
        icon: Material icon name

    Returns:
        HTML-formatted placeholder content
    """
    return f"""
    <div style="
        text-align: center;
        padding: 40px 20px;
        color: #666;
        background: #f9f9f9;
        border-radius: 8px;
        border: 2px dashed #ddd;
    ">
        <span class="material-icons" style="
            font-size: 48px;
            color: #ccc;
            margin-bottom: 16px;
            display: block;
        ">{icon}</span>
        <h3 style="margin: 0 0 8px 0; color: #888;">{title}</h3>
        <p style="margin: 0; font-size: 14px; line-height: 1.5;">{message}</p>
    </div>
    """


def get_theme_colors(dark_theme: bool = False) -> Dict[str, str]:
    """Get color palette for current theme.

    Args:
        dark_theme: Whether to use dark theme colors

    Returns:
        Dictionary of theme colors
    """
    from ssat.app.config.ui_config import THEME_CONFIG

    theme_key = "dark" if dark_theme else "light"
    return THEME_CONFIG[theme_key]


def apply_theme_to_style(base_style: str, dark_theme: bool = False) -> str:
    """Apply theme colors to a CSS style string.

    Args:
        base_style: Base CSS style string
        dark_theme: Whether to use dark theme

    Returns:
        Theme-adjusted CSS style string
    """
    colors = get_theme_colors(dark_theme)

    # Simple theme color replacement (could be extended)
    if dark_theme:
        base_style = base_style.replace("#f9f9f9", "#2d2d2d")
        base_style = base_style.replace("#f8f9fa", "#333333")
        base_style = base_style.replace("#ffffff", "#1e1e1e")
        base_style = base_style.replace("#333", "#ffffff")
        base_style = base_style.replace("#666", "#cccccc")

    return base_style


def create_responsive_row(*components, **kwargs) -> pn.Row:
    """Create a responsive row that adapts to screen size.

    Args:
        *components: Components to include in the row
        **kwargs: Additional Panel Row parameters

    Returns:
        Panel Row with responsive sizing
    """
    default_kwargs = {
        "sizing_mode": "stretch_width",
        "margin": (10, 0),
    }
    default_kwargs.update(kwargs)

    return pn.Row(*components, **default_kwargs)


def create_responsive_column(*components, **kwargs) -> pn.Column:
    """Create a responsive column that adapts to screen size.

    Args:
        *components: Components to include in the column
        **kwargs: Additional Panel Column parameters

    Returns:
        Panel Column with responsive sizing
    """
    default_kwargs = {
        "sizing_mode": "stretch_width",
        "margin": (0, 10),
    }
    default_kwargs.update(kwargs)

    return pn.Column(*components, **default_kwargs)


def get_responsive_width(screen_size: str = "desktop") -> Optional[int]:
    """Get responsive width based on screen size.

    Args:
        screen_size: Screen size category ('mobile', 'tablet', 'desktop')

    Returns:
        Width in pixels or None for stretch
    """
    from ssat.app.config.ui_config import BREAKPOINTS

    if screen_size == "mobile":
        return None  # Full width on mobile
    elif screen_size == "tablet":
        return BREAKPOINTS["md"]
    else:  # desktop
        return None  # Full width with max constraints
