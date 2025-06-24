"""Reusable Card Components for SSAT Model Comparison App.

This module contains reusable card components that provide consistent
styling and layout throughout the application.
"""

from typing import List, Optional, Union

import panel as pn
import panel_material_ui as pmui

from ssat.app.config.ui_config import LAYOUT_CONFIG
from ssat.app.utils.ui_helpers import get_sizing_mode


def create_plot_card(
    title: str, 
    plot_component: pn.viewable.Viewable, 
    explanation: str = None,
    dark_theme: bool = False
) -> pmui.Card:
    """Create a card specifically for plot components.
    
    Args:
        title: Card title
        plot_component: Panel plot component (e.g., pn.pane.Matplotlib)
        explanation: Optional explanation text
        dark_theme: Whether to use dark theme styling
        
    Returns:
        Styled card with plot
    """
    components = [plot_component]
    
    if explanation:
        explanation_style = """
        background: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        margin-top: 15px;
        """
        
        if dark_theme:
            explanation_style = """
            background: #2e2e2e; 
            padding: 15px; 
            border-radius: 8px; 
            margin-top: 15px;
            color: #e0e0e0;
            """
        
        explanation_html = f"""
        <div style="{explanation_style}">
            <p style="margin: 0; font-size: 13px; line-height: 1.6;">{explanation}</p>
        </div>
        """
        components.append(pn.pane.HTML(explanation_html))
    
    return pmui.Card(
        *components,
        title=title,
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def create_status_card(app) -> pmui.Card:
    """Create a status display card.

    Args:
        app: The main application instance

    Returns:
        Card with status information
    """
    # Use @pn.depends to avoid document conflicts
    @pn.depends(app.param.status_message)
    def status_pane(status_message):
        return pn.pane.HTML(
            status_message,
            sizing_mode=get_sizing_mode(),
        )

    return pmui.Card(
        status_pane,
        title="Status",
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def create_info_card(
    title: str,
    content: Union[str, pn.viewable.Viewable],
    icon: Optional[str] = None,
    color: str = "primary",
) -> pmui.Card:
    """Create an informational card.

    Args:
        title: Card title
        content: Card content (HTML string or Panel component)
        icon: Optional Material icon name
        color: Card color theme

    Returns:
        Styled information card
    """
    # Create title with optional icon
    if icon:
        title_html = f"""
        <h4 style="margin: 0 0 15px 0; color: #2E7D32; display: flex; align-items: center;">
            <span class="material-icons" style="margin-right: 8px; font-size: 20px;">{icon}</span>
            {title}
        </h4>
        """
    else:
        title_html = f"<h4 style='margin: 0 0 15px 0; color: #2E7D32;'>{title}</h4>"

    # Create content component
    if isinstance(content, str):
        content_component = pn.pane.HTML(content, sizing_mode=get_sizing_mode())
    else:
        content_component = content

    return pmui.Card(
        pn.pane.HTML(title_html),
        content_component,
        title=title,
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def create_placeholder_card(
    title: str,
    message: str,
    icon: str = "info",
    action_text: Optional[str] = None,
    action_callback=None,
) -> pmui.Card:
    """Create a placeholder card for empty states.

    Args:
        title: Card title
        message: Placeholder message
        icon: Material icon name
        action_text: Optional action button text
        action_callback: Optional action button callback

    Returns:
        Placeholder card component
    """
    content = f"""
    <div style="
        text-align: center;
        padding: 30px 20px;
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
        <h3 style="margin: 0 0 8px 0; color: #888; font-size: 18px;">{title}</h3>
        <p style="margin: 0; font-size: 14px; line-height: 1.5;">{message}</p>
    """

    components = [pn.pane.HTML(content)]

    # Add action button if provided
    if action_text and action_callback:
        content += "</div>"
        components[0] = pn.pane.HTML(content)

        action_button = pmui.Button(
            label=action_text,
            variant="outlined",
            color="primary",
            sizing_mode="fixed",
            width=150,
            margin=(10, 0),
            on_click=action_callback,
        )
        components.append(pn.Spacer(height=10))
        components.append(pn.Row(pn.Spacer(), action_button, pn.Spacer()))
    else:
        content += "</div>"
        components[0] = pn.pane.HTML(content)

    return pmui.Card(
        *components,
        title=title,
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def create_metrics_card(
    title: str = "Performance Metrics", data=None, explanation: bool = True
) -> pmui.Card:
    """Create a performance metrics display card.

    Args:
        title: Card title
        data: Metrics data or plot component
        explanation: Whether to include metrics explanation

    Returns:
        Metrics display card
    """
    if data is None or (hasattr(data, "empty") and data.empty):
        # Show placeholder when no data
        content = create_placeholder_card(
            "No Performance Data",
            "Train models to see performance comparison metrics including accuracy, MAE, and log-likelihood.",
            "analytics",
        )

        return pmui.Card(
            content,
            title=title,
            margin=LAYOUT_CONFIG["card_spacing"],
            sizing_mode=get_sizing_mode(),
        )

    # Handle both plot components and other data
    if hasattr(data, '_repr_html_') or hasattr(data, 'object') or isinstance(data, pn.viewable.Viewable):
        # This is a plot or Panel component
        components = [data]
    else:
        # Fallback placeholder
        content_html = """
        <div style="text-align: center; padding: 20px; background: #f0f8ff; border-radius: 8px;">
            <h4 style="color: #1976D2; margin: 0 0 10px 0;">📊 Performance Metrics</h4>
            <p style="margin: 0; color: #333;">Metrics visualization will appear here after model training.</p>
        </div>
        """
        components = [pn.pane.HTML(content_html)]

    if explanation:
        explanation_html = """
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px;">
            <h5 style="color: #2E7D32; margin: 0 0 10px 0;">📈 Understanding Metrics</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6; font-size: 13px;">
                <li><strong>Accuracy:</strong> Percentage of correct predictions (higher is better)</li>
                <li><strong>MAE:</strong> Mean Absolute Error for goal spreads (lower is better)</li>
                <li><strong>Log-Likelihood:</strong> Model fit quality (less negative is better)</li>
                <li><strong>Brier Score:</strong> Probability prediction accuracy (lower is better)</li>
                <li><strong>RPS:</strong> Ranked probability score for ordered outcomes (lower is better)</li>
            </ul>
        </div>
        """
        components.append(pn.pane.HTML(explanation_html))

    return pmui.Card(
        *components,
        title=title,
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def create_predictions_card(
    title: str = "Prediction Analysis", data=None, explanation: bool = True
) -> pmui.Card:
    """Create a predictions display card.

    Args:
        title: Card title
        data: Predictions data or visualization component
        explanation: Whether to include predictions explanation

    Returns:
        Predictions display card
    """
    if data is None or (hasattr(data, "empty") and data.empty):
        # Show placeholder when no data
        content = create_placeholder_card(
            "No Predictions Available",
            "Generate predictions to see win probabilities, goal spreads, and model agreement analysis.",
            "insights",
        )

        return pmui.Card(
            content,
            title=title,
            margin=LAYOUT_CONFIG["card_spacing"],
            sizing_mode=get_sizing_mode(),
        )

    # Handle both visualization components and other data
    if hasattr(data, '_repr_html_') or hasattr(data, 'object') or isinstance(data, pn.viewable.Viewable):
        # This is a visualization component (plot or tabs)
        components = [data]
    else:
        # Fallback placeholder
        content_html = """
        <div style="text-align: center; padding: 20px; background: #f0f8ff; border-radius: 8px;">
            <h4 style="color: #1976D2; margin: 0 0 10px 0;">🔮 Prediction Analysis</h4>
            <p style="margin: 0; color: #333;">Prediction visualizations will appear here after generating predictions.</p>
        </div>
        """
        components = [pn.pane.HTML(content_html)]

    if explanation:
        explanation_html = """
        <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-top: 15px;">
            <h5 style="color: #1976D2; margin: 0 0 10px 0;">🔮 Understanding Predictions</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6; font-size: 13px;">
                <li><strong>Probability Heatmap:</strong> Win probabilities by team matchups (red = home favored)</li>
                <li><strong>Goal Spreads:</strong> Expected goal difference (positive = home favored)</li>
                <li><strong>Model Agreement:</strong> Low variance indicates reliable consensus</li>
                <li><strong>Confidence:</strong> Higher agreement suggests more reliable predictions</li>
            </ul>
        </div>
        """
        components.append(pn.pane.HTML(explanation_html))

    return pmui.Card(
        *components,
        title=title,
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def create_data_summary_card(app) -> pmui.Card:
    """Create a data summary card.

    Args:
        app: The main application instance

    Returns:
        Card with data summary information
    """
    # Use @pn.depends to avoid document conflicts
    @pn.depends(app.param.data_loaded, app.param.league, app.param.season, app.param.filtered_data)
    def summary_content(data_loaded, league, season, filtered_data):
        return pn.pane.HTML(
            _generate_data_summary_html(data_loaded, league, season, filtered_data),
            sizing_mode=get_sizing_mode(),
        )

    return pmui.Card(
        summary_content,
        title="Dataset Overview",
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def _generate_data_summary_html(
    data_loaded: bool, league: str, season: List[int], filtered_data
) -> str:
    """Generate HTML for data summary.

    Args:
        data_loaded: Whether data is loaded
        league: Selected league
        season: Selected seasons
        filtered_data: The filtered DataFrame

    Returns:
        HTML string with data summary
    """
    if not data_loaded or filtered_data is None:
        return """
        <div style="text-align: center; padding: 20px; color: #666;">
            <span class="material-icons" style="font-size: 48px; color: #ddd; display: block; margin-bottom: 10px;">table_view</span>
            <h4 style="margin: 0 0 8px 0; color: #888;">No Data Loaded</h4>
            <p style="margin: 0; font-size: 14px;">Apply data filters to load dataset information.</p>
        </div>
        """

    season_text = (
        ", ".join(map(str, season)) if isinstance(season, list) else str(season)
    )
    
    # Calculate real statistics from filtered data
    try:
        total_matches = len(filtered_data)
        unique_teams = len(set(filtered_data['home_team'].unique()) | set(filtered_data['away_team'].unique()))
        avg_home_goals = filtered_data['home_goals'].mean()
        avg_away_goals = filtered_data['away_goals'].mean()
        home_wins = (filtered_data['home_goals'] > filtered_data['away_goals']).sum()
        home_win_rate = (home_wins / total_matches * 100) if total_matches > 0 else 0
    except Exception:
        # Fallback to placeholder if calculation fails
        total_matches = len(filtered_data) if hasattr(filtered_data, '__len__') else 0
        unique_teams = 0
        avg_home_goals = 0
        avg_away_goals = 0 
        home_win_rate = 0

    return f"""
    <div>
        <h4 style="color: #2E7D32; margin: 0 0 15px 0;">📊 Dataset Summary</h4>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px 0; font-weight: bold;">League:</td>
                <td style="padding: 8px 0;">{league}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px 0; font-weight: bold;">Season(s):</td>
                <td style="padding: 8px 0;">{season_text}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px 0; font-weight: bold;">Total Matches:</td>
                <td style="padding: 8px 0;">{total_matches:,}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px 0; font-weight: bold;">Unique Teams:</td>
                <td style="padding: 8px 0;">{unique_teams}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px 0; font-weight: bold;">Avg Home Goals:</td>
                <td style="padding: 8px 0;">{avg_home_goals:.1f}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 8px 0; font-weight: bold;">Avg Away Goals:</td>
                <td style="padding: 8px 0;">{avg_away_goals:.1f}</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; font-weight: bold;">Home Win Rate:</td>
                <td style="padding: 8px 0; color: #2E7D32;">{home_win_rate:.1f}%</td>
            </tr>
        </table>
    </div>
    """
