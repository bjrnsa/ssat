"""Overview Page for SSAT Model Comparison App.

This module creates the overview page showing status, data summary,
and general application information.
"""

import panel as pn

from ssat.app.components.cards import (
    create_data_summary_card,
    create_info_card,
    create_status_card,
)
from ssat.app.utils.ui_helpers import (
    create_responsive_column,
    create_responsive_row,
    create_welcome_message,
    get_sizing_mode,
)


def create_overview_page(app) -> pn.Column:
    """Create the overview page content.

    Args:
        app: The main application instance

    Returns:
        Column containing overview page components
    """
    # Welcome section
    welcome_html = create_welcome_message()
    welcome_pane = pn.pane.HTML(welcome_html, sizing_mode=get_sizing_mode())

    # Status card
    status_card = create_status_card(app)

    # Data summary card
    data_summary_card = create_data_summary_card(app)

    # Quick start guide
    quick_start_content = """
    <div style="padding: 15px; background: #f0f8ff; border-radius: 8px;">
        <h4 style="color: #1976D2; margin: 0 0 12px 0;">🚀 Quick Start Guide</h4>
        <ol style="margin: 0; padding-left: 20px; color: #333; line-height: 1.8;">
            <li><strong>Select Models:</strong> Choose 2+ models from the sidebar (Frequentist or Bayesian)</li>
            <li><strong>Configure Data:</strong> Set league, season(s), and training split in Data Filters</li>
            <li><strong>Apply Filters:</strong> Click "Apply Filters" to load the dataset</li>
            <li><strong>Train Models:</strong> Click "Train Models" to fit selected models to the data</li>
            <li><strong>Generate Predictions:</strong> Create predictions for analysis and comparison</li>
            <li><strong>Analyze Results:</strong> View performance metrics and predictions in other tabs</li>
        </ol>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 13px;">
            💡 <em>Navigate between tabs to explore different aspects of the model comparison.</em>
        </p>
    </div>
    """
    quick_start_card = create_info_card(
        "Quick Start", quick_start_content, icon="play_circle_outline"
    )

    # Application features info
    features_content = """
    <div style="padding: 15px; background: #f8f9fa; border-radius: 8px;">
        <h4 style="color: #2E7D32; margin: 0 0 12px 0;">✨ Dashboard Features</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; color: #333;">
            <div>
                <h5 style="margin: 0 0 8px 0; color: #2E7D32;">🧠 Models</h5>
                <ul style="margin: 0; padding-left: 15px; font-size: 13px; line-height: 1.6;">
                    <li>6+ Frequentist models</li>
                    <li>6+ Bayesian models</li>
                    <li>Side-by-side comparison</li>
                    <li>Real SSAT integration</li>
                </ul>
            </div>
            <div>
                <h5 style="margin: 0 0 8px 0; color: #1976D2;">📊 Analysis</h5>
                <ul style="margin: 0; padding-left: 15px; font-size: 13px; line-height: 1.6;">
                    <li>Performance metrics</li>
                    <li>Prediction analysis</li>
                    <li>Model agreement</li>
                    <li>Interactive visualization</li>
                </ul>
            </div>
        </div>
    </div>
    """
    features_card = create_info_card("Features", features_content, icon="stars")

    # Create layout with responsive design
    page = create_responsive_column(
        welcome_pane,
        create_responsive_row(
            create_responsive_column(status_card, quick_start_card),
            create_responsive_column(data_summary_card, features_card),
        ),
    )

    return page
