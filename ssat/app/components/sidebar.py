"""Sidebar Components for SSAT Model Comparison App.

This module contains the sidebar UI components including model selectors,
data filters, and action buttons.
"""

from typing import List

import panel as pn
import panel_material_ui as pmui

from ssat.app.config.app_config import AVAILABLE_MODELS
from ssat.app.config.ui_config import LAYOUT_CONFIG
from ssat.app.utils.ui_helpers import get_sizing_mode


def create_sidebar(app) -> List[pn.viewable.Viewable]:
    """Create sidebar components for the application.

    Args:
        app: The main application instance

    Returns:
        List of sidebar components
    """
    components = []

    # Welcome/intro section
    intro_section = _create_intro_section()
    components.append(intro_section)

    # Model configuration card
    model_config_card = _create_model_config_card(app)
    components.append(model_config_card)

    # Data filters card
    data_filters_card = _create_data_filters_card(app)
    components.append(data_filters_card)

    # Action buttons card
    actions_card = _create_actions_card(app)
    components.append(actions_card)

    return components


def _create_intro_section() -> pn.pane.HTML:
    """Create the introduction/welcome section.

    Returns:
        HTML pane with welcome content
    """
    content = """
    <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;">
        <h3 style="color: #2E7D32; margin: 0 0 8px 0; font-size: 18px;">
            🏆 SSAT Dashboard
        </h3>
        <p style="margin: 0; font-size: 13px; line-height: 1.4; color: #555;">
            <strong>Statistical Sports Analysis Toolkit</strong><br>
            Compare ML models for sports predictions
        </p>
    </div>
    """
    return pn.pane.HTML(content, sizing_mode=get_sizing_mode())


def _create_model_config_card(app) -> pmui.Card:
    """Create the model configuration card.

    Args:
        app: The main application instance

    Returns:
        Card with model configuration controls
    """
    # Model type selector
    model_type_select = pmui.Select.from_param(
        app.param.model_type, label="Model Type", sizing_mode=get_sizing_mode()
    )

    # Model selection multi-select
    model_select = _create_model_multiselect(app)

    # Model info display
    model_info = pn.pane.HTML(
        pn.bind(_get_model_info_html, app.param.model_type, app.param.selected_models),
        sizing_mode=get_sizing_mode(),
    )

    return pmui.Card(
        pn.pane.HTML(
            "<h4 style='margin: 0 0 15px 0; color: #2E7D32;'>🧠 Model Configuration</h4>"
        ),
        model_type_select,
        pn.Spacer(height=10),
        model_select,
        pn.Spacer(height=10),
        model_info,
        title="Models",
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def _create_model_multiselect(app) -> pmui.MultiSelect:
    """Create the model multi-select widget.

    Args:
        app: The main application instance

    Returns:
        Multi-select widget for model selection
    """
    model_select = pmui.MultiSelect(
        label="Selected Models",
        value=app.selected_models,
        options=AVAILABLE_MODELS[app.model_type],
        height=LAYOUT_CONFIG["component_height"]["multiselect"],
        sizing_mode=get_sizing_mode(),
    )

    # Link to app parameter
    model_select.link(app, value="selected_models")

    # Update options when model type changes
    def update_options(model_type):
        model_select.options = AVAILABLE_MODELS[model_type]
        # Reset selection to defaults when type changes
        app.selected_models = AVAILABLE_MODELS[model_type][:2]

    pn.bind(update_options, app.param.model_type, watch=True)

    return model_select


def _get_model_info_html(model_type: str, selected_models: List[str]) -> str:
    """Generate HTML info about selected models.

    Args:
        model_type: Current model type
        selected_models: List of selected models

    Returns:
        HTML string with model info
    """
    if not selected_models:
        return "<p style='color: #666; font-size: 12px; margin: 5px 0;'><em>No models selected</em></p>"

    model_count = len(selected_models)
    models_text = ", ".join(selected_models[:3])
    if model_count > 3:
        models_text += f" (+{model_count - 3} more)"

    return f"""
    <div style="background: #f0f8ff; padding: 10px; border-radius: 6px; margin: 5px 0;">
        <p style="margin: 0; font-size: 12px; color: #1976D2;">
            <strong>{model_count} {model_type} models selected:</strong><br>
            {models_text}
        </p>
    </div>
    """


def _create_data_filters_card(app) -> pmui.Card:
    """Create the data filters card.

    Args:
        app: The main application instance

    Returns:
        Card with data filter controls
    """
    # League selector
    league_select = pmui.Select.from_param(
        app.param.league, label="League", sizing_mode=get_sizing_mode()
    )

    # Season multi-select
    season_select = pmui.MultiSelect.from_param(
        app.param.season, label="Season(s)", height=80, sizing_mode=get_sizing_mode()
    )

    # Training split slider
    train_split_slider = pmui.FloatSlider.from_param(
        app.param.train_split, label="Training Split (%)", sizing_mode=get_sizing_mode()
    )

    # Apply filters button
    apply_button = pmui.Button(
        label="Apply Filters",
        variant="outlined",
        color="primary",
        sizing_mode=get_sizing_mode(),
        on_click=app.on_apply_filters,
    )

    # Data status info
    data_status = pn.pane.HTML(
        pn.bind(_get_data_status_html, app.param.data_loaded),
        sizing_mode=get_sizing_mode(),
    )

    return pmui.Card(
        pn.pane.HTML(
            "<h4 style='margin: 0 0 15px 0; color: #2E7D32;'>🗂️ Data Filters</h4>"
        ),
        league_select,
        pn.Spacer(height=10),
        season_select,
        pn.Spacer(height=10),
        train_split_slider,
        pn.Spacer(height=15),
        apply_button,
        pn.Spacer(height=10),
        data_status,
        title="Data",
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def _get_data_status_html(data_loaded: bool) -> str:
    """Generate HTML for data status.

    Args:
        data_loaded: Whether data is loaded

    Returns:
        HTML string with data status
    """
    if data_loaded:
        return """
        <div style="background: #e8f5e8; padding: 8px; border-radius: 4px; margin: 5px 0;">
            <p style="margin: 0; font-size: 11px; color: #2E7D32;">
                ✅ <strong>Data loaded:</strong> 1,449 matches ready
            </p>
        </div>
        """
    else:
        return """
        <div style="background: #fff3e0; padding: 8px; border-radius: 4px; margin: 5px 0;">
            <p style="margin: 0; font-size: 11px; color: #F57C00;">
                ⏳ <strong>Data:</strong> Apply filters to load data
            </p>
        </div>
        """


def _create_actions_card(app) -> pmui.Card:
    """Create the action buttons card.

    Args:
        app: The main application instance

    Returns:
        Card with action buttons
    """
    # Train models button
    train_button = pmui.Button(
        label="Train Models",
        variant="contained",
        color="primary",
        sizing_mode=get_sizing_mode(),
        disabled=pn.bind(_train_button_disabled, app.param.data_loaded, app.param.busy),
        on_click=app.on_train_models,
    )

    # Generate predictions button
    predict_button = pmui.Button(
        label="Generate Predictions",
        variant="contained",
        color="secondary",
        sizing_mode=get_sizing_mode(),
        disabled=pn.bind(
            _predict_button_disabled, app.param.models_trained, app.param.busy
        ),
        on_click=app.on_generate_predictions,
    )

    # Export results button
    export_button = pmui.Button(
        label="Export Results",
        variant="outlined",
        color="primary",
        sizing_mode=get_sizing_mode(),
        disabled=pn.bind(
            _export_button_disabled, app.param.predictions_generated, app.param.busy
        ),
        on_click=app.on_export_results,
    )

    # Progress status
    progress_status = pn.pane.HTML(
        pn.bind(
            _get_progress_status_html,
            app.param.models_trained,
            app.param.predictions_generated,
        ),
        sizing_mode=get_sizing_mode(),
    )

    return pmui.Card(
        pn.pane.HTML("<h4 style='margin: 0 0 15px 0; color: #2E7D32;'>⚡ Actions</h4>"),
        train_button,
        pn.Spacer(height=10),
        predict_button,
        pn.Spacer(height=10),
        export_button,
        pn.Spacer(height=15),
        progress_status,
        title="Actions",
        margin=LAYOUT_CONFIG["card_spacing"],
        sizing_mode=get_sizing_mode(),
    )


def _train_button_disabled(data_loaded: bool, busy: bool) -> bool:
    """Determine if train button should be disabled."""
    return not data_loaded or busy


def _predict_button_disabled(models_trained: bool, busy: bool) -> bool:
    """Determine if predict button should be disabled."""
    return not models_trained or busy


def _export_button_disabled(predictions_generated: bool, busy: bool) -> bool:
    """Determine if export button should be disabled."""
    return not predictions_generated or busy


def _get_progress_status_html(models_trained: bool, predictions_generated: bool) -> str:
    """Generate HTML for progress status.

    Args:
        models_trained: Whether models are trained
        predictions_generated: Whether predictions are generated

    Returns:
        HTML string with progress status
    """
    steps = [
        ("Load Data", True),  # Always available after applying filters
        ("Train Models", models_trained),
        ("Generate Predictions", predictions_generated),
    ]

    html = "<div style='font-size: 11px; color: #666;'>"
    html += "<p style='margin: 0 0 5px 0; font-weight: bold;'>Progress:</p>"

    for step_name, completed in steps:
        icon = "✅" if completed else "⏳"
        color = "#2E7D32" if completed else "#999"
        html += f"<p style='margin: 2px 0; color: {color};'>{icon} {step_name}</p>"

    html += "</div>"
    return html
