"""Models Page for SSAT Model Comparison App.

This module creates the models page showing detailed information
about selected models and their configurations.
"""

from typing import List

import panel as pn

from ssat.app.components.cards import create_info_card
from ssat.app.config.app_config import AVAILABLE_MODELS, MODEL_DESCRIPTIONS
from ssat.app.utils.ui_helpers import get_sizing_mode


def create_models_page(app) -> pn.Column:
    """Create the models page content.

    Args:
        app: The main application instance

    Returns:
        Column containing models page components
    """
    # Current configuration card - using @pn.depends to avoid document conflicts
    @pn.depends(app.param.model_type, app.param.selected_models)
    def config_content(model_type, selected_models):
        return pn.pane.HTML(_create_config_content(model_type, selected_models))
    
    config_card = create_info_card(
        "Current Configuration",
        config_content,
        icon="settings",
    )

    # Model descriptions card - using @pn.depends to avoid document conflicts
    @pn.depends(app.param.model_type)
    def descriptions_content(model_type):
        return pn.pane.HTML(_create_model_descriptions_content(model_type))
    
    descriptions_card = create_info_card(
        "Available Models",
        descriptions_content,
        icon="psychology",
    )

    # Model comparison tips
    tips_content = """
    <div style="padding: 15px; background: #fff3e0; border-radius: 8px;">
        <h4 style="color: #F57C00; margin: 0 0 12px 0;">💡 Model Comparison Tips</h4>
        <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.8;">
            <li><strong>Frequentist vs Bayesian:</strong> Frequentist models provide point estimates, Bayesian models include uncertainty</li>
            <li><strong>Model Selection:</strong> Choose 2-4 models for clear comparison without overwhelming visualization</li>
            <li><strong>Data Requirements:</strong> Some models work better with larger datasets or specific sports</li>
            <li><strong>Performance Metrics:</strong> Compare accuracy, MAE, and log-likelihood across models</li>
            <li><strong>Prediction Analysis:</strong> Look for consensus between models for reliable predictions</li>
        </ul>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 13px;">
            📚 <em>Each model type has different strengths - experiment to find the best fit for your data.</em>
        </p>
    </div>
    """
    tips_card = create_info_card(
        "Tips & Best Practices", tips_content, icon="lightbulb"
    )

    # Create layout
    page = pn.Column(
        config_card,
        pn.Row(descriptions_card, tips_card, sizing_mode=get_sizing_mode()),
        sizing_mode=get_sizing_mode(),
    )

    return page


def _create_config_content(model_type: str, selected_models: List[str]) -> str:
    """Create HTML content for current configuration.

    Args:
        model_type: Current model type
        selected_models: List of selected models

    Returns:
        HTML string with configuration details
    """
    if not selected_models:
        return """
        <div style="text-align: center; padding: 20px; color: #666;">
            <span class="material-icons" style="font-size: 48px; color: #ddd; display: block; margin-bottom: 10px;">psychology_alt</span>
            <h4 style="margin: 0 0 8px 0; color: #888;">No Models Selected</h4>
            <p style="margin: 0; font-size: 14px;">Select models from the sidebar to see configuration details.</p>
        </div>
        """

    model_list = ""
    for i, model in enumerate(selected_models, 1):
        description = MODEL_DESCRIPTIONS.get(model_type, {}).get(
            model, "No description available"
        )
        model_list += f"""
        <div style="background: #f9f9f9; padding: 12px; border-radius: 6px; margin: 8px 0; border-left: 4px solid #2E7D32;">
            <h5 style="margin: 0 0 6px 0; color: #2E7D32;">{i}. {model}</h5>
            <p style="margin: 0; font-size: 13px; color: #555; line-height: 1.4;">{description}</p>
        </div>
        """

    return f"""
    <div>
        <div style="background: #e8f5e8; padding: 12px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #2E7D32;">📊 {model_type} Models</h4>
            <p style="margin: 0; font-size: 14px; color: #333;">
                <strong>{len(selected_models)} models selected</strong> for comparison
            </p>
        </div>
        <div>
            <h5 style="margin: 0 0 10px 0; color: #333;">Selected Models:</h5>
            {model_list}
        </div>
    </div>
    """


def _create_model_descriptions_content(model_type: str) -> str:
    """Create HTML content for model descriptions.

    Args:
        model_type: Current model type

    Returns:
        HTML string with model descriptions
    """
    models = AVAILABLE_MODELS.get(model_type, [])
    descriptions = MODEL_DESCRIPTIONS.get(model_type, {})

    if not models:
        return "<p>No models available for this type.</p>"

    content = f"""
    <div>
        <h4 style="margin: 0 0 15px 0; color: #1976D2;">🧠 {model_type} Model Library</h4>
        <p style="margin: 0 0 15px 0; color: #666; font-size: 14px;">
            Choose from {len(models)} available {model_type.lower()} models, each with unique strengths:
        </p>
    """

    for model in models:
        description = descriptions.get(model, "Model description not available")
        content += f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e0e0e0;">
            <h5 style="margin: 0 0 8px 0; color: #1976D2; display: flex; align-items: center;">
                <span class="material-icons" style="margin-right: 8px; font-size: 18px;">psychology</span>
                {model}
            </h5>
            <p style="margin: 0; font-size: 13px; color: #555; line-height: 1.5;">{description}</p>
        </div>
        """

    content += "</div>"
    return content
