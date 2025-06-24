"""Results Page for SSAT Model Comparison App.

This module creates the results page showing performance metrics,
prediction analysis, and model comparison visualizations.
"""

import panel as pn

from ssat.app.components.cards import (
    create_metrics_card,
    create_placeholder_card,
    create_predictions_card,
)
from ssat.app.utils.ui_helpers import get_sizing_mode


def create_results_page(app) -> pn.Column:
    """Create the results page content.

    Args:
        app: The main application instance

    Returns:
        Column containing results page components
    """
    # Performance metrics section
    metrics_card = pn.bind(
        _create_metrics_section,
        app.param.models_trained,
        app.param.model_type,
        app.param.selected_models,
    )

    # Predictions analysis section
    predictions_card = pn.bind(
        _create_predictions_section,
        app.param.predictions_generated,
        app.param.model_type,
        app.param.selected_models,
    )

    # Model comparison insights
    insights_card = pn.bind(
        _create_insights_section,
        app.param.models_trained,
        app.param.predictions_generated,
        app.param.selected_models,
    )

    # Create layout
    page = pn.Column(
        metrics_card, predictions_card, insights_card, sizing_mode=get_sizing_mode()
    )

    return page


def _create_metrics_section(
    models_trained: bool, model_type: str, selected_models
) -> pn.viewable.Viewable:
    """Create the performance metrics section.

    Args:
        models_trained: Whether models have been trained
        model_type: Current model type
        selected_models: List of selected models

    Returns:
        Metrics card component
    """
    if not models_trained:
        return create_placeholder_card(
            "Performance Metrics",
            "Train your selected models to see detailed performance comparison including accuracy, MAE, log-likelihood, and other metrics.",
            "analytics",
        )

    # When models are trained, show metrics visualization placeholder
    return create_metrics_card(
        title="Performance Metrics",
        data={"trained": True},  # Placeholder data
        explanation=True,
    )


def _create_predictions_section(
    predictions_generated: bool, model_type: str, selected_models
) -> pn.viewable.Viewable:
    """Create the predictions analysis section.

    Args:
        predictions_generated: Whether predictions have been generated
        model_type: Current model type
        selected_models: List of selected models

    Returns:
        Predictions card component
    """
    if not predictions_generated:
        return create_placeholder_card(
            "Prediction Analysis",
            "Generate predictions to see win probabilities, goal spreads, and model agreement analysis with interactive visualizations.",
            "insights",
        )

    # When predictions are generated, show analysis visualization placeholder
    return create_predictions_card(
        title="Prediction Analysis",
        data={"generated": True},  # Placeholder data
        explanation=True,
    )


def _create_insights_section(
    models_trained: bool, predictions_generated: bool, selected_models
) -> pn.viewable.Viewable:
    """Create the model insights and comparison section.

    Args:
        models_trained: Whether models have been trained
        predictions_generated: Whether predictions have been generated
        selected_models: List of selected models

    Returns:
        Insights card component
    """
    if not models_trained:
        content = """
        <div style="text-align: center; padding: 30px; color: #666; background: #f9f9f9; border-radius: 8px; border: 2px dashed #ddd;">
            <span class="material-icons" style="font-size: 48px; color: #ccc; margin-bottom: 16px; display: block;">psychology_alt</span>
            <h3 style="margin: 0 0 8px 0; color: #888;">Model Insights</h3>
            <p style="margin: 0; font-size: 14px; line-height: 1.5;">
                Complete model training and prediction generation to see detailed insights and recommendations.
            </p>
        </div>
        """
    elif not predictions_generated:
        content = """
        <div style="padding: 20px; background: #fff3e0; border-radius: 8px; text-align: center;">
            <h4 style="color: #F57C00; margin: 0 0 10px 0;">⏳ Insights Pending</h4>
            <p style="margin: 0; color: #333; font-size: 14px;">
                Generate predictions to unlock comprehensive model insights and comparison analysis.
            </p>
        </div>
        """
    else:
        model_count = len(selected_models) if selected_models else 0
        content = f"""
        <div>
            <h4 style="color: #2E7D32; margin: 0 0 15px 0;">🎯 Model Insights & Recommendations</h4>

            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #2E7D32;">📊 Performance Summary</h5>
                <p style="margin: 0; color: #333; font-size: 14px; line-height: 1.6;">
                    Analyzed {model_count} models across multiple performance dimensions.
                    Detailed metrics comparison available above.
                </p>
            </div>

            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #1976D2;">🔮 Prediction Quality</h5>
                <p style="margin: 0; color: #333; font-size: 14px; line-height: 1.6;">
                    Model agreement analysis and prediction reliability assessment completed.
                    Review prediction visualizations for detailed analysis.
                </p>
            </div>

            <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #F57C00;">💡 Recommendations</h5>
                <ul style="margin: 5px 0 0 0; padding-left: 20px; color: #333; font-size: 14px; line-height: 1.6;">
                    <li>Compare accuracy and MAE metrics to identify top-performing models</li>
                    <li>Look for consensus in predictions across different model types</li>
                    <li>Consider ensemble approaches when models show complementary strengths</li>
                    <li>Export results for detailed analysis and reporting</li>
                </ul>
            </div>
        </div>
        """

    from ssat.app.components.cards import create_info_card

    return create_info_card("Model Insights", content, icon="lightbulb")
