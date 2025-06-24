"""Documentation Page for SSAT Model Comparison App.

This module creates the documentation page with usage instructions,
model descriptions, and help content.
"""

import panel as pn

from ssat.app.components.cards import create_info_card
from ssat.app.config.app_config import APP_INFO
from ssat.app.utils.ui_helpers import get_sizing_mode


def create_docs_page(app) -> pn.Column:
    """Create the documentation page content.

    Args:
        app: The main application instance

    Returns:
        Column containing documentation page components
    """
    # Usage guide section
    usage_guide_card = _create_usage_guide_card()

    # Model reference section
    model_reference_card = _create_model_reference_card()

    # FAQ and troubleshooting
    faq_card = _create_faq_card()

    # Technical details
    technical_card = _create_technical_details_card()

    # Create layout
    page = pn.Column(
        usage_guide_card,
        pn.Row(model_reference_card, faq_card, sizing_mode=get_sizing_mode()),
        technical_card,
        sizing_mode=get_sizing_mode(),
    )

    return page


def _create_usage_guide_card() -> pn.viewable.Viewable:
    """Create the usage guide card."""
    content = """
    <div>
        <h4 style="color: #2E7D32; margin: 0 0 15px 0;">🚀 Complete Usage Guide</h4>

        <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #1976D2;">1. 🧠 Model Selection</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
                <li><strong>Choose Model Type:</strong> Select between Frequentist (classical statistics) or Bayesian (uncertainty quantification)</li>
                <li><strong>Select Models:</strong> Pick 2-4 models for comparison. More models = more complex visualization</li>
                <li><strong>Model Switching:</strong> Change model type to see different approaches to the same problem</li>
            </ul>
        </div>

        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #2E7D32;">2. 🗂️ Data Configuration</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
                <li><strong>League Selection:</strong> Choose from 7 handball leagues with real match data</li>
                <li><strong>Season Filtering:</strong> Select one or multiple seasons (2024-2026 available)</li>
                <li><strong>Training Split:</strong> Adjust the percentage of data used for training (50-90%)</li>
                <li><strong>Apply Filters:</strong> Click to load the filtered dataset for analysis</li>
            </ul>
        </div>

        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #F57C00;">3. ⚡ Model Training & Analysis</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
                <li><strong>Train Models:</strong> Fit selected models to your filtered dataset</li>
                <li><strong>Generate Predictions:</strong> Create predictions for analysis and comparison</li>
                <li><strong>View Results:</strong> Analyze performance metrics in the Results tab</li>
                <li><strong>Export Data:</strong> Save results for external analysis and reporting</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #666;">4. 📊 Interpretation & Next Steps</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
                <li><strong>Performance Metrics:</strong> Higher accuracy and lower MAE indicate better models</li>
                <li><strong>Model Agreement:</strong> Consensus across models suggests reliable predictions</li>
                <li><strong>Data Explorer:</strong> Use the interactive tools to understand your data better</li>
                <li><strong>Iterate:</strong> Try different model combinations and data filters</li>
            </ul>
        </div>
    </div>
    """

    return create_info_card("Usage Guide", content, icon="menu_book")


def _create_model_reference_card() -> pn.viewable.Viewable:
    """Create the model reference card."""
    content = """
    <div>
        <h4 style="color: #1976D2; margin: 0 0 15px 0;">🧠 Model Reference</h4>

        <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #1976D2;">📊 Frequentist Models</h5>
            <div style="font-size: 13px; color: #333;">
                <p style="margin: 0 0 8px 0;"><strong>Bradley-Terry:</strong> Paired comparison with logistic regression</p>
                <p style="margin: 0 0 8px 0;"><strong>GSSD:</strong> Generalized Scores Standard Deviation model</p>
                <p style="margin: 0 0 8px 0;"><strong>Poisson:</strong> Classical goal-scoring model</p>
                <p style="margin: 0 0 8px 0;"><strong>TOOR:</strong> Team Offense-Offense Rating model</p>
                <p style="margin: 0 0 8px 0;"><strong>ZSD:</strong> Zero-Score Distribution model</p>
                <p style="margin: 0;"><strong>PRP:</strong> Possession-based Rating Process model</p>
            </div>
        </div>

        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #2E7D32;">🎯 Bayesian Models</h5>
            <div style="font-size: 13px; color: #333;">
                <p style="margin: 0 0 8px 0;"><strong>Poisson:</strong> Bayesian goal-scoring with MCMC</p>
                <p style="margin: 0 0 8px 0;"><strong>NegBinom:</strong> Overdispersed goal modeling</p>
                <p style="margin: 0 0 8px 0;"><strong>Skellam:</strong> Direct goal difference modeling</p>
                <p style="margin: 0 0 8px 0;"><strong>SkellamZero:</strong> Zero-inflated for frequent draws</p>
                <p style="margin: 0 0 8px 0;"><strong>PoissonDecay:</strong> Time-weighted Poisson model</p>
                <p style="margin: 0;"><strong>SkellamDecay:</strong> Time-weighted Skellam model</p>
            </div>
        </div>

        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #F57C00;">🎯 Model Selection Tips</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 13px; line-height: 1.6;">
                <li>Bradley-Terry: Best for team rankings and head-to-head comparisons</li>
                <li>Poisson variants: Good for goal-based sports with count data</li>
                <li>Skellam models: Ideal for direct score difference prediction</li>
                <li>Bayesian models: Include uncertainty estimates and credible intervals</li>
            </ul>
        </div>
    </div>
    """

    return create_info_card("Model Reference", content, icon="psychology")


def _create_faq_card() -> pn.viewable.Viewable:
    """Create the FAQ and troubleshooting card."""
    content = """
    <div>
        <h4 style="color: #F57C00; margin: 0 0 15px 0;">❓ FAQ & Troubleshooting</h4>

        <div style="margin: 15px 0;">
            <h5 style="margin: 0 0 8px 0; color: #F57C00;">Q: Which models should I choose?</h5>
            <p style="margin: 0 0 15px 0; color: #333; font-size: 13px; line-height: 1.6;">
                Start with Bradley-Terry and GSSD for Frequentist, or Poisson and Skellam for Bayesian.
                These provide good baselines for comparison.
            </p>
        </div>

        <div style="margin: 15px 0;">
            <h5 style="margin: 0 0 8px 0; color: #F57C00;">Q: How much training data do I need?</h5>
            <p style="margin: 0 0 15px 0; color: #333; font-size: 13px; line-height: 1.6;">
                Minimum 50 matches recommended. More data generally improves model performance.
                Use 70-80% for training, 20-30% for testing.
            </p>
        </div>

        <div style="margin: 15px 0;">
            <h5 style="margin: 0 0 8px 0; color: #F57C00;">Q: What do the metrics mean?</h5>
            <p style="margin: 0 0 15px 0; color: #333; font-size: 13px; line-height: 1.6;">
                <strong>Accuracy:</strong> % correct predictions (higher better)<br>
                <strong>MAE:</strong> Average prediction error (lower better)<br>
                <strong>Log-Likelihood:</strong> Model fit quality (less negative better)
            </p>
        </div>

        <div style="margin: 15px 0;">
            <h5 style="margin: 0 0 8px 0; color: #F57C00;">Q: Models won't train - what's wrong?</h5>
            <p style="margin: 0 0 15px 0; color: #333; font-size: 13px; line-height: 1.6;">
                Ensure you've: 1) Applied data filters, 2) Selected 2+ models, 3) Have sufficient data.
                Check the status messages for specific guidance.
            </p>
        </div>

        <div style="background: #ffebee; padding: 12px; border-radius: 6px; margin: 15px 0;">
            <h5 style="margin: 0 0 8px 0; color: #D32F2F;">🔧 Common Issues</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 12px; line-height: 1.5;">
                <li>Insufficient data: Apply filters and ensure dataset loads</li>
                <li>No models selected: Choose at least 2 models for comparison</li>
                <li>Training errors: Check model compatibility with your data</li>
                <li>Slow performance: Reduce number of models or data size</li>
            </ul>
        </div>
    </div>
    """

    return create_info_card("FAQ & Troubleshooting", content, icon="help")


def _create_technical_details_card() -> pn.viewable.Viewable:
    """Create the technical details card."""
    content = f"""
    <div>
        <h4 style="color: #666; margin: 0 0 15px 0;">⚙️ Technical Details</h4>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #666;">🏗️ Application Architecture</h5>
            <div style="font-size: 13px; color: #333; line-height: 1.6;">
                <p style="margin: 0 0 8px 0;"><strong>Framework:</strong> Panel + Panel Material UI</p>
                <p style="margin: 0 0 8px 0;"><strong>Backend:</strong> SSAT Statistical Models Library</p>
                <p style="margin: 0 0 8px 0;"><strong>Data:</strong> Real handball match data (1,449 matches)</p>
                <p style="margin: 0 0 8px 0;"><strong>Models:</strong> 12+ Frequentist and Bayesian implementations</p>
                <p style="margin: 0;"><strong>Version:</strong> {APP_INFO["version"]}</p>
            </div>
        </div>

        <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #1976D2;">📊 Data Sources</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 13px; line-height: 1.6;">
                <li>European Championship handball matches</li>
                <li>Liga ASOBAL (Spain), Starligue (France)</li>
                <li>Danish and Swedish national leagues</li>
                <li>Real historical data from 2024-2026 seasons</li>
            </ul>
        </div>

        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #2E7D32;">🔬 Model Implementation</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 13px; line-height: 1.6;">
                <li>Frequentist models: MLE estimation with sklearn integration</li>
                <li>Bayesian models: MCMC sampling with CmdStanPy</li>
                <li>Cross-validation and holdout testing</li>
                <li>Multiple performance metrics and uncertainty quantification</li>
            </ul>
        </div>

        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5 style="margin: 0 0 12px 0; color: #F57C00;">📚 References & Links</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 13px; line-height: 1.6;">
                <li><strong>SSAT Library:</strong> Statistical Sports Analysis Toolkit</li>
                <li><strong>Panel Framework:</strong> <a href="https://panel.holoviz.org">panel.holoviz.org</a></li>
                <li><strong>Material UI:</strong> Modern component design system</li>
                <li><strong>Sports Analytics:</strong> Academic research and industry applications</li>
            </ul>
        </div>
    </div>
    """

    return create_info_card("Technical Details", content, icon="settings")
