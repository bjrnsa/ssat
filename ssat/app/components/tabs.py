"""Tab Components for SSAT Model Comparison App.

This module creates the main tab interface and manages navigation
between different sections of the application.
"""

import panel_material_ui as pmui


def create_main_tabs(app) -> pmui.Tabs:
    """Create the main tab interface.

    Args:
        app: The main application instance

    Returns:
        Tabs component with all application sections
    """
    # Import page components
    from ssat.app.pages.data import create_data_page
    from ssat.app.pages.docs import create_docs_page
    from ssat.app.pages.models import create_models_page
    from ssat.app.pages.overview import create_overview_page
    from ssat.app.pages.results import create_results_page

    # Create tab pages
    overview_page = create_overview_page(app)
    models_page = create_models_page(app)
    results_page = create_results_page(app)
    data_page = create_data_page(app)
    docs_page = create_docs_page(app)

    # Create tabs with icons and labels from config
    tabs = pmui.Tabs(
        ("Overview", overview_page),
        ("Models", models_page),
        ("Results", results_page),
        ("Data Explorer", data_page),
        ("Documentation", docs_page),
        dynamic=True,
        tabs_location="above",
    )

    return tabs
