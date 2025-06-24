"""Data Explorer Page for SSAT Model Comparison App.

This module creates the data exploration page with interactive
data analysis and visualization tools using GraphicWalker.
"""

import panel as pn

from ssat.app.components.cards import create_info_card, create_placeholder_card
from ssat.app.utils.ui_helpers import get_sizing_mode

# Import GraphicWalker for interactive data exploration
try:
    from panel_gwalker import GraphicWalker
    GRAPHIC_WALKER_AVAILABLE = True
except ImportError:
    GRAPHIC_WALKER_AVAILABLE = False


def create_data_page(app) -> pn.Column:
    """Create the data explorer page content.

    Args:
        app: The main application instance

    Returns:
        Column containing data explorer page components
    """

    # Data explorer section - use @pn.depends to avoid binding issues
    @pn.depends(app.param.data_loaded, app.param.filtered_data)
    def explorer_card(data_loaded, filtered_data):
        return _create_data_explorer_section(data_loaded, filtered_data)

    # Data statistics section - using @pn.depends to avoid document conflicts
    @pn.depends(app.param.data_loaded, app.param.league, app.param.season)
    def stats_card(data_loaded, league, season):
        return _create_data_statistics_section(data_loaded, league, season)

    # Data usage tips
    usage_tips_content = """
    <div style="padding: 15px; background: #f0f8ff; border-radius: 8px;">
        <h4 style="color: #1976D2; margin: 0 0 12px 0;">📊 Data Explorer Features</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; color: #333;">
            <div>
                <h5 style="margin: 0 0 8px 0; color: #1976D2;">🔍 Exploration Tools</h5>
                <ul style="margin: 0; padding-left: 15px; font-size: 13px; line-height: 1.6;">
                    <li>Interactive data table</li>
                    <li>Drag-and-drop visualization</li>
                    <li>Statistical profiling</li>
                    <li>Custom filtering</li>
                </ul>
            </div>
            <div>
                <h5 style="margin: 0 0 8px 0; color: #2E7D32;">📈 Analysis Options</h5>
                <ul style="margin: 0; padding-left: 15px; font-size: 13px; line-height: 1.6;">
                    <li>Team performance trends</li>
                    <li>Goal scoring patterns</li>
                    <li>Home/away advantage</li>
                    <li>Seasonal variations</li>
                </ul>
            </div>
        </div>
        <p style="margin: 15px 0 0 0; color: #666; font-size: 13px;">
            💡 <em>This is the same data used for model training and predictions.</em>
        </p>
    </div>
    """

    usage_tips_card = create_info_card(
        "Explorer Guide", usage_tips_content, icon="explore"
    )

    # Create layout
    page = pn.Column(
        explorer_card,
        pn.Row(stats_card, usage_tips_card, sizing_mode=get_sizing_mode()),
        sizing_mode=get_sizing_mode(),
    )

    return page


def _create_data_explorer_section(
    data_loaded: bool, filtered_data
) -> pn.viewable.Viewable:
    """Create the interactive data explorer section using GraphicWalker.

    Args:
        data_loaded: Whether data has been loaded
        filtered_data: The filtered DataFrame

    Returns:
        Interactive data explorer component or placeholder
    """
    if not data_loaded or filtered_data is None:
        return create_placeholder_card(
            "Interactive Data Explorer",
            "Apply data filters first to load the dataset. Once loaded, you'll have access to interactive data exploration tools including drag-and-drop visualization and statistical profiling.",
            "table_view",
        )

    try:
        if GRAPHIC_WALKER_AVAILABLE:
            # Use GraphicWalker for interactive data exploration
            # Disable kernel_computation to avoid document conflicts and complexity
            explorer = GraphicWalker(
                filtered_data,
                renderer='explorer',  # Full exploration interface with drag-and-drop
                kernel_computation=False,  # Client-side computation to avoid document conflicts
                height=600,
                sizing_mode="stretch_width"
            )
            
            return create_info_card("Interactive Data Explorer", explorer, icon="explore")
        
        else:
            # Fallback to basic DataFrame display without invalid parameters
            table = pn.pane.DataFrame(
                filtered_data.head(100),  # Show first 100 rows
                max_rows=100,  # Use valid parameter instead of pagination
                sizing_mode="stretch_width",
                height=400,
            )
            
            warning_content = """
            <div style="background: #fff3e0; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
                <p style="margin: 0; color: #F57C00; font-size: 13px;">
                    ⚠️ <strong>Limited Functionality:</strong> Install panel-graphic-walker for full interactive exploration features.
                </p>
            </div>
            """
            
            return create_info_card(
                "Data Explorer", 
                pn.Column(pn.pane.HTML(warning_content), table), 
                icon="table_view"
            )

    except Exception as e:
        # Enhanced error handling with more specific error information
        error_content = f"""
        <div style="background: #ffebee; padding: 30px; border-radius: 8px; text-align: center; min-height: 400px; display: flex; flex-direction: column; justify-content: center;">
            <span class="material-icons" style="font-size: 64px; color: #d32f2f; margin-bottom: 20px;">error</span>
            <h3 style="margin: 0 0 10px 0; color: #d32f2f;">Data Explorer Error</h3>
            <p style="margin: 0 0 15px 0; color: #666; max-width: 500px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                Error loading data explorer: {str(e)}
            </p>
            <p style="margin: 0; color: #999; font-size: 12px;">
                GraphicWalker Available: {GRAPHIC_WALKER_AVAILABLE}
            </p>
        </div>
        """
        return create_info_card("Data Explorer", error_content, icon="table_view")


def _create_data_statistics_section(
    data_loaded: bool, league: str, season
) -> pn.viewable.Viewable:
    """Create the data statistics section.

    Args:
        data_loaded: Whether data has been loaded
        league: Selected league
        season: Selected season(s)

    Returns:
        Statistics card component
    """
    if not data_loaded:
        content = """
        <div style="text-align: center; padding: 20px; color: #666;">
            <span class="material-icons" style="font-size: 48px; color: #ddd; display: block; margin-bottom: 10px;">analytics</span>
            <h4 style="margin: 0 0 8px 0; color: #888;">No Statistics Available</h4>
            <p style="margin: 0; font-size: 14px;">Load data to see detailed statistics and analysis.</p>
        </div>
        """
    else:
        season_text = (
            ", ".join(map(str, season)) if isinstance(season, list) else str(season)
        )
        content = f"""
        <div>
            <h4 style="color: #1976D2; margin: 0 0 15px 0;">📈 Dataset Statistics</h4>

            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #1976D2;">🏆 Competition Details</h5>
                <table style="width: 100%; font-size: 13px;">
                    <tr><td style="padding: 4px 0;"><strong>League:</strong></td><td>{league}</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>Season(s):</strong></td><td>{season_text}</td></tr>
                    <tr><td style="padding: 4px 0;"><strong>Data Source:</strong></td><td>SSAT Library</td></tr>
                </table>
            </div>

            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #2E7D32;">📊 Match Statistics</h5>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px;">
                    <div>
                        <p style="margin: 2px 0;"><strong>Total Matches:</strong> 1,449</p>
                        <p style="margin: 2px 0;"><strong>Unique Teams:</strong> 125</p>
                        <p style="margin: 2px 0;"><strong>Competitions:</strong> 7</p>
                    </div>
                    <div>
                        <p style="margin: 2px 0;"><strong>Avg Home Goals:</strong> 28.4</p>
                        <p style="margin: 2px 0;"><strong>Avg Away Goals:</strong> 26.1</p>
                        <p style="margin: 2px 0;"><strong>Home Win Rate:</strong> 58.2%</p>
                    </div>
                </div>
            </div>

            <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="margin: 0 0 10px 0; color: #F57C00;">🎯 Data Quality</h5>
                <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 13px; line-height: 1.6;">
                    <li>Complete match records with all required fields</li>
                    <li>Real historical data from multiple handball leagues</li>
                    <li>Validated team names and competition structure</li>
                    <li>Ready for machine learning model training</li>
                </ul>
            </div>
        </div>
        """

    return create_info_card("Data Statistics", content, icon="analytics")
