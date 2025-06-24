"""Plotting Components for SSAT Model Comparison App.

This module contains plotting functions adapted for the new app with
Material UI compatibility, theme support, and responsive sizing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _set_style(dark_theme: bool = False):
    """Set the plotting style based on the theme.
    
    Args:
        dark_theme: Whether to use dark theme styling
    """
    if dark_theme:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")


def create_performance_plot(
    model_metrics: pd.DataFrame, dark_theme: bool = False
) -> Figure:
    """Create model performance comparison plot.
    
    Args:
        model_metrics: DataFrame with model performance metrics
        dark_theme: Whether to use dark theme styling
        
    Returns:
        matplotlib Figure object
    """
    _set_style(dark_theme)
    
    if model_metrics is None or model_metrics.empty:
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Train models to see performance comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Model Performance Comparison")
        if dark_theme:
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#1e1e1e')
        return fig

    # Define metric configurations
    metric_configs = {
        'accuracy': {
            'title': 'Accuracy', 
            'ylabel': 'Accuracy', 
            'color': '#2196F3', 
            'format': '{:.3f}', 
            'ylim': (0, 1)
        },
        'mae': {
            'title': 'Mean Absolute Error', 
            'ylabel': 'MAE (goals)', 
            'color': '#F44336', 
            'format': '{:.2f}', 
            'ylim': None
        },
        'mse': {
            'title': 'Mean Squared Error', 
            'ylabel': 'MSE', 
            'color': '#FF9800', 
            'format': '{:.2f}', 
            'ylim': None
        },
        'log_likelihood': {
            'title': 'Log-Likelihood', 
            'ylabel': 'Log-Likelihood', 
            'color': '#4CAF50', 
            'format': '{:.0f}', 
            'ylim': None
        },
        'brier': {
            'title': 'Brier Score', 
            'ylabel': 'Brier Score', 
            'color': '#FF5722', 
            'format': '{:.3f}', 
            'ylim': None
        },
        'log_loss': {
            'title': 'Log Loss', 
            'ylabel': 'Log Loss', 
            'color': '#9C27B0', 
            'format': '{:.3f}', 
            'ylim': None
        },
        'rps': {
            'title': 'Ranked Probability Score', 
            'ylabel': 'RPS', 
            'color': '#607D8B', 
            'format': '{:.3f}', 
            'ylim': None
        },
        'calibration': {
            'title': 'Calibration Error', 
            'ylabel': 'Calibration Error', 
            'color': '#E91E63', 
            'format': '{:.3f}', 
            'ylim': None
        },
        'ignorance': {
            'title': 'Ignorance Score', 
            'ylabel': 'Ignorance', 
            'color': '#FFC107', 
            'format': '{:.3f}', 
            'ylim': None
        }
    }

    # Get available metrics from data
    available_metrics = [
        col for col in model_metrics.columns 
        if col in metric_configs and col != 'model'
    ]
    
    if not available_metrics:
        # No valid metrics found
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "No valid metrics found in training results",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=14
        )
        ax.set_title("Model Performance Comparison")
        return fig
    
    models = model_metrics["model"].unique().tolist()
    
    # Calculate optimal subplot arrangement
    n_metrics = len(available_metrics)
    if n_metrics <= 3:
        rows, cols = 1, n_metrics
        figsize = (4 * cols, 4)
    elif n_metrics <= 6:
        rows, cols = 2, 3
        figsize = (12, 8)
    else:
        rows, cols = 3, 3
        figsize = (12, 10)

    fig = Figure(figsize=figsize)
    
    # Set background color for dark theme
    if dark_theme:
        fig.patch.set_facecolor('#1e1e1e')
    
    # Create subplots for each metric
    for i, metric in enumerate(available_metrics):
        ax = fig.add_subplot(rows, cols, i + 1)
        config = metric_configs[metric]
        
        if dark_theme:
            ax.set_facecolor('#1e1e1e')
        
        values = model_metrics[metric].tolist()
        bars = ax.bar(models, values, color=config['color'], alpha=0.7)
        
        ax.set_title(config['title'])
        ax.set_ylabel(config['ylabel'])
        if config['ylim']:
            ax.set_ylim(config['ylim'])
        
        # Rotate x-axis labels if more than 3 models
        if len(models) > 3:
            ax.tick_params(axis="x", rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                label_text = config['format'].format(value)
                y_offset = max(bar.get_height() * 0.02, abs(bar.get_height()) * 0.01)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    
    # Adjust layout to prevent overlap
    fig.tight_layout(pad=2.0)
    
    return fig


def create_prediction_heatmap(
    predictions_data: pd.DataFrame, dark_theme: bool = False
) -> Figure:
    """Create prediction probability heatmap.
    
    Args:
        predictions_data: DataFrame with prediction probabilities
        dark_theme: Whether to use dark theme styling
        
    Returns:
        matplotlib Figure object
    """
    _set_style(dark_theme)
    
    if predictions_data is None or predictions_data.empty:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "Generate predictions to see probability heatmap",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=14
        )
        ax.set_title("Prediction Probability Heatmap")
        if dark_theme:
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#1e1e1e')
        return fig
    
    fig = Figure(figsize=(12, 8))
    if dark_theme:
        fig.patch.set_facecolor('#1e1e1e')
    
    # Create heatmap for home win probabilities
    ax = fig.add_subplot(111)
    if dark_theme:
        ax.set_facecolor('#1e1e1e')
    
    # Pivot data for heatmap
    if 'home_prob' in predictions_data.columns:
        heatmap_data = predictions_data.pivot_table(
            index='home_team', 
            columns='away_team', 
            values='home_prob', 
            aggfunc='mean'
        )
        
        im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(heatmap_data.index)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Home Win Probability')
        
        ax.set_title('Home Win Probability Heatmap')
        ax.set_xlabel('Away Team')
        ax.set_ylabel('Home Team')
    else:
        ax.text(0.5, 0.5, "No probability data available", 
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title('Prediction Heatmap')
    
    fig.tight_layout()
    return fig


def create_model_agreement_plot(
    predictions_data: pd.DataFrame, dark_theme: bool = False
) -> Figure:
    """Create model agreement analysis plot.
    
    Args:
        predictions_data: DataFrame with predictions from multiple models
        dark_theme: Whether to use dark theme styling
        
    Returns:
        matplotlib Figure object
    """
    _set_style(dark_theme)
    
    if predictions_data is None or predictions_data.empty:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "Generate predictions to see model agreement analysis",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=14
        )
        ax.set_title("Model Agreement Analysis")
        if dark_theme:
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#1e1e1e')
        return fig
    
    fig = Figure(figsize=(12, 6))
    if dark_theme:
        fig.patch.set_facecolor('#1e1e1e')
    
    # Create agreement analysis
    ax = fig.add_subplot(111)
    if dark_theme:
        ax.set_facecolor('#1e1e1e')
    
    if 'model' in predictions_data.columns and 'home_prob' in predictions_data.columns:
        # Calculate standard deviation of predictions across models for each match
        agreement_data = predictions_data.groupby('match_id').agg({
            'home_prob': ['mean', 'std'],
            'draw_prob': ['mean', 'std'],
            'away_prob': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        agreement_data.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in agreement_data.columns.values]
        
        # Plot agreement (low std = high agreement)
        if 'home_prob_std' in agreement_data.columns:
            x = range(len(agreement_data))
            ax.bar(x, agreement_data['home_prob_std'], alpha=0.7, color='#2196F3')
            ax.set_xlabel('Match Index')
            ax.set_ylabel('Prediction Standard Deviation')
            ax.set_title('Model Agreement (Lower = More Agreement)')
        else:
            ax.text(0.5, 0.5, "Insufficient data for agreement analysis",
                   ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No model prediction data available",
               ha="center", va="center", transform=ax.transAxes)
        ax.set_title('Model Agreement Analysis')
    
    fig.tight_layout()
    return fig