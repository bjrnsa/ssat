import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np

# plt.style.use('default')
# sns.set_palette("husl")

def _set_style(dark_theme: bool):
    """Set the plotting style based on the theme."""
    if dark_theme:
        plt.style.use('dark_background')
        # sns.set_style("darkgrid")
    else:
        plt.style.use('default')
        # sns.set_style("whitegrid")

def create_performance_comparison_plot(model_results, dark_theme: bool=False) -> Figure:
    """Create model performance comparison plot."""
    _set_style(dark_theme)

    if not model_results:
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Train models to see performance comparison', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Model Performance Comparison')
        return fig
    
    fig = Figure(figsize=(15, 6))
    
    # Create subplots for different metrics
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    models = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] for m in models]
    maes = [model_results[m]['mae'] for m in models]
    log_likelihoods = [model_results[m]['log_likelihood'] for m in models]
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MAE comparison
    bars2 = ax2.bar(models, maes, color='lightcoral', alpha=0.7)
    ax2.set_title('Mean Absolute Error')
    ax2.set_ylabel('MAE (goals)')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, mae in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mae:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Log-likelihood comparison
    bars3 = ax3.bar(models, log_likelihoods, color='lightgreen', alpha=0.7)
    ax3.set_title('Log-Likelihood')
    ax3.set_ylabel('Log-Likelihood')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, ll in zip(bars3, log_likelihoods):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{ll:.0f}', ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    return fig

def _create_prediction_comparison_plot(comparison_data, dark_theme: bool=False) -> Figure:
    """Create prediction comparison visualization."""
    _set_style(dark_theme)
    
    if comparison_data is None:
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Generate predictions to see comparison', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Prediction Comparison')
        return fig
    
    fig = Figure(figsize=(16, 8))
    
    # Create heatmap of home win probabilities
    ax1 = fig.add_subplot(221)
    pivot_home = comparison_data.pivot_table(
        values='home_prob', index='match_id', columns='model', aggfunc='mean'
    )
    im1 = ax1.imshow(pivot_home.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('Home Win Probabilities')
    ax1.set_xticks(range(len(pivot_home.columns)))
    ax1.set_xticklabels(pivot_home.columns, rotation=45)
    ax1.set_yticks(range(len(pivot_home.index)))
    ax1.set_yticklabels([idx.split(' vs ')[0][:8] + '...' for idx in pivot_home.index], fontsize=8)
    fig.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Create scatter plot of predicted spreads
    ax2 = fig.add_subplot(222)
    models = comparison_data['model'].unique()
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = comparison_data[comparison_data['model'] == model]
        ax2.scatter(range(len(model_data)), model_data['predicted_spread'], 
                    label=model, alpha=0.7, color=colors[i])
    
    ax2.set_title('Predicted Goal Spreads')
    ax2.set_xlabel('Match Index')
    ax2.set_ylabel('Predicted Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Create probability distribution comparison
    ax3 = fig.add_subplot(223)
    models = comparison_data['model'].unique()
    x_pos = np.arange(len(models))
    width = 0.25
    
    avg_home = [comparison_data[comparison_data['model'] == m]['home_prob'].mean() for m in models]
    avg_draw = [comparison_data[comparison_data['model'] == m]['draw_prob'].mean() for m in models]
    avg_away = [comparison_data[comparison_data['model'] == m]['away_prob'].mean() for m in models]
    
    ax3.bar(x_pos - width, avg_home, width, label='Home Win', alpha=0.8)
    ax3.bar(x_pos, avg_draw, width, label='Draw', alpha=0.8)
    ax3.bar(x_pos + width, avg_away, width, label='Away Win', alpha=0.8)
    
    ax3.set_title('Average Win Probabilities by Model')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Probability')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Create model agreement analysis
    ax4 = fig.add_subplot(224)
    if len(models) >= 2:
        # Calculate correlation between model predictions
        pivot_spread = comparison_data.pivot_table(
            values='predicted_spread', index='match_id', columns='model', aggfunc='mean'
        )
        corr_matrix = pivot_spread.corr()
        
        im4 = ax4.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_title('Model Prediction Correlation')
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45)
        ax4.set_yticks(range(len(corr_matrix.index)))
        ax4.set_yticklabels(corr_matrix.index)
        
        # Add correlation values to heatmap
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        fig.colorbar(im4, ax=ax4, shrink=0.8)
    else:
        ax4.text(0.5, 0.5, 'Need 2+ models for correlation analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model Prediction Correlation')
    
    fig.tight_layout()
    return fig