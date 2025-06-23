#!/usr/bin/env python3
"""
Test data parameter functionality for SSAT Model Comparison App.

This test suite verifies that the app correctly handles data parameter changes
and resets appropriately when new data is provided.
"""

import pytest
import pandas as pd
import numpy as np
from ssat.apps.ssat_model_comparison.app import SSATModelComparisonApp


def create_sample_data(num_matches=100, teams=None):
    """Create sample match data for testing."""
    if teams is None:
        teams = ['Team A', 'Team B', 'Team C', 'Team D']
    
    np.random.seed(42)
    
    data = []
    for _ in range(num_matches):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        # Simulate goals
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        
        data.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'goal_diff': home_goals - away_goals,
            'spread': home_goals - away_goals,
            'league': 'Test League',
            'season': 2024
        })
    
    return pd.DataFrame(data)


def test_app_with_custom_data():
    """Test that app can be initialized with custom data."""
    # Create custom data
    custom_data = create_sample_data(50, ['Alpha', 'Beta', 'Gamma'])
    
    # Initialize app with custom data
    app = SSATModelComparisonApp(data=custom_data)
    
    # Verify the data was set
    assert len(app.data) == 50
    assert set(app.data['home_team'].unique()) <= {'Alpha', 'Beta', 'Gamma'}
    assert app.league == 'Test League'
    assert app.season == 2024


def test_data_parameter_change_triggers_reset():
    """Test that changing data parameter triggers app reset."""
    # Initialize app with default data
    app = SSATModelComparisonApp()
    
    # Train some models to set state
    app.selected_models = ['Bradley-Terry', 'GSSD']
    app.models_trained = True
    app.predictions_generated = True
    app.model_results = {
        'Bradley-Terry': {'accuracy': 0.85, 'mae': 1.2, 'log_likelihood': -120.5},
        'GSSD': {'accuracy': 0.82, 'mae': 1.3, 'log_likelihood': -125.0}
    }
    
    # Create new data and set it
    new_data = create_sample_data(30, ['X Team', 'Y Team'])
    
    # Set new data - this should trigger reset
    app.data = new_data
    
    # Verify reset occurred
    assert app.model_results == {}
    assert len(app.data) == 30


def test_data_change_updates_options():
    """Test that data change updates league and season options."""
    # Initialize app 
    app = SSATModelComparisonApp()
    
    # Create data with multiple leagues and seasons
    data_rows = []
    for league in ['League 1', 'League 2']:
        for season in [2022, 2023, 2024]:
            for _ in range(10):
                data_rows.append({
                    'home_team': 'Team A',
                    'away_team': 'Team B', 
                    'home_goals': 1,
                    'away_goals': 0,
                    'goal_diff': 1,
                    'spread': 1,
                    'league': league,
                    'season': season
                })
    
    new_data = pd.DataFrame(data_rows)
    
    # Set new data
    app.data = new_data
    
    # Check that league options were updated
    assert 'League 1' in app.param.league.objects
    assert 'League 2' in app.param.league.objects
    
    # Check that season objects were updated
    assert 2022 in app.param.season.objects
    assert 2024 in app.param.season.objects
    assert app.season == 2024  # Should default to latest


def test_data_change_handles_missing_columns():
    """Test that app gracefully handles data with missing expected columns."""
    # Initialize app
    app = SSATModelComparisonApp()
    
    # Create minimal data (missing some columns)
    minimal_data = pd.DataFrame({
        'home_team': ['A', 'B'],
        'away_team': ['B', 'A'], 
        'home_goals': [1, 2],
        'away_goals': [0, 1]
    })
    
    # This should not crash - app should handle missing columns gracefully
    app.data = minimal_data
    
    # Verify app still functions
    assert len(app.data) == 2
    assert app.league == 'All'  # Should fall back to default


def test_empty_data_handling():
    """Test that app handles empty data gracefully."""
    # Initialize app
    app = SSATModelComparisonApp()
    
    # Set empty data
    empty_data = pd.DataFrame()
    app.data = empty_data
    
    # Should fall back to defaults
    assert app.param.league.objects == ['All']
    assert app.league == 'All'
    assert 2020 in app.param.season.objects
    assert 2029 in app.param.season.objects


def test_none_data_handling():
    """Test that app handles None data gracefully."""
    # Initialize app
    app = SSATModelComparisonApp()
    
    # This might happen if someone explicitly sets data to None
    app.data = None
    
    # Should fall back to defaults without crashing
    assert app.param.league.objects == ['All']
    assert app.league == 'All'


if __name__ == "__main__":
    pytest.main([__file__])
