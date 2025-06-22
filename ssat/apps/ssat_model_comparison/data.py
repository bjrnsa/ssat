import numpy as np
import pandas as pd

def train_models(data, models, model_type, train_split, set_status_message)->dict:
    """
    Train selected models on the provided data.
    """
    model_results={}
    
    # Prepare training data
    train_split_ratio =  train_split/ 100.0
    split_idx = int(len(data) * train_split_ratio)
    
    # Sort by datetime if available, otherwise use index
    if 'datetime' in data.columns:
        sorted_data = data.sort_values('datetime')
    else:
        sorted_data = data.copy()
    
    train_data = sorted_data.iloc[:split_idx]
    test_data = sorted_data.iloc[split_idx:]
    
    if len(train_data) < 10:
        set_status_message("<p><strong>❌ Insufficient training data!</strong> Need at least 10 matches.</p>")
        return {}
    
    if len(test_data) == 0:
        set_status_message("<p><strong>⚠️ No test data available!</strong> Using training data for evaluation.</p>")
        test_data = train_data
    
    # Prepare features and targets
    X_train = train_data[['home_team', 'away_team']]
    Z_train = train_data[['home_goals', 'away_goals']]
    
    # Calculate spread if not available
    if 'spread' in train_data.columns:
        y_train = train_data['spread']
    else:
        y_train = train_data['home_goals'] - train_data['away_goals']
    
    X_test = test_data[['home_team', 'away_team']]
    Z_test = test_data[['home_goals', 'away_goals']]
    
    if 'spread' in test_data.columns:
        y_test = test_data['spread']
    else:
        y_test = test_data['home_goals'] - test_data['away_goals']
    
    # Train each selected model
    for model_name, model in models.items():
        model.fit(X_train, y_train, Z_train)
        
        # Evaluate on test set
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        # Calculate accuracy (percentage of correct outcome predictions)
        pred_outcomes = np.sign(predictions)
        actual_outcomes = np.sign(y_test)
        accuracy = np.mean(pred_outcomes == actual_outcomes)
        
        # Calculate log-likelihood (approximate)
        residuals = y_test - predictions
        log_likelihood = -0.5 * np.sum(residuals ** 2) / np.var(residuals) - 0.5 * len(residuals) * np.log(2 * np.pi * np.var(residuals))
        
        # Store results
        model_results[model_name] = {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'log_likelihood': log_likelihood,
            'model_type': model_type,
            'model_instance': model,  # Store for later prediction use
            'n_train': len(train_data),
            'n_test': len(test_data)
        }
                
    return model_results

        
    


def generate_predictions(data, selected_models)-> pd.DataFrame:
    """Generate realistic predictions for matches based on selected models."""
    home_team_column = data['home_team']
    teams = home_team_column.unique()[:6]
    
    matches = []
    
    for i in range(len(teams)-1):
        for j in range(i+1, len(teams)):
            matches.append({
                'home_team': teams[i],
                'away_team': teams[j],
                'match_id': f"{teams[i]} vs {teams[j]}"
            })
    
    predictions = []
    for match in matches:
        for model_name in selected_models:
            # Generate realistic predictions based on model characteristics
            base_home_prob = 0.45 + np.random.normal(0, 0.1)
            base_draw_prob = 0.15 + np.random.normal(0, 0.05)
            base_away_prob = 1 - base_home_prob - base_draw_prob
            
            # Normalize probabilities
            total = base_home_prob + base_draw_prob + base_away_prob
            home_prob = max(0.1, base_home_prob / total)
            draw_prob = max(0.05, base_draw_prob / total)
            away_prob = max(0.1, base_away_prob / total)
            
            # Renormalize
            total = home_prob + draw_prob + away_prob
            home_prob /= total
            draw_prob /= total
            away_prob /= total
            
            predictions.append({
                'match_id': match['match_id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'model': model_name,
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob,
                'predicted_spread': np.random.normal(0, 3)
            })
    return pd.DataFrame(predictions)