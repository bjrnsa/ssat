from sklearn.metrics import mean_absolute_error
import pandas as pd
from ssat.frequentist import BradleyTerry, GSSD

# Load sample data
match_df = pd.read_parquet("ssat/data/sample_handball_match_data.parquet")

# Prepare data
X = match_df[["home_team", "away_team"]]
y = match_df["home_goals"] - match_df["away_goals"]  # spread
Z = match_df[["home_goals", "away_goals"]]

# Train-test split
train_size = int(len(match_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
Z_train, Z_test = Z[:train_size], Z[train_size:]

# Fit Model
gssd_model = GSSD()
gssd_model.fit(X_train, y_train, Z_train)

# Detailed team strength analysis
team_stats = gssd_model.get_team_ratings()
print("Team Offensive/Defensive Breakdown:")
print(team_stats[['pfh', 'pah', 'pfa', 'paa']].head())

# Model coefficients
coeffs = team_stats.loc['Coefficients']
print(f"Home offense coefficient: {coeffs['pfh']:.3f}")
print(f"Home defense coefficient: {coeffs['pah']:.3f}")