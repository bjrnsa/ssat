# %% [markdown]
"""# Handball Match Analysis using Multiple Statistical Models.

This notebook analyzes handball match data using various statistical models to predict match outcomes
and analyze model performance. We'll use several statistical approaches including:

- Bradley-Terry model for paired comparisons
- TOOR (Team Offense-Offense Rating)
- GSSD (Goal Scoring Statistical Distribution)
- ZSD (Zero-Score Distribution)
- PRP (Possession-based Rating Process)
- Poisson model
"""

# %% [markdown]
"""
## Setup and Imports

First, let's import the required packages and set up our visualization preferences.
"""
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ssat.frequentist import GSSD, PRP, TOOR, ZSD, BradleyTerry, Poisson

# Configure plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# %% [markdown]
"""
## Data Loading and Preparation

Load the handball match data and extract relevant features and target variables.
"""

# %%
# Load the data
print("Loading handball match data...")
df = pd.read_pickle("ssat/data/sample_handball_data.pkl")

# Extract features and target variables
X = df[["home_team", "away_team"]]
Z = df[["home_goals", "away_goals"]]
y = df["spread"]

# Display basic information about the dataset
print("\nDataset Overview:")
print(df.info())
print("\nSample of the data:")
print(df.head())

# %% [markdown]
"""
## Model Configuration

Set up prediction parameters and initialize our model containers.
"""

# %%
# Set prediction parameters
outcome = "draw"
point_spread = 0
include_draw = True

# Initialize DataFrames for predictions
preds_df = pd.DataFrame(index=X.index)
preds_proba_df = pd.DataFrame(index=X.index)

# Define models
models = [BradleyTerry(), TOOR(), GSSD(), ZSD(), PRP(), Poisson()]

# %% [markdown]
"""
## Model Fitting and Prediction

Fit each model and generate predictions.
"""

# %%
# Fit models and generate predictions
for model in models:
    model.fit(X, y, Z)
    preds_df[model.NAME.lower()] = model.predict(X)
    preds_proba_df[model.NAME.lower()] = model.predict_proba(
        X, Z, point_spread, include_draw, outcome
    )

# %% [markdown]
"""
## Model Performance Analysis

### 1. Prediction Distributions
"""

# %%
# Model Prediction Distribution
plt.figure(figsize=(12, 6))
preds_df.boxplot()
plt.title("Distribution of Model Predictions")
plt.xticks(rotation=45)
plt.ylabel("Predicted Spread")
plt.tight_layout()
# plt.savefig("plots/prediction_distribution.png")
plt.show()

# %% [markdown]
"""
### 2. Probability Distributions
"""

# %%
# Probability Distribution
plt.figure(figsize=(12, 6))
preds_proba_df.boxplot()
plt.title("Distribution of Predicted Probabilities")
plt.xticks(rotation=45)
plt.ylabel("Predicted Probability")
plt.tight_layout()
# plt.savefig("plots/probability_distribution.png")
plt.show()

# %% [markdown]
"""
### 3. Model Correlations
"""

# %%
# Model Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(preds_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation between Model Predictions")
plt.tight_layout()
# plt.savefig("plots/model_correlation.png")
plt.show()

# %% [markdown]
"""
## Statistical Analysis

Calculate and analyze prediction and probability statistics.
"""

# %%
# Calculate prediction statistics
stats_dict = {
    "Mean": preds_df.mean(),
    "Std": preds_df.std(),
    "Min": preds_df.min(),
    "Max": preds_df.max(),
}
prediction_stats = pd.DataFrame(stats_dict).round(3)

# Calculate probability statistics
proba_stats_dict = {
    "Mean Probability": preds_proba_df.mean(),
    "Std Probability": preds_proba_df.std(),
    "Min Probability": preds_proba_df.min(),
    "Max Probability": preds_proba_df.max(),
}
probability_stats = pd.DataFrame(proba_stats_dict).round(3)

print("Prediction Statistics:")
print(prediction_stats)
print("\nProbability Statistics:")
print(probability_stats)

# %%
