"""This module contains the ImpliedOdds class for calculating implied probabilities from bookmaker odds."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize


class ImpliedOdds:
    """Class for calculating implied probabilities from bookmaker odds."""

    def __init__(self, methods: Optional[list] = None):
        """Initialize the ImpliedOdds class."""
        if methods is None:
            methods = [
                "multiplicative",
                "additive",
                "power",
                "shin",
                "differential_margin_weighting",
                "odds_ratio",
            ]

        self.methods = {
            "multiplicative": self.multiplicative,
            "additive": self.additive,
            "power": self.power,
            "shin": self.shin,
            "differential_margin_weighting": self.differential_margin_weighting,
            "odds_ratio": self.odds_ratio,
        }

        self.methods = {method: self.methods[method] for method in methods}

    def process_odds(self, odds) -> pd.DataFrame:
        """Process a single set of odds using all available methods."""
        results = {}
        margins = {}
        for method_name, method_func in self.methods.items():
            result = method_func(odds)
            results[method_name] = result["implied_probabilities"]
            margins[f"{method_name}_margin"] = result["margin"]

        # Combine probabilities and margins
        df = pd.DataFrame(results)
        margin_df = pd.DataFrame([margins])
        return pd.concat([df, margin_df], axis=1)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple sets of odds from a DataFrame."""
        if len(df.columns) != 3:
            raise ValueError("DataFrame must contain exactly three columns")

        results = pd.DataFrame(
            [self.process_odds(row.values).iloc[0] for _, row in df.iterrows()]
        )

        return results

    def get_implied_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get implied probabilities for multiple matches."""
        if len(df.columns) != 3:
            raise ValueError("DataFrame must contain exactly three columns")

        outcomes = ["home", "draw", "away"]
        results = []

        for idx, row in df.iterrows():
            odds = row.values
            for method_name, method_func in self.methods.items():
                try:
                    result = method_func(odds)
                    for outcome, prob in zip(outcomes, result["implied_probabilities"]):
                        results.append(
                            {
                                "match_id": idx,
                                "outcome": outcome,
                                "method": method_name,
                                "probability": prob,
                            }
                        )
                except Exception as e:
                    print(f"Error in {method_name} for {idx}: {e}")
                    continue

        result_df = pd.DataFrame(results)
        return result_df.pivot(
            index=["match_id", "outcome"], columns="method", values="probability"
        ).reset_index("outcome")

    def get_margins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get margins for multiple matches."""
        if len(df.columns) != 3:
            raise ValueError("DataFrame must contain exactly three columns")

        results = []
        for idx, row in df.iterrows():
            odds = row.values
            match_margins = {"match_id": idx}
            for method_name, method_func in self.methods.items():
                try:
                    result = method_func(odds)
                    match_margins[method_name] = result["margin"]
                except Exception as e:
                    print(f"Error in {method_name} for {idx}: {e}")
                    continue
            results.append(match_margins)

        return pd.DataFrame(results).set_index("match_id")

    def multiplicative(self, odds) -> dict:
        """Calculate implied probabilities using multiplicative method."""
        odds = np.array(odds)
        inv_odds = 1.0 / odds
        normalized = inv_odds / np.sum(inv_odds)
        margin = np.sum(inv_odds) - 1
        return {
            "implied_probabilities": normalized,
            "method": "multiplicative",
            "margin": margin,
        }

    def additive(self, odds) -> dict:
        """Calculate implied probabilities using additive method."""
        odds = np.array(odds)
        inv_odds = 1.0 / odds
        normalized = inv_odds + 1 / len(inv_odds) * (1 - np.sum(inv_odds))
        margin = np.sum(inv_odds) - 1
        return {
            "implied_probabilities": normalized,
            "method": "additive",
            "margin": margin,
        }

    def power(self, odds) -> dict:
        """Calculate implied probabilities using power method."""
        odds = np.array(odds)
        inv_odds = 1.0 / odds
        margin = np.sum(inv_odds) - 1

        def _power(k, inv_odds):
            implied = inv_odds**k
            return implied

        def _power_error(k, inv_odds):
            implied = _power(k, inv_odds)
            return 1 - np.sum(implied)

        res = optimize.ridder(_power_error, 0, 100, args=(inv_odds,))
        normalized = _power(res, inv_odds)
        return {
            "implied_probabilities": normalized,
            "method": "power",
            "k": res,
            "margin": margin,
        }

    def shin(self, odds) -> dict:
        """Calculate implied probabilities using Shin method."""
        odds = np.array(odds)
        inv_odds = 1.0 / odds
        margin = np.sum(inv_odds) - 1

        def _shin_error(z, inv_odds):
            implied = _shin(z, inv_odds)
            return 1 - np.sum(implied)

        def _shin(z, inv_odds):
            implied = (
                (z**2 + 4 * (1 - z) * inv_odds**2 / np.sum(inv_odds)) ** 0.5 - z
            ) / (2 - 2 * z)
            return implied

        res = optimize.ridder(_shin_error, 0, 100, args=(inv_odds,))
        normalized = _shin(res, inv_odds)
        return {
            "implied_probabilities": normalized,
            "method": "shin",
            "z": res,
            "margin": margin,
        }

    def differential_margin_weighting(self, odds) -> dict:
        """Calculate implied probabilities using differential margin weighting method."""
        odds = np.array(odds)
        inv_odds = 1.0 / odds
        margin = np.sum(inv_odds) - 1
        n_odds = len(odds)
        fair_odds = (n_odds * odds) / (n_odds - (margin * odds))
        return {
            "implied_probabilities": 1 / fair_odds,
            "method": "differential_margin_weighting",
            "margin": margin,
        }

    def odds_ratio(self, odds) -> dict:
        """Calculate implied probabilities using odds ratio method."""
        odds = np.array(odds)
        inv_odds = 1.0 / odds
        margin = np.sum(inv_odds) - 1

        def _or_error(c, inv_odds):
            implied = _or(c, inv_odds)
            return 1 - np.sum(implied)

        def _or(c, inv_odds):
            y = inv_odds / (c + inv_odds - (c * inv_odds))
            return y

        res = optimize.ridder(_or_error, 0, 100, args=(inv_odds,))
        normalized = _or(res, inv_odds)
        return {
            "implied_probabilities": normalized,
            "method": "odds_ratio",
            "c": res,
            "margin": margin,
        }
