#include functions.stan
data {
  int N; // Number of matches
  int T; // Number of teams
  array[N] int<lower=1, upper=T> home_team_idx_match; // Home team index
  array[N] int<lower=1, upper=T> away_team_idx_match; // Away team index
  array[N] int home_goals_match; // Home goals
  array[N] int away_goals_match; // Away goals
  vector[N] raw_weights_optional; // (Optional) weights
}
parameters {
  real intercept;
  real home_advantage;
  real<lower=0.001, upper=100> tau;
  vector[T] attack_raw_team;
  vector[T] defence_raw_team;
}
transformed parameters {
  vector[T] attack_team;
  vector[T] defence_team;
  vector[N] lambda_home_match;
  vector[N] lambda_away_match;
  real<lower=0> sigma = inv_sqrt(tau);
  vector[N] weights_match; // Normalized weights

  // Normalize weights to sum to N
  real sum_weights = sum(raw_weights_optional);
  for (i in 1 : N) {
    weights_match[i] = raw_weights_optional[i] * N / sum_weights;
  }

  attack_team = attack_raw_team - mean(attack_raw_team);
  defence_team = defence_raw_team - mean(defence_raw_team);

  lambda_home_match = exp(intercept + home_advantage
                          + attack_team[home_team_idx_match]
                          + defence_team[away_team_idx_match]);
  lambda_away_match = exp(intercept + attack_team[away_team_idx_match]
                          + defence_team[home_team_idx_match]);
}
model {
  home_advantage ~ normal(0, 1);
  intercept ~ normal(2, 1);
  tau ~ gamma(2, 0.5);

  attack_raw_team ~ normal(0, sigma);
  defence_raw_team ~ normal(0, sigma);

  for (i in 1 : N) {
    target += weights_match[i]
              * (poisson_lpmf(home_goals_match[i] | lambda_home_match[i])
                 + poisson_lpmf(away_goals_match[i] | lambda_away_match[i]));
  }
}
generated quantities {
  vector[N] ll_home_match;
  vector[N] ll_away_match;
  vector[N] pred_home_goals_match;
  vector[N] pred_away_goals_match;
  vector[N] pred_goal_diff_match;

  for (i in 1 : N) {
    // Log likelihood
    ll_home_match[i] = poisson_lpmf(home_goals_match[i] | lambda_home_match[i]);
    ll_away_match[i] = poisson_lpmf(away_goals_match[i] | lambda_away_match[i]);

    // Generate predictions
    pred_home_goals_match[i] = poisson_rng(lambda_home_match[i]);
    pred_away_goals_match[i] = poisson_rng(lambda_away_match[i]);
    pred_goal_diff_match[i] = pred_home_goals_match[i]
                              - pred_away_goals_match[i];
  }
}
