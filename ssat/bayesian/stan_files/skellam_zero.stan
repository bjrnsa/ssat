#include functions.stan
data {
  int N; // Number of matches
  int T; // Number of teams
  array[N] int<lower=1, upper=T> home_team_idx_match; // Home team index
  array[N] int<lower=1, upper=T> away_team_idx_match; // Away team index
  array[N] int goal_diff_match; // Goal difference
  vector[N] raw_weights_optional; // (Optional) weights
}
parameters {
  real intercept;
  real home_advantage;
  real<lower=0.001, upper=100> tau;
  vector[T] attack_raw_team;
  vector[T] defence_raw_team;
  real<lower=0, upper=1> zi; // Zero-inflation parameter
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

  zi ~ beta(2, 18);

  // Zero-inflated Skellam model for goal differences
  for (i in 1 : N) {
    target += weights_match[i]
              * zero_inflated_skellam_lpmf(goal_diff_match[i] | lambda_home_match[i], lambda_away_match[i], zi);
  }
}
generated quantities {
  vector[N] ll_zi_skellam_match;
  vector[N] pred_goal_diff_match;
  vector[N] pred_lambda_home_match;
  vector[N] pred_lambda_away_match;

  for (i in 1 : N) {
    // Log likelihood for zero-inflated Skellam
    ll_zi_skellam_match[i] = zero_inflated_skellam_lpmf(goal_diff_match[i] | lambda_home_match[i], lambda_away_match[i], zi);

    // Generate predictions
    pred_lambda_home_match[i] = lambda_home_match[i];
    pred_lambda_away_match[i] = lambda_away_match[i];
    pred_goal_diff_match[i] = zero_inflated_skellam_rng(lambda_home_match[i],
                                                        lambda_away_match[i],
                                                        zi);
  }
}
