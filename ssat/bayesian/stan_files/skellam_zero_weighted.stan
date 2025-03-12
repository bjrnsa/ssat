functions {
  // Numerically stable Skellam implementation
  real skellam_lpmf(int k, real mu1, real mu2) {
    // Prevent extreme parameter values
    real lambda1 = fmin(700, mu1);
    real lambda2 = fmin(700, mu2);

    // Handle edge cases
    if (lambda1 < 1e-10 && lambda2 < 1e-10) {
      return k == 0 ? 0.0 : negative_infinity();
    }

    // For large lambda values, use asymptotic approximation
    if (lambda1 > 100 || lambda2 > 100) {
      // Gaussian approximation for large values
      real mean_diff = lambda1 - lambda2;
      real var_sum = lambda1 + lambda2;
      return normal_lpdf(k | mean_diff, sqrt(var_sum));
    } else {
      // Standard calculation for moderate values
      real log_term = -(lambda1 + lambda2)
                      + (k / 2.0) * log(lambda1 / lambda2);
      log_term += log(modified_bessel_first_kind(k,
                                                 2 * sqrt(lambda1 * lambda2)));
      return log_term;
    }
  }

  // Zero-inflated Skellam implementation
  real zero_inflated_skellam_lpmf(int k, real mu1, real mu2, real gamma) {
    if (k == 0) {
      return log_sum_exp(log(gamma),
                         log1m(gamma) + skellam_lpmf(0 | mu1, mu2));
    } else {
      return log1m(gamma) + skellam_lpmf(k | mu1, mu2);
    }
  }

  // Skellam random number generator
  int skellam_rng(real mu1, real mu2) {
    // For numerical stability, cap the lambda values
    real lambda1 = fmin(700, mu1);
    real lambda2 = fmin(700, mu2);

    // Handle edge cases
    if (lambda1 < 1e-10 && lambda2 < 1e-10) {
      return 0;
    }

    // For large lambda values, use normal approximation
    if (lambda1 > 100 || lambda2 > 100) {
      real mean_diff = lambda1 - lambda2;
      real var_sum = lambda1 + lambda2;
      return int_step(normal_rng(mean_diff, sqrt(var_sum)));
    } else {
      // Standard approach: difference of two Poisson random variables
      return poisson_rng(lambda1) - poisson_rng(lambda2);
    }
  }

  // Zero-inflated Skellam random number generator
  int zero_inflated_skellam_rng(real mu1, real mu2, real gamma) {
    if (bernoulli_rng(gamma)) {
      return 0;
    } else {
      return skellam_rng(mu1, mu2);
    }
  }


}
data {
  int N; // Number of matches
  int T; // Number of teams
  array[N] int<lower=1, upper=T> home_team_idx_match;
  array[N] int<lower=1, upper=T> away_team_idx_match;
  array[N] int home_goals_obs_match;
  array[N] int away_goals_obs_match;
  array[N] int days_since_match; // Days since each match (relative to most recent)
}
transformed data {
  array[N] int goal_diff_match;

  for (i in 1 : N) {
    goal_diff_match[i] = home_goals_obs_match[i] - away_goals_obs_match[i];
  }
}
parameters {
  real home_advantage;
  real intercept;
  real<lower=0.001, upper=100> tau_attack;
  real<lower=0.001, upper=100> tau_defence;
  vector[T] attack_raw_team;
  vector[T] defence_raw_team;
  real<lower=0, upper=1> gamma; // Zero-inflation parameter
  real<lower=0, upper=0.02> xi; // Dixon-Coles time decay parameter
}
transformed parameters {
  vector[T] attack_team;
  vector[T] defence_team;
  vector[N] log_lambda_home_match;
  vector[N] log_lambda_away_match;
  vector[N] lambda_home_match;
  vector[N] lambda_away_match;
  real<lower=0> sigma_attack = inv_sqrt(tau_attack);
  real<lower=0> sigma_defence = inv_sqrt(tau_defence);
  vector[N] weights_match; // Calculate weights inside the model

  // Calculate weights based on xi
  for (i in 1 : N) {
    weights_match[i] = exp(-xi * days_since_match[i]);
  }

  attack_team = attack_raw_team - mean(attack_raw_team);
  defence_team = defence_raw_team - mean(defence_raw_team);

  log_lambda_home_match = intercept + home_advantage
                          + attack_team[home_team_idx_match]
                          + defence_team[away_team_idx_match];
  log_lambda_away_match = intercept + attack_team[away_team_idx_match]
                          + defence_team[home_team_idx_match];

  lambda_home_match = exp(log_lambda_home_match);
  lambda_away_match = exp(log_lambda_away_match);
}
model {
  home_advantage ~ normal(0, 1);
  intercept ~ normal(0, 1);
  gamma ~ beta(1, 9); // Prior for zero-inflation parameter

  tau_attack ~ gamma(0.1, 0.1);
  tau_defence ~ gamma(0.1, 0.1);

  attack_raw_team ~ normal(0, sigma_attack);
  defence_raw_team ~ normal(0, sigma_defence);

  // Papers suggest optimal xi is 0.0018
  xi ~ gamma(2, 0.01);

  // Weighted
  for (i in 1 : N) {
    target += weights_match[i]
              * zero_inflated_skellam_lpmf(goal_diff_match[i] | lambda_home_match[i], lambda_away_match[i], gamma);
  }
}
generated quantities {
  vector[N] ll_skellam_match;
  vector[N] ll_zi_skellam_match;
  vector[N] pred_goal_diff_match;

  for (i in 1 : N) {
    // Log likelihood for standard Skellam (for comparison)
    ll_skellam_match[i] = skellam_lpmf(goal_diff_match[i] | lambda_home_match[i], lambda_away_match[i]);

    // Log likelihood for zero-inflated Skellam
    ll_zi_skellam_match[i] = zero_inflated_skellam_lpmf(goal_diff_match[i] | lambda_home_match[i], lambda_away_match[i], gamma);

    // Generate predictions
    pred_goal_diff_match[i] = zero_inflated_skellam_rng(lambda_home_match[i],
                                                        lambda_away_match[i],
                                                        gamma);
  }
}
