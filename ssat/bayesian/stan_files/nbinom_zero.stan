functions {
  int all_ones(int N, vector weights_match) {
    int all_ones = 1;
    for (n in 1 : N) {
      if (weights_match[n] != 1) {
        all_ones = 0;
        break;
      }
    }
    return all_ones;
  }

  // Custom PMF for difference of two negative binomials
  real neg_binomial_diff_lpmf(int k, real log_mu1, real log_mu2, real phi1,
                              real phi2) {
    real mu1 = exp(log_mu1);
    real mu2 = exp(log_mu2);
    real log_p = negative_infinity();

    // We need to sum over all possible combinations that result in difference k
    // This is computationally expensive but necessary for exact calculation
    int max_iter = 100; // Practical limit for summation
    int start = max(0, k);

    for (i in start : (start + max_iter)) {
      real log_term = neg_binomial_2_log_lpmf(i | log_mu1, phi1)
                      + neg_binomial_2_log_lpmf(i - k | log_mu2, phi2);
      log_p = log_sum_exp(log_p, log_term);

      // Early stopping if terms become negligible
      if (log_term < log_p - 20)
        break;
    }

    return log_p;
  }

  // RNG for difference of two negative binomials
  int neg_binomial_diff_rng(real log_mu1, real log_mu2, real phi1, real phi2) {
    int x = neg_binomial_2_log_rng(log_mu1, phi1);
    int y = neg_binomial_2_log_rng(log_mu2, phi2);
    return x - y;
  }
}
data {
  int N; // Number of matches
  int T; // Number of teams
  array[N] int<lower=1, upper=T> home_team_idx_match; // Home team index
  array[N] int<lower=1, upper=T> away_team_idx_match; // Away team index
  array[N] int home_goals_obs_match; // Home team goals
  array[N] int away_goals_obs_match; // Away team goals
  vector[N] weights_match;
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

  // Dispersion parameters for negative binomial
  real<lower=0> phi_home;
  real<lower=0> phi_away;
}
transformed parameters {
  vector[T] attack_team;
  vector[T] defence_team;
  vector[N] log_lambda_home_match;
  vector[N] log_lambda_away_match;
  real<lower=0> sigma_attack = inv_sqrt(tau_attack);
  real<lower=0> sigma_defence = inv_sqrt(tau_defence);

  attack_team = attack_raw_team - mean(attack_raw_team);
  defence_team = defence_raw_team - mean(defence_raw_team);

  log_lambda_home_match = intercept + home_advantage
                          + attack_team[home_team_idx_match]
                          + defence_team[away_team_idx_match];
  log_lambda_away_match = intercept + attack_team[away_team_idx_match]
                          + defence_team[home_team_idx_match];
}
model {
  home_advantage ~ normal(0, 1);
  intercept ~ normal(0, 1);

  tau_attack ~ gamma(0.1, 0.1);
  tau_defence ~ gamma(0.1, 0.1);

  // Priors for dispersion parameters
  phi_home ~ gamma(1, 0.1);
  phi_away ~ gamma(1, 0.1);

  attack_raw_team ~ normal(0, sigma_attack);
  defence_raw_team ~ normal(0, sigma_defence);

  // Model for goal difference directly
  if (all_ones(N, weights_match) == 1) {
    for (i in 1 : N) {
      goal_diff_match[i] ~ neg_binomial_diff(log_lambda_home_match[i],
                                             log_lambda_away_match[i],
                                             phi_home, phi_away);
    }
  } else {
    for (i in 1 : N) {
      target += weights_match[i]
                * neg_binomial_diff_lpmf(goal_diff_match[i] | log_lambda_home_match[i], log_lambda_away_match[i], phi_home, phi_away);
    }
  }
}
generated quantities {
  vector[N] ll_diff_match;
  vector[N] pred_goal_diff_match;

  for (i in 1 : N) {
    // Log likelihood for goal difference
    ll_diff_match[i] = neg_binomial_diff_lpmf(goal_diff_match[i] | log_lambda_home_match[i], log_lambda_away_match[i], phi_home, phi_away);

    // Generate predictions
    pred_goal_diff_match[i] = neg_binomial_diff_rng(log_lambda_home_match[i],
                                                    log_lambda_away_match[i],
                                                    phi_home, phi_away);


  }
}
