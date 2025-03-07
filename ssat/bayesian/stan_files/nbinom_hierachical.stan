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

  if (all_ones(N, weights_match) == 1) {
    home_goals_obs_match ~ neg_binomial_2_log(log_lambda_home_match,
                                              phi_home);
    away_goals_obs_match ~ neg_binomial_2_log(log_lambda_away_match,
                                              phi_away);
  } else {
    for (i in 1 : N) {
      target += weights_match[i]
                * neg_binomial_2_log_lpmf(home_goals_obs_match[i] | log_lambda_home_match[i], phi_home);
      target += weights_match[i]
                * neg_binomial_2_log_lpmf(away_goals_obs_match[i] | log_lambda_away_match[i], phi_away);
    }
  }
}
generated quantities {
  vector[N] ll_home_match;
  vector[N] ll_away_match;
  vector[N] ll_total_match;
  vector[N] pp_home_goals_match;
  vector[N] pp_away_goals_match;
  vector[N] pred_home_goals_match;
  vector[N] pred_away_goals_match;

  for (i in 1 : N) {
    ll_home_match[i] = neg_binomial_2_log_lpmf(home_goals_obs_match[i] | log_lambda_home_match[i], phi_home);
    ll_away_match[i] = neg_binomial_2_log_lpmf(away_goals_obs_match[i] | log_lambda_away_match[i], phi_away);
    ll_total_match[i] = ll_home_match[i] + ll_away_match[i];
    pp_home_goals_match[i] = neg_binomial_2_log_rng(log_lambda_home_match[i],
                                                    phi_home);
    pp_away_goals_match[i] = neg_binomial_2_log_rng(log_lambda_away_match[i],
                                                    phi_away);
  }

  for (i in 1 : N) {
    pred_home_goals_match[i] = neg_binomial_2_log_rng(intercept
                                                      + home_advantage
                                                      + attack_team[home_team_idx_match[i]]
                                                      + defence_team[away_team_idx_match[i]],
                                                      phi_home);
    pred_away_goals_match[i] = neg_binomial_2_log_rng(intercept
                                                      + attack_team[away_team_idx_match[i]]
                                                      + defence_team[home_team_idx_match[i]],
                                                      phi_away);
  }
}
