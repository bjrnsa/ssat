data {
  int N; // Number of matches
  int T; // Number of teams
  array[N] int<lower=1, upper=T> home_team_idx_match; // Home team index
  array[N] int<lower=1, upper=T> away_team_idx_match; // Away team index
  array[N] int<lower=-1, upper=1> outcome_match; // -1: away win, 0: draw, 1: home win
//   vector[T] starting_elo; // Starting ELO ratings
}
parameters {
  vector[T] rating; // Ratings for each team
  real<lower=0> home_adv; // Home field advantage
  real<lower=0, upper=100> K; // ELO sensitivity constant
  real<lower=0> draw_width; // Parameter controlling draw probability
}
transformed parameters {
  vector[N] p_home_win;
  vector[N] p_draw;
  vector[N] p_away_win;

  for (n in 1:N) {
    real rating_diff = rating[home_team_idx_match[n]] + home_adv - rating[away_team_idx_match[n]];

    // Base probability of home team being better
    real p_better = 1 / (1 + pow(10, -rating_diff / 400));

    // Draw probability peaks when teams are evenly matched and decreases as rating difference increases
    p_draw[n] = exp(-pow(rating_diff / draw_width, 2) / 2);

    // Adjust win/loss probabilities to account for draw possibility
    p_home_win[n] = p_better * (1 - p_draw[n]);
    p_away_win[n] = (1 - p_better) * (1 - p_draw[n]);
  }
}
model {
  // Priors
  rating ~ normal(1200, 50);
  home_adv ~ normal(50, 25);
  K ~ normal(50, 25);
  draw_width ~ normal(100, 25); // Prior for draw width parameter

  // Likelihood using categorical distribution
  for (n in 1:N) {
    if (outcome_match[n] == 1)
      target += log(p_home_win[n]);
    else if (outcome_match[n] == 0)
      target += log(p_draw[n]);
    else
      target += log(p_away_win[n]);
  }
}
generated quantities {
  vector[T] new_rating = rating;
  vector[N] rating_change;

  for (n in 1:N) {
    real expected_score;
    real actual_score;

    // Expected score is weighted sum of possible outcomes
    expected_score = p_home_win[n] + 0.5 * p_draw[n];

    // Actual score based on outcome
    if (outcome_match[n] == 1)
      actual_score = 1.0;
    else if (outcome_match[n] == 0)
      actual_score = 0.5;
    else
      actual_score = 0.0;

    // Calculate the rating change
    rating_change[n] = K * (actual_score - expected_score);

    // Apply the rating changes incrementally
    new_rating[home_team_idx_match[n]] += rating_change[n];
    new_rating[away_team_idx_match[n]] -= rating_change[n];
  }
}
