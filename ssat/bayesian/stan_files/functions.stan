functions {
  real skellam_lpmf(int k, real mu1, real mu2) {
    real total;
    real log_prob;
    real sqrt_term;

    sqrt_term = 2 *sqrt(mu1 * mu2);
    total = (-mu1 - mu2) + (log(mu1) - log(mu2)) * k / 2;

    if (sqrt_term < 700) {
      log_prob = total
                 + log(modified_bessel_first_kind(k,
                                                sqrt_term));
    } else {
      log_prob = total
                 + log(modified_bessel_first_kind(k,
                                                700));
    }
    return log_prob;
  }
  // Zero-inflated Skellam implementation
  real zero_inflated_skellam_lpmf(int k, real mu1, real mu2, real zi) {
    if (k == 0) {
      return log_sum_exp(log(zi), log1m(zi) + skellam_lpmf(0 | mu1, mu2));
    } else {
      return log1m(zi) + skellam_lpmf(k | mu1, mu2);
    }
  }

  // Zero-inflated Skellam random number generator
  int zero_inflated_skellam_rng(real mu1, real mu2, real zi) {
    if (bernoulli_rng(zi)) {
      return 0;
    } else {
      return poisson_rng(mu1) - poisson_rng(mu2);
    }
  }

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
