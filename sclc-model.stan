// Revised version of sclc-no-IF.stan. Changes (see sclc-model-review.md):
//
//  * Fixed the convolution integral for new SCLC cases: the gamma shape
//    should be psi/shape + 1, not shape + psi.
//  * phi is positive_ordered (phi[1] < phi[2]), as in the Gaussian-process
//    calibration. (It was unordered from 2026-10 until 2026-11-23, while the
//    priors had mistakenly used mean sojourn times as rates.)
//  * evolve_state works on the three undiagnosed states only, with explicit
//    scalar formulas instead of the 4x4 eigendecomposition. The diagnosed
//    states were never used by the likelihood.
//  * The incidence convolution integrals are computed by fixed-node
//    Gauss-Legendre quadrature of positive integrands, replacing
//    lgamma/gamma_p. State probabilities can never go negative, and the
//    case mu + phi[1] == phi[2] needs no special handling. Follow-up
//    records are evolved in equal steps of at most MAX_STEP years so that
//    20 nodes stay accurate. (An earlier 12-term series version failed
//    when incidence was high, producing NaNs and divergences.)
//  * map_rect replaced by reduce_sum over ragged records, so follow-up
//    records no longer need to be split to give 7 records per patient.
//  * The interval-finding (IF) rates were fixed at 0, so they and the
//    mid_screening flag have been removed.
//  * The patient loop stops after a diagnosis (screen-detected or not).

functions {

  // (exp(-a*t) - exp(-b*t)) / (b - a), smooth and accurate across a == b.
  // Symmetric in a and b; using the smaller rate keeps every factor <= 1.
  real exp_divdiff(real a, real b, real t) {
    real x = abs(a - b) * t;
    // Both branches compute (1 - exp(-x))/x and agree to ~1e-16 at the
    // switch, so this does not introduce a discontinuity
    real h = x > 1e-3
             ? -expm1(-x) / x
             : 1 - x / 2 + x^2 / 6 - x^3 / 24 + x^4 / 120;
    return t * exp(-fmin(a, b) * t) * h;
  }

  // Evolve the undiagnosed state vector (no lung cancer, limited SCLC,
  // extensive SCLC) from t0 to t1. The result is not normalised: the missing
  // mass is the probability of clinical diagnosis during (t0, t1].
  //
  // With a = mu + phi_L (rate of leaving limited) and b = phi_E:
  //   none'    = none * S
  //   limited' = limited * exp(-a tau) + none * G
  //   ext'     = ext * exp(-b tau) + mu * (limited * K(tau) + none * D)
  // where S is Gompertz survival over (t0, t1], f(s) is the Gompertz
  // incidence density at s given no cancer at t0,
  // K(r) = (exp(-a r) - exp(-b r)) / (b - a), and
  //   G = int_{t0}^{t1} f(s) exp(-a (t1 - s)) ds
  //   D = int_{t0}^{t1} f(s) K(t1 - s) ds.
  // Both integrands are positive, so Gauss-Legendre quadrature (nodes gl_u
  // on [0, 1], weights gl_w summing to 1) can never give a negative
  // probability. Keep tau <= MAX_STEP (see transformed data) for accuracy.
  vector evolve_state(
      real        t0,
      real        t1,
      vector      y,
      real        shape,
      real        rate,
      real        mu,
      real        phi_L,
      real        phi_E,
      data vector gl_u,
      data vector gl_w
    ) {

    int  n   = rows(gl_u);
    real a   = mu + phi_L;
    real b   = phi_E;
    real tau = t1 - t0;
    // Gompertz cumulative hazard scale at t0
    real lb  = rate * exp(shape * t0) / shape;

    // Quadrature nodes s in (t0, t1); r = t1 - s is the time since incidence
    vector[n] s = t0 + tau * gl_u;
    vector[n] r = t1 - s;
    // Weighted incidence density: hazard at s times survival from t0 to s
    vector[n] wf = tau * gl_w .* exp(log(rate) + shape * s - lb * expm1(shape * (s - t0)));

    vector[n] K_r;
    for (i in 1:n) {
      K_r[i] = exp_divdiff(a, b, r[i]);
    }

    vector[3] result;
    result[1] = y[1] * exp(-lb * expm1(shape * tau));
    result[2] = y[2] * exp(-a * tau) + y[1] * dot_product(wf, exp(-a * r));
    result[3] = y[3] * exp(-b * tau)
                + mu * (y[2] * exp_divdiff(a, b, tau) + y[1] * dot_product(wf, K_r));
    return result;
  }

  // Weighted log likelihood for patients start:end (for reduce_sum).
  // The state vector holds the probabilities of (no lung cancer,
  // undiagnosed limited SCLC, undiagnosed extensive SCLC), conditional on
  // the patient's history so far.
  real partial_loglik(
      array[] int        pt_slice,
      int                start,
      int                end,
      array[] vector     sens_SCLC,
      array[] vector     sens_SCLC_final,
      vector             b_SCLC_cons,
      real               b_SCLC_plco,
      real               mu,
      vector             phi,
      real               m_SCLC,
      real               c_SCLC,
      data real          beta_former,
      data real          beta_current,
      data vector        wt,
      data vector        plco,
      data array[] int   cigsmok,
      data array[] int   rndgroup,
      data array[] int   pt_first,
      data array[] int   pt_records,
      data array[] int   recordtype,
      data array[] int   event,
      data array[] int   final_screen,
      data array[] real  t_l,
      data array[] real  t_r,
      data array[] int   n_steps,
      data vector        gl_u,
      data vector        gl_w
    ) {

    real total = 0;

    for (pt in start:end) {
      real shape_SCLC = m_SCLC * (beta_former + (beta_current - beta_former) * cigsmok[pt]);
      real rate_SCLC  = exp(c_SCLC + m_SCLC * plco[pt]);
      int  grp        = rndgroup[pt];

      vector[3] state = softmax(append_row(0.0, b_SCLC_plco * plco[pt] + b_SCLC_cons));
      real loglik_p = 0;

      for (j in 1:pt_records[pt]) {
        int r = pt_first[pt] + j - 1;

        if (recordtype[r] == 1) {
          // It's a screen
          vector[2] sens = final_screen[r] == 1 ? sens_SCLC_final[grp] : sens_SCLC[grp];

          if (event[r] == 0) {
            // Negative screen
            state[2:3] = (1 - sens) .* state[2:3];
            real p_neg = sum(state);
            loglik_p += log(p_neg);
            // We have hit a logical impossibility in the model
            if (p_neg <= 0) break;
            state /= p_neg;
          } else {
            // Screen-detected lung cancer
            loglik_p += log(sens[event[r]] * state[1 + event[r]]);
            break;
          }

        } else if (t_r[r] > t_l[r] || event[r] != 0) {
          // It's a follow-up: evolve in n_steps[r] equal steps
          real step = (t_r[r] - t_l[r]) / n_steps[r];
          for (k in 1:n_steps[r]) {
            state = evolve_state(t_l[r] + (k - 1) * step, t_l[r] + k * step,
                                 state, shape_SCLC, rate_SCLC,
                                 mu, phi[1], phi[2], gl_u, gl_w);
          }

          if (event[r] == 0) {
            // Survived cancer diagnosis-free
            real d = sum(state);
            loglik_p += log(d);
            // We have hit a logical impossibility in the model
            if (d <= 0) break;
            state /= d;
          } else {
            // Non-screen-detected lung cancer: density of presentation
            // at t_r from the relevant undiagnosed state
            loglik_p += log(phi[event[r]] * state[1 + event[r]]);
            break;
          }
        }
      }

      total += wt[pt] * loglik_p;
    }

    return total;
  }

  // Posterior predictive category of an SCLC diagnosis:
  //   1 pre-screening, 2 at T0 screen, 3 T0-T1 interval, 4 at T1 screen,
  //   5 T1-T2 interval, 6 at T2 screen, 7 post-screening.
  //
  // A clinical diagnosis at time t is classified from the patient's screens
  // before t (n_before gives how many previous screens, and the label l_last
  // indicates whether it was T0, T1 or T2):
  //
  //   No screen before t
  //     -> pre-screening
  //   More than 1.5 years since the last screen, or the last screen was T2
  //     -> post-screening
  //   Otherwise, by the last screen attended:
  //     T0
  //       -> T0-T1 interval
  //     T1
  //       -> T1-T2 interval
  int clinical_category(real t, int n_before, real s_last, int l_last) {
    if (n_before == 0) return 1;
    if (t - s_last > 1.5 || l_last == 2) return 7;
    if (l_last == 0) return 3;
    return 5;
  }

  // Simulate one patient's SCLC history from the model, with the patient's
  // screens (times, T0/T1/T2 labels, final-screen flags) and end of
  // follow-up as given, and return ( category, stage ) of the diagnosis, or
  // ( 0, 0 ) if SCLC is not diagnosed during follow-up.
  tuple(int, int) simulate_patient_rng(
      real           plco,
      int            cig,
      int            grp,
      int            n_scr,
      array[] real   scr_time,
      array[] int    scr_label,
      array[] int    scr_final,
      real           censor,
      array[] vector sens_SCLC,
      array[] vector sens_SCLC_final,
      vector         b_SCLC_cons,
      real           b_SCLC_plco,
      real           mu,
      vector         phi,
      real           m_SCLC,
      real           c_SCLC,
      real           beta_former,
      real           beta_current
    ) {

    // Baseline state: 0 = no lung cancer, 1 = limited, 2 = extensive
    int state  = categorical_rng(softmax(append_row(0.0, b_SCLC_plco * plco + b_SCLC_cons))) - 1;
    real shape = m_SCLC * (beta_former + (beta_current - beta_former) * cig);
    real rate  = exp(c_SCLC + m_SCLC * plco);

    // Times of entering limited (T_L) and extensive (T_E) SCLC and of
    // clinical presentation (T_clin)
    real T_L = 0;
    if (state == 0) {
      T_L = log1p(shape * exponential_rng(1) / rate) / shape;
    }
    real prog   = exponential_rng(mu);
    real pres_L = exponential_rng(phi[1]);
    real pres_E = exponential_rng(phi[2]);
    real T_E    = positive_infinity();
    if (state == 2) {
      T_E = 0;
    } else if (prog < pres_L) {
      T_E = T_L + prog;
    }
    real T_clin     = is_inf(T_E) ? T_L + pres_L : T_E + pres_E;
    int  clin_stage = is_inf(T_E) ? 1 : 2;

    for (k in 1:n_scr) {
      real s = scr_time[k];
      if (T_clin <= s) {
        return (
          clinical_category(
            T_clin,
            k - 1,
            k > 1 ? scr_time[k - 1] : 0.0,
            k > 1 ? scr_label[k - 1] : 0
          ),
          clin_stage
        );
      }
      int st = s < T_L ? 0 : (s < T_E ? 1 : 2);
      if (st > 0) {
        real se = scr_final[k] == 1 ? sens_SCLC_final[grp,st] : sens_SCLC[grp,st];
        if (bernoulli_rng(se)) {
          return ( 2 + 2 * scr_label[k], st );
        }
      }
    }
    if (T_clin <= censor) {
      return (
        clinical_category(
          T_clin,
          n_scr,
          n_scr > 0 ? scr_time[n_scr] : 0.0,
          n_scr > 0 ? scr_label[n_scr] : 0
        ),
        clin_stage
      );
    }
    return ( 0, 0 );
  }
}

//
// The input data are:
// N                The number of follow-up records
// N_pt             The number of patients
// wt               The weight for each patient
// raw_plco[N_pt]   The PLCOm2012 linear predictor for each patient
// mean_plco        The mean PLCOm2012 linear predictor
// cigsmok[N_pt]    Is patient current smoker? (1 = Yes, 0 = No)
// pt_records[N_pt] How many records are associated with each patient
// pt_first[N_pt]   The index (1:N) for the first record for
//                  each patient
// t_l[N]           Years from randomisation to start of record
// t_r[N]           Years from randomisation to end of record
// recordtype[N]    Type of record (1 = Screen, 2 = Follow-up)
// event[N]         Event at end of record
//                    0 = Negative screen (recordtype = 1) / Survive (recordtype = 2)
//                    1 = Screen-detected / Non-screen-detected Limited SCLC
//                    2 = Screen-detected / Non-screen-detected Extensive SCLC
// grainsize        Grain size for reduce_sum (patients per work unit)
// N_g              No longer used; kept so existing data files still work
//
// The natural model incorporates a multinomial logistic regression for the
// baseline health state (at time of prevalent screen), a Gompertz
// distribution for the time to preclinical limited SCLC, and exponential
// distributions for the times to progression (limited to extensive) and
// clinical presentation (from each stage).
//
data {

  real mean_plco;
  real beta_current;
  real beta_former;

  // NLST data
  int<lower=0>                      N;
  int<lower=0>                      N_pt;
  vector[N_pt]                      wt;
  vector[N_pt]                      raw_plco;
  array[N_pt] int<lower=0,upper=1>  cigsmok;
  array[N_pt] int<lower=1>          pt_records;
  array[N_pt] int<lower=1,upper=N>  pt_first;
  array[N_pt] int<lower=1,upper=2>  rndgroup;
  array[N]    int<lower=1,upper=2>  recordtype;
  array[N]    int<lower=0,upper=2>  event;
  array[N]    int<lower=0,upper=1>  final_screen;
  array[N]    real<lower=0>         t_l;
  array[N]    real<lower=0>         t_r;
  int<lower=1>                      grainsize;
  int<lower=1>                      N_g;

  // Posterior predictive cohort, usually every trial participant (built by
  // sclc_ppc_data.py). Each patient has up to 3 screens: times, labels
  // (0, 1, 2 for T0, T1, T2) and final-screen flags, padded with zeros, and
  // an end of follow-up. N_pp = 0 switches the simulation off.
  int<lower=0>                          N_pp;
  vector[N_pp]                          pp_raw_plco;
  array[N_pp] int<lower=0,upper=1>      pp_cigsmok;
  array[N_pp] int<lower=1,upper=2>      pp_rndgroup;
  array[N_pp] int<lower=0,upper=3>      pp_n_screens;
  array[N_pp, 3] real<lower=0>          pp_screen_time;
  array[N_pp, 3] int<lower=0,upper=2>   pp_screen_label;
  array[N_pp, 3] int<lower=0,upper=1>   pp_screen_final;
  vector<lower=0>[N_pp]                 pp_censor;

}

transformed data {
  vector[N_pt] plco = raw_plco - mean_plco;
  vector[N_pp] pp_plco = pp_raw_plco - mean_plco;
  array[N_pt] int pt_index = linspaced_int_array(N_pt, 1, N_pt);

  // Follow-up records are evolved in equal steps of at most MAX_STEP years,
  // which keeps N_QUAD-node quadrature accurate (relative error below ~1e-10
  // for rates up to ~120/year). Splitting a record does not change the
  // likelihood.
  real MAX_STEP = 0.5;
  int  N_QUAD   = 20;

  array[N] int n_steps;
  for (r in 1:N) {
    n_steps[r] = max(1, to_int(ceil((t_r[r] - t_l[r]) / MAX_STEP)));
  }

  // Gauss-Legendre nodes and weights by the Golub-Welsch algorithm,
  // rescaled to nodes on [0, 1] and weights summing to 1
  vector[N_QUAD] gl_u;
  vector[N_QUAD] gl_w;
  {
    matrix[N_QUAD, N_QUAD] J = rep_matrix(0, N_QUAD, N_QUAD);
    for (k in 1:(N_QUAD - 1)) {
      J[k, k + 1] = k / sqrt(4.0 * k^2 - 1);
      J[k + 1, k] = J[k, k + 1];
    }
    gl_u = (eigenvalues_sym(J) + 1) / 2;
    gl_w = square(eigenvectors_sym(J)[1]');
  }
}

parameters {

  array[2] ordered[2] logit_sens_SCLC;

  // Threshold effect for the final screen
  vector<lower=0>[2] log_or_sens_final;

  real<lower=0> b_SCLC_plco;
  vector<offset=-7, multiplier=2>[2] b_SCLC_cons;

  real<offset=-10.5> c_SCLC;
  real<lower=0> m_SCLC;

  real<lower=0> mu;
  positive_ordered[2] phi;

}

transformed parameters {

  array[2] vector[2] sens_SCLC;
  array[2] vector[2] sens_SCLC_final;

  // We DO NOT include Jacobian adjustments here because we specify priors on
  // on the logit scale
  sens_SCLC[1] = inv_logit(logit_sens_SCLC[1]);
  sens_SCLC[2] = inv_logit(logit_sens_SCLC[2]);

  sens_SCLC_final[1] = inv_logit(logit_sens_SCLC[1] + log_or_sens_final[1]);
  sens_SCLC_final[2] = inv_logit(logit_sens_SCLC[2] + log_or_sens_final[2]);

}

model {

  // Fairly flat prior on sensitivity for limited SCLC, low probability mass
  // for very low or very high sensitivity
  target += normal_lpdf(logit_sens_SCLC[1,1] | 0.0, sqrt(2.0));
  target += normal_lpdf(logit_sens_SCLC[2,1] | 0.0, sqrt(2.0));

  // Sensitivity for extensive SCLC is expected to be very high, so the prior
  // was estimated by least-squares fitting to targets of expected value 0.98
  // and 90% of probability mass above 0.9 using R
  //
  // > logitnorm::twCoefLogitnormE(0.98, 0.9, 0.1)
  //
  // This does not actually achieve expected value 0.98, but graphically it
  // looks acceptable, and the mode is >0.999.
  target += normal_lpdf(logit_sens_SCLC[1,2] | 5.220108, 2.286569);
  target += normal_lpdf(logit_sens_SCLC[2,2] | 5.220108, 2.286569);

  target += gamma_lpdf(log_or_sens_final | 4.0, 4.0/exp(-1.5));

  matrix[2,2] cov_preclin_inc;
  cov_preclin_inc[1,1] = 1;
  cov_preclin_inc[1,2] = -4.0/5.0;
  cov_preclin_inc[2,1] = -4.0/5.0;
  cov_preclin_inc[2,2] = 25.0/16.0;
  target += multi_normal_lpdf([m_SCLC,  c_SCLC] | [2, -10.5], cov_preclin_inc);

  target += gamma_lpdf(b_SCLC_plco     | 10.0, 10.0);
  target += normal_lpdf(b_SCLC_cons[1] | -7, 2);
  target += normal_lpdf(b_SCLC_cons[2] | -7, 2);

  // Gamma priors with shape 4 (coefficient of variation 0.5). Means from
  // ten Haaf et al.'s stage-specific probabilities of progression and mean
  // sojourn times: stages IA-IIIB collapsed into limited SCLC (probability
  // of progressing 0.97^3 * 0.89 * 0.80 = 0.650, mean time 2.473 years),
  // so mu = 0.650 / 2.473 and phi[1] = 0.350 / 2.473; extensive SCLC mean
  // time 0.51 years, so phi[2] = 1 / 0.51. The same means as the log-normal
  // priors of the Gaussian-process calibration.
  target += gamma_lpdf(mu     | 4.0, 4.0/0.262766);
  target += gamma_lpdf(phi[1] | 4.0, 4.0/0.141599);
  target += gamma_lpdf(phi[2] | 4.0, 4.0/1.960784);

  // LIKELIHOOD
  target += reduce_sum(
    partial_loglik, pt_index, grainsize,
    sens_SCLC, sens_SCLC_final, b_SCLC_cons, b_SCLC_plco,
    mu, phi, m_SCLC, c_SCLC,
    beta_former, beta_current, wt, plco, cigsmok, rndgroup,
    pt_first, pt_records, recordtype, event, final_screen, t_l, t_r,
    n_steps, gl_u, gl_w
  );

}

generated quantities {
  // Gap between the two non-zero eigenvalues; values near 0 are where the
  // old eigendecomposition broke down
  real eig_gap = mu + phi[1] - phi[2];

  // Posterior predictive counts of SCLC diagnoses in the cohort pp_*, by
  // arm, stage (1 limited, 2 extensive) and category (1 pre-screening,
  // 2 T0 screen, 3 T0-T1 interval, 4 T1 screen, 5 T1-T2 interval,
  // 6 T2 screen, 7 post-screening; see clinical_category)
  array[2, 2, 7] int pp_count = rep_array(0, 2, 2, 7);
  for (i in 1:N_pp) {
    tuple(int, int) dx = simulate_patient_rng(
      pp_plco[i],
      pp_cigsmok[i],
      pp_rndgroup[i],
      pp_n_screens[i],
      pp_screen_time[i],
      pp_screen_label[i],
      pp_screen_final[i],
      pp_censor[i],
      sens_SCLC,
      sens_SCLC_final,
      b_SCLC_cons,
      b_SCLC_plco,
      mu,
      phi,
      m_SCLC,
      c_SCLC,
      beta_former,
      beta_current
    );
    if (dx.1 > 0) {
      pp_count[pp_rndgroup[i], dx.2, dx.1] += 1;
    }
  }
}
