// NSCLC screening model with 7 undiagnosed stages and a per-patient random
// effect on progression. See nsclc-recommendations.md.
//
// Undiagnosed stage k (k = 1..7) progresses to stage k+1 at rate
// z * lambda[k] (no progression from stage 7) and presents clinically at rate
// xi[k]. z is a mean-one log-normal random effect, sampled non-centred. New
// cases enter stage 1 with a Gompertz hazard rate * exp(shape * t).
//
// State evolution over a follow-up step uses product integration with
// uniformization:
//  * the Gompertz factor exp(shape * s) is moved into the matrix exponential,
//    and only the smooth survival factor is interpolated, by a degree-DEG
//    polynomial at equally spaced points, so the stiff exponential decay is
//    handled exactly whatever the progression and presentation rates;
//  * exp(A h) and the integrals of each interpolation basis polynomial times
//    the matrix exponential come from a single uniformization sequence, which
//    stays accurate and non-negative when rates are equal or nearly equal;
//  * these depend on the patient and the step length but not on when the
//    step happens, so they are computed once per patient for each distinct
//    step length (found from the data in transformed data). Each step then
//    costs a 7x7 matrix-vector product and DEG + 1 survival evaluations;
//  * steps are split into equal sub-steps sharing the same setup, so that
//    both the uniformization argument and the Gompertz cumulative hazard per
//    sub-step stay small. Parameter values needing more than N_SUB_MAX
//    sub-steps (absurd Gompertz hazards) are rejected.
// Records are split into equal steps of at most MAX_STEP years.
//
// Weighted patients (wt != 1, e.g. non-cancer patients kept by thinning) have
// their random effect integrated out numerically (log_marginal_lik), and the
// weight multiplies the log of that marginal likelihood. Unweighted patients
// have a sampled non-centred random effect eta, as before. Weighting the
// likelihood given a sampled eta would treat the wt patients a weighted
// patient stands for as sharing one random effect, which lets the sampler
// give weighted patients extreme random effects and distorts the posterior.
// The integration stays accurate for large sigma_re, where Gauss-Hermite
// quadrature fails (see log_marginal_lik).
// eta[i] belongs to the i-th unweighted patient in data order.
//
// CHOICES TO CHECK (marked "ASSUMPTION" below): the priors, the sensitivity
// structure (increasing with stage on the logit scale, per arm), the
// final-screen threshold effect, and the random-effect parameterisation.

functions {

  // Setup for steps of length h for one patient. Each step is split into
  // n_sub equal sub-steps of length hs = h / n_sub, chosen so that
  //  * x = q * hs <= x_max, so the Poisson weights never underflow, and
  //  * the Gompertz cumulative hazard over any sub-step up to the patient's
  //    last follow-up time t_end is at most h_max, so the survival factor is
  //    smooth enough for degree-DEG interpolation (error ~5e-12 at 0.25).
  // Sub-steps of the same length share this setup, so splitting is cheap.
  // If more than n_sub_max sub-steps would be needed (only at absurd
  // parameter values), the setup is skipped and the caller rejects the
  // proposal. With B = A - shape * I, returns
  //   E   = exp(A hs)
  //   Psi = matrix whose row j + 1 is hs * int_0^1 L_j(rho) exp(B hs rho) e1 drho
  //   hs and n_sub
  // where L_j are the Lagrange basis polynomials on DEG + 1 equally spaced
  // points (monomial coefficients in c_lagrange[m + 1, j + 1]).
  //
  // Uniformization of B, with q = max(psi) + shape and P = I + B / q (lower
  // bidiagonal and non-negative), gives
  //   exp(B hs) = sum_n Poisson(n; x) P^n,  exp(A hs) = exp(shape hs) exp(B hs)
  //   hs int_0^1 rho^m exp(B hs rho) e1 drho
  //     = hs sum_n (n+1)...(n+m) / x^(m+1) * Pr[Poisson(x) >= n+m+1] P^n e1
  // with all weights non-negative. The series stops once the Poisson tail is
  // below tol (bound: tail <= w_n * x / (n + 1 - x) for n + 1 > x). The number
  // of terms and sub-steps depends on parameters, but changing them moves the
  // result by at most ~tol (sub-stepping is exact), so this adds no
  // meaningful discontinuity.
  tuple(matrix, matrix, real, real) step_setup(
      real        h,
      vector      psi,
      vector      prog,
      real        shape,
      real        rate,
      data real   t_end,
      data matrix c_lagrange,
      data real   x_max,
      data real   h_max,
      data int    n_sub_max,
      data real   tol,
      data int    max_terms
    ) {

    int  K     = rows(psi);
    int  n_pts = rows(c_lagrange);
    int  deg   = n_pts - 1;
    real q     = max(psi) + shape;

    // Bound on the cumulative hazard over a sub-step: at most
    // rate * exp(shape * t_end) * expm1(shape * hs) / shape <= H_bound / n_sub
    real H_bound = rate * exp(shape * t_end) * expm1(shape * h) / shape;
    real n_sub   = fmax(fmax(1, ceil(q * h / x_max)), ceil(H_bound / h_max));
    if (is_inf(n_sub) || n_sub > n_sub_max) {
      return (rep_matrix(0, K, K), rep_matrix(0, n_pts, K), 0, n_sub);
    }
    real hs = h / n_sub;
    real x  = q * hs;

    vector[K]     p_diag = 1 - (psi + shape) / q;
    vector[K - 1] p_sub  = prog / q;

    // Number of terms: first n with n + 1 > x and tail bound below tol
    int  n_max = 0;
    {
      real w = exp(-x);
      while ((n_max + 1 <= x || w * x / (n_max + 1 - x) >= tol) && n_max < max_terms) {
        n_max += 1;
        w *= x / n_max;
      }
    }

    // Poisson probabilities pois[n + 1] up to n_max + deg + 1, and upper
    // tails tail[k + 1] = Pr[Poisson(x) >= k], summed from the top so that
    // small tails keep full relative accuracy
    int n_all = n_max + deg + 1;
    vector[n_all + 1] pois;
    pois[1] = exp(-x);
    for (n in 1:n_all) {
      pois[n + 1] = pois[n] * x / n;
    }
    vector[n_all + 2] tail;
    tail[n_all + 2] = 0;
    for (i in 1:(n_all + 1)) {
      int k = n_all + 2 - i;
      tail[k] = pois[k] + tail[k + 1];
    }

    matrix[K, K]     Pn = diag_matrix(rep_vector(1, K));
    matrix[K, K]     E  = pois[1] * Pn;
    matrix[n_pts, K] M  = rep_matrix(0, n_pts, K);
    for (n in 0:n_max) {
      if (n > 0) {
        // Bidiagonal P times lower-triangular P^(n-1)
        Pn = diag_pre_multiply(p_diag, Pn)
             + append_row(rep_row_vector(0, K), diag_pre_multiply(p_sub, Pn[1:(K - 1)]));
        E += pois[n + 1] * Pn;
      }
      for (m in 0:deg) {
        real rising = 1;
        for (i in 1:m) {
          rising *= n + i;
        }
        M[m + 1] += hs * rising / x^(m + 1) * tail[n + m + 2] * Pn[:, 1]';
      }
    }

    return (exp(shape * hs) * E, c_lagrange' * M, hs, n_sub);
  }

  // Evolve (no lung cancer, undiagnosed stages 1..K) over one step of length
  // n_sub * hs starting at t0, using a setup from step_setup. Returns the new
  // 'no lung cancer' probability and stage vector, not normalised: the
  // missing mass is the probability of clinical diagnosis during the step.
  //
  // Over each sub-step (a, b], with r = b - s,
  //   none(b) = none(a) * S(b)
  //   y(b)    = E y(a) + none(a) * rate * exp(shape b)
  //                      * int_0^hs S(b - r) exp(B r) e1 dr
  // where S(s) = exp(-lb * expm1(shape (s - a))) is the Gompertz survival
  // from a. Interpolating S at s_j = b - rho_j hs makes the integral
  // sum_j S(s_j) Psi[j + 1].
  tuple(real, vector) evolve_step(
      real        t0,
      real        none,
      vector      y,
      real        shape,
      real        rate,
      matrix      E,
      matrix      Psi,
      real        hs,
      real        n_sub,
      data vector rho
    ) {

    real      none_cur = none;
    vector[rows(y)] y_cur = y;
    int k = 0;
    while (k < n_sub) {
      real a = t0 + k * hs;
      k += 1;
      real lb = rate * exp(shape * a) / shape;
      vector[rows(rho)] S = exp(-lb * expm1(shape * hs * (1 - rho)));
      y_cur = E * y_cur + none_cur * rate * exp(shape * (a + hs)) * (Psi' * S);
      none_cur *= exp(-lb * expm1(shape * hs));
    }
    return (none_cur, y_cur);
  }

  // Log likelihood of patient pt's records given their random effect z.
  // Returns negative infinity at absurd parameter values (see step_setup).
  real patient_loglik(
      int                pt,
      real               z,
      array[] vector     sens,
      array[] vector     sens_final,
      vector             b_cons,
      real               b_plco,
      real               m_NSCLC,
      real               c_NSCLC,
      vector             lambda,
      vector             xi,
      data real          beta_former,
      data real          beta_current,
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
      data array[] int   step_len_idx,
      data array[] int   pt_n_len,
      data array[,] real pt_len,
      data array[] real  pt_t_end,
      data vector        rho,
      data matrix        c_lagrange,
      data real          x_max,
      data real          h_max,
      data int           n_sub_max,
      data real          tol,
      data int           max_terms
    ) {

    real shape = m_NSCLC * (beta_former + (beta_current - beta_former) * cigsmok[pt]);
    real rate  = exp(c_NSCLC + m_NSCLC * plco[pt]);
    int  grp   = rndgroup[pt];

    vector[6] prog = z * lambda;
    vector[7] psi  = append_row(prog, 0) + xi;

    // Setups for each of this patient's distinct step lengths
    int n_len = pt_n_len[pt];
    int n_pts = rows(c_lagrange);
    array[n_len] matrix[7, 7]     E;
    array[n_len] matrix[n_pts, 7] Psi;
    array[n_len] real             hs;
    array[n_len] real             n_sub;
    for (l in 1:n_len) {
      tuple(matrix[7, 7], matrix[n_pts, 7], real, real) su = step_setup(
        pt_len[pt, l], psi, prog, shape, rate, pt_t_end[pt], c_lagrange,
        x_max, h_max, n_sub_max, tol, max_terms);
      // Gompertz hazard too large to integrate (absurd parameter values):
      // reject
      if (is_inf(su.4) || su.4 > n_sub_max) {
        return negative_infinity();
      }
      E[l]     = su.1;
      Psi[l]   = su.2;
      hs[l]    = su.3;
      n_sub[l] = su.4;
    }

    // Baseline state at the prevalence screen
    vector[8] p0 = softmax(append_row(0.0, b_plco * plco[pt] + b_cons));
    real      none = p0[1];
    vector[7] y    = p0[2:8];
    real loglik_p = 0;

    for (j in 1:pt_records[pt]) {
      int r = pt_first[pt] + j - 1;

      if (recordtype[r] == 1) {
        // It's a screen
        vector[7] se = final_screen[r] == 1 ? sens_final[grp] : sens[grp];

        if (event[r] == 0) {
          // Negative screen
          y = (1 - se) .* y;
          real p_neg = none + sum(y);
          loglik_p += log(p_neg);
          // We have hit a logical impossibility in the model
          if (p_neg <= 0) break;
          none /= p_neg;
          y    /= p_neg;
        } else {
          // Screen-detected lung cancer at stage event[r]
          loglik_p += log(se[event[r]] * y[event[r]]);
          break;
        }

      } else if (t_r[r] > t_l[r] || event[r] != 0) {
        // It's a follow-up: evolve in n_steps[r] equal steps (none for a
        // zero-length record ending in a diagnosis)
        int l = step_len_idx[r];
        if (l > 0) {
          for (k in 1:n_steps[r]) {
            tuple(real, vector[7]) st = evolve_step(
              t_l[r] + (k - 1) * pt_len[pt, l], none, y, shape, rate,
              E[l], Psi[l], hs[l], n_sub[l], rho);
            none = st.1;
            y    = st.2;
          }
        }

        if (event[r] == 0) {
          // Survived diagnosis-free
          real d = none + sum(y);
          loglik_p += log(d);
          // We have hit a logical impossibility in the model
          if (d <= 0) break;
          none /= d;
          y    /= d;
        } else {
          // Clinical presentation at t_r from stage event[r]
          loglik_p += log(xi[event[r]] * y[event[r]]);
          break;
        }
      }
    }

    return loglik_p;
  }

  // Log likelihood for the unweighted patients re_pts[start:end] (for
  // reduce_sum), conditional on their sampled random effects. The
  // non-centred random-effect variables are the sliced argument, so each
  // slice only receives its own patients' values.
  real partial_loglik_re(
      array[] real       eta_slice,
      int                start,
      int                end,
      data array[] int   re_pts,
      array[] vector     sens,
      array[] vector     sens_final,
      vector             b_cons,
      real               b_plco,
      real               m_NSCLC,
      real               c_NSCLC,
      vector             lambda,
      vector             xi,
      real               sigma_re,
      data real          beta_former,
      data real          beta_current,
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
      data array[] int   step_len_idx,
      data array[] int   pt_n_len,
      data array[,] real pt_len,
      data array[] real  pt_t_end,
      data vector        rho,
      data matrix        c_lagrange,
      data real          x_max,
      data real          h_max,
      data int           n_sub_max,
      data real          tol,
      data int           max_terms
    ) {

    real total = 0;
    for (i in start:end) {
      // ASSUMPTION: mean-one log-normal random effect on all progression rates
      real z = exp(sigma_re * eta_slice[i - start + 1] - 0.5 * square(sigma_re));
      total += patient_loglik(
        re_pts[i], z, sens, sens_final, b_cons, b_plco, m_NSCLC, c_NSCLC,
        lambda, xi, beta_former, beta_current, plco, cigsmok, rndgroup,
        pt_first, pt_records, recordtype, event, final_screen, t_l, t_r,
        n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange,
        x_max, h_max, n_sub_max, tol, max_terms);
    }
    return total;
  }

  // The quadratic through (x[k], y[k]), k = 1..3, evaluated at x0
  real quad_interp(vector x, vector y, real x0) {
    return y[1] * (x0 - x[2]) * (x0 - x[3]) / ((x[1] - x[2]) * (x[1] - x[3]))
         + y[2] * (x0 - x[1]) * (x0 - x[3]) / ((x[2] - x[1]) * (x[2] - x[3]))
         + y[3] * (x0 - x[1]) * (x0 - x[2]) / ((x[3] - x[1]) * (x[3] - x[2]));
  }

  // Log of the marginal likelihood of patient pt, E[L(z)] over eta ~ N(0, 1)
  // with z = exp(sigma_re * eta - sigma_re^2 / 2), by the trapezoid rule in
  // eta with tail extrapolation in u = log z.
  //
  // Nodes are eta_j = j h for |eta_j| <= t_eta, with h = min(h_eta_max,
  // kappa / sigma_re), so the node spacing in u is at most kappa however
  // large sigma_re is. The trapezoid rule converges geometrically for
  // integrands analytic in a strip; here the strip is |Im u| < pi / 2 (the
  // progression rates z * lambda keep a positive real part), giving relative
  // errors around exp(-pi^2 / kappa).
  //
  // The likelihood is only evaluated at nodes with u in [u_lo, u_hi]. Outside
  // that window log L follows its asymptotic forms, as quadratics through
  // values that include the window's ends, so that the result stays
  // continuous as nodes cross the edges:
  //  * z -> 0:   log L is smooth in z (from z = 0, z_lo / e and z_lo)
  //  * z -> inf: log L is smooth in 1 / z, approaching its limit like 1 / z
  //              (a cancer starting within ~1 / z years of the last
  //              observation has not progressed; from z_hi / e^2, z_hi / e
  //              and z_hi)
  // So the number of likelihood evaluations is bounded (at most about
  // (u_hi - u_lo) / kappa + 7) for any sigma_re, and the expensive z beyond
  // e^u_hi (uniformization cost grows with z) are never evaluated.
  real log_marginal_lik(
      int                pt,
      real               sigma_re,
      data real          t_eta,
      data real          h_eta_max,
      data real          kappa,
      data real          u_lo,
      data real          u_hi,
      array[] vector     sens,
      array[] vector     sens_final,
      vector             b_cons,
      real               b_plco,
      real               m_NSCLC,
      real               c_NSCLC,
      vector             lambda,
      vector             xi,
      data real          beta_former,
      data real          beta_current,
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
      data array[] int   step_len_idx,
      data array[] int   pt_n_len,
      data array[,] real pt_len,
      data array[] real  pt_t_end,
      data vector        rho,
      data matrix        c_lagrange,
      data real          x_max,
      data real          h_max,
      data int           n_sub_max,
      data real          tol,
      data int           max_terms
    ) {

    real h   = fmin(h_eta_max, kappa / sigma_re);
    real s2  = 0.5 * square(sigma_re);
    int  J   = 0;
    while ((J + 1) * h <= t_eta) {
      J += 1;
    }
    real u_min = -sigma_re * J * h - s2;
    real u_max =  sigma_re * J * h - s2;

    // Left tail: log L(z) quadratic in z for z < exp(u_lo), through z = 0,
    // exp(u_lo - 1) and exp(u_lo)
    vector[3] xl = [0, exp(u_lo - 1), exp(u_lo)]';
    vector[3] yl = rep_vector(0, 3);
    if (u_min < u_lo) {
      for (k in 1:3) {
        yl[k] = patient_loglik(pt, xl[k], sens, sens_final, b_cons, b_plco, m_NSCLC, c_NSCLC, lambda, xi, beta_former, beta_current, plco, cigsmok, rndgroup, pt_first, pt_records, recordtype, event, final_screen, t_l, t_r, n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange, x_max, h_max, n_sub_max, tol, max_terms);
      }
    }
    // Right tail: log L(z) quadratic in 1 / z for z > exp(u_hi), through
    // z = exp(u_hi - 2), exp(u_hi - 1) and exp(u_hi)
    vector[3] xr = [exp(-u_hi), exp(1 - u_hi), exp(2 - u_hi)]';
    vector[3] yr = rep_vector(0, 3);
    if (u_max > u_hi) {
      for (k in 1:3) {
        yr[k] = patient_loglik(pt, 1 / xr[k], sens, sens_final, b_cons, b_plco, m_NSCLC, c_NSCLC, lambda, xi, beta_former, beta_current, plco, cigsmok, rndgroup, pt_first, pt_records, recordtype, event, final_screen, t_l, t_r, n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange, x_max, h_max, n_sub_max, tol, max_terms);
      }
    }

    real acc  = negative_infinity();
    real norm = negative_infinity();
    for (j in -J:J) {
      real eta = j * h;
      real lw  = -0.5 * square(eta);
      real u   = sigma_re * eta - s2;
      real ll;
      if (u < u_lo) {
        ll = quad_interp(xl, yl, exp(u));
      } else if (u > u_hi) {
        ll = quad_interp(xr, yr, exp(-u));
      } else {
        ll = patient_loglik(pt, exp(u), sens, sens_final, b_cons, b_plco, m_NSCLC, c_NSCLC, lambda, xi, beta_former, beta_current, plco, cigsmok, rndgroup, pt_first, pt_records, recordtype, event, final_screen, t_l, t_r, n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange, x_max, h_max, n_sub_max, tol, max_terms);
      }
      acc  = log_sum_exp(acc, lw + ll);
      norm = log_sum_exp(norm, lw);
    }
    return acc - norm;
  }

  // Weighted log likelihood for the weighted patients in mg_slice (for
  // reduce_sum), with the random effect integrated out (log_marginal_lik).
  // A weighted patient stands for wt patients, each with their own random
  // effect, so it is the marginal likelihood, not the likelihood given one
  // sampled eta, that is raised to the power wt.
  real partial_loglik_mg(
      array[] int        mg_slice,
      int                start,
      int                end,
      real               sigma_re,
      data vector        wt,
      data real          t_eta,
      data real          h_eta_max,
      data real          kappa,
      data real          u_lo,
      data real          u_hi,
      array[] vector     sens,
      array[] vector     sens_final,
      vector             b_cons,
      real               b_plco,
      real               m_NSCLC,
      real               c_NSCLC,
      vector             lambda,
      vector             xi,
      data real          beta_former,
      data real          beta_current,
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
      data array[] int   step_len_idx,
      data array[] int   pt_n_len,
      data array[,] real pt_len,
      data array[] real  pt_t_end,
      data vector        rho,
      data matrix        c_lagrange,
      data real          x_max,
      data real          h_max,
      data int           n_sub_max,
      data real          tol,
      data int           max_terms
    ) {

    real total = 0;
    for (i in 1:size(mg_slice)) {
      int pt = mg_slice[i];
      total += wt[pt] * log_marginal_lik(
        pt, sigma_re, t_eta, h_eta_max, kappa, u_lo, u_hi, sens, sens_final, b_cons, b_plco, m_NSCLC, c_NSCLC, lambda, xi, beta_former, beta_current, plco, cigsmok, rndgroup, pt_first, pt_records, recordtype, event, final_screen, t_l, t_r, n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange, x_max, h_max, n_sub_max, tol, max_terms);
    }
    return total;
  }

  // Beta(a, b) log density of p = inv_logit(x), summed over the elements of
  // x and computed from x itself: beta_logit_lpdf(x | a, b) equals
  // beta_lpdf(inv_logit(x) | a, b), but is finite for any x. beta_lpdf(p | a, b)
  // is NaN when p rounds to exactly 1 and b = 1 (it evaluates
  // (b - 1) * log1m(p) = 0 * -inf), and -inf when p rounds to 0 or 1
  // otherwise. Note that this is the density of p, not of x: the Jacobian of
  // the logit transform is not included and must be added separately. Being
  // an _lpdf function, it can be used as target += beta_logit_lpdf(x | a, b)
  // or x ~ beta_logit(a, b) (the same value: it has no constants that ~ drops).
  real beta_logit_lpdf(vector x, real a, real b) {
    return (a - 1) * sum(log_inv_logit(x)) + (b - 1) * sum(log1m_inv_logit(x))
           - num_elements(x) * lbeta(a, b);
  }

  // Categories for the posterior predictive check (as in sclc-model.stan):
  //
  //   1 pre-screening
  //   2 at T0 screen
  //   3 T0-T1 interval
  //   4 at T1 screen
  //   5 T1-T2 interval
  //   6 at T2 screen
  //   7 post-screening
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

  // Undiagnosed stage at time s, given the times each stage was entered
  // (positive infinity for stages never entered); 0 before stage 1 (or the
  // baseline stage) is entered
  int stage_at(vector T_enter, real s) {
    for (i in 1:7) {
      int k = 8 - i;
      if (T_enter[k] <= s) return k;
    }
    return 0;
  }

  // Simulate one patient's NSCLC history from the model, with the patient's
  // screens (times, T0/T1/T2 labels, final-screen flags) and end of
  // follow-up as given, and return ( category, stage ) of the diagnosis, or
  // ( 0, 0 ) if NSCLC is not diagnosed during follow-up. The random effect
  // is drawn afresh: the simulated patient is a new patient with the same
  // covariates and screening schedule.
  tuple(int, int) simulate_patient_rng(
      real           plco,
      int            cig,
      int            grp,
      int            n_scr,
      array[] real   scr_time,
      array[] int    scr_label,
      array[] int    scr_final,
      real           censor,
      array[] vector sens,
      array[] vector sens_final,
      vector         b_cons,
      real           b_plco,
      real           m_NSCLC,
      real           c_NSCLC,
      vector         lambda,
      vector         xi,
      real           sigma_re,
      real           beta_former,
      real           beta_current
    ) {

    real shape = m_NSCLC * (beta_former + (beta_current - beta_former) * cig);
    real rate  = exp(c_NSCLC + m_NSCLC * plco);
    real z     = exp(sigma_re * std_normal_rng() - 0.5 * square(sigma_re));

    // Baseline state: 0 = no lung cancer, k = undiagnosed stage k. Without
    // lung cancer at baseline, stage 1 is entered at a Gompertz time.
    int  k = categorical_rng(softmax(append_row(0.0, b_plco * plco + b_cons))) - 1;
    real t = 0;
    if (k == 0) {
      k = 1;
      t = log1p(shape * exponential_rng(1) / rate) / shape;
    }

    // Times of entering each stage, until clinical presentation at T_clin
    // from stage k
    vector[7] T_enter = rep_vector(positive_infinity(), 7);
    T_enter[k] = t;
    real T_clin = positive_infinity();
    while (is_inf(T_clin)) {
      real pres = exponential_rng(xi[k]);
      real prog = k < 7 ? exponential_rng(z * lambda[k]) : positive_infinity();
      if (prog < pres) {
        t += prog;
        k += 1;
        T_enter[k] = t;
      } else {
        T_clin = t + pres;
      }
    }
    int clin_stage = k;

    for (j in 1:n_scr) {
      real s = scr_time[j];
      if (T_clin <= s) {
        return (
          clinical_category(
            T_clin,
            j - 1,
            j > 1 ? scr_time[j - 1] : 0.0,
            j > 1 ? scr_label[j - 1] : 0
          ),
          clin_stage
        );
      }
      int st = stage_at(T_enter, s);
      if (st > 0) {
        real se = scr_final[j] == 1 ? sens_final[grp, st] : sens[grp, st];
        if (bernoulli_rng(se)) {
          return ( 2 + 2 * scr_label[j], st );
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
// The input data are as for the SCLC model, except that events are stages:
// N                The number of records
// N_pt             The number of patients
// wt               The weight for each patient (patients with wt != 1 have their
//                  random effect integrated out; see the header)
// raw_plco[N_pt]   The PLCOm2012 linear predictor for each patient
// mean_plco        The mean PLCOm2012 linear predictor
// beta_former,
// beta_current     Gompertz shape multipliers for former and current smokers
// cigsmok[N_pt]    Is patient current smoker? (1 = Yes, 0 = No)
// pt_records[N_pt] How many records are associated with each patient
// pt_first[N_pt]   The index (1:N) for the first record for each patient
// rndgroup[N_pt]   Randomisation group (1 or 2)
// t_l[N]           Years from randomisation to start of record
// t_r[N]           Years from randomisation to end of record
// recordtype[N]    Type of record (1 = Screen, 2 = Follow-up)
// event[N]         Event at end of record
//                    0   = Negative screen (recordtype = 1) / Survive (recordtype = 2)
//                    1-7 = Screen-detected / clinically presenting NSCLC at that stage
// final_screen[N]  Is this the final screen? (1 = Yes, 0 = No)
// grainsize        Grain size for reduce_sum (patients per work unit)
// N_pp, pp_*       Posterior predictive cohort (see the data block); N_pp = 0
//                  switches the simulation off
//
data {

  real mean_plco;
  real beta_current;
  real beta_former;

  int<lower=0>                      N;
  int<lower=0>                      N_pt;
  vector[N_pt]                      wt;
  vector[N_pt]                      raw_plco;
  array[N_pt] int<lower=0,upper=1>  cigsmok;
  array[N_pt] int<lower=1>          pt_records;
  array[N_pt] int<lower=1,upper=N>  pt_first;
  array[N_pt] int<lower=1,upper=2>  rndgroup;
  array[N]    int<lower=1,upper=2>  recordtype;
  array[N]    int<lower=0,upper=7>  event;
  array[N]    int<lower=0,upper=1>  final_screen;
  array[N]    real<lower=0>         t_l;
  array[N]    real<lower=0>         t_r;
  int<lower=1>                      grainsize;

  // Posterior predictive cohort, usually every trial participant (built by
  // nsclc_ppc_data.py). Each patient has up to 3 screens: times, labels
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

  // Prior means for the progression and presentation rates (gamma priors
  // with shape 4, so rate = 4 / mean, coefficient of variation 0.5). From
  // ten Haaf et al.'s stage-specific probabilities of progression P and mean
  // sojourn times T: for one stage, lambda + xi = 1 / T, xi = (1 - P) / T.
  //  * IA (P 0.85, T 1.82 y) split into IA1-IA3, each with progression
  //    probability p = 0.85^(1/3) and sojourn 1.82 / (1 + p + p^2) y (each
  //    sub-stage weighted by the probability of reaching it).
  //  * IB (0.88, 0.64 y) and II (0.93, 0.46 y) as given.
  //  * III combines IIIA (0.87, 0.46 y) and IIIB (0.76, 0.36 y):
  //    P = 0.87 * 0.76, T = 0.46 + 0.87 * 0.36.
  //  * IV: xi = 1 / 0.74.
  // The same means as the log-normal priors of the Gaussian-process
  // calibration. (Until 2026-11-23 the IA and III means ignored the
  // probabilities of reaching each sub-stage.)
  vector[6] lambda_prior_mean = [1.480541, 1.480541, 1.480541, 1.375, 2.021739, 0.855147]';
  vector[7] xi_prior_mean = [0.082418, 0.082418, 0.082418, 0.1875, 0.152174, 0.438179, 1.351351]';

  // Follow-up records are evolved in equal steps of at most MAX_STEP years.
  // Splitting a record does not change the likelihood. With DEG = 5, the
  // survival interpolation was accurate to ~1e-12 or better in tests, for
  // Gompertz hazards up to ~0.25/year.
  real MAX_STEP = 0.5;
  int  DEG      = 5;

  // Sub-steps: largest q * (sub-step) for uniformization, largest Gompertz
  // cumulative hazard per sub-step, and the most sub-steps per step before
  // the parameter values are rejected as absurd
  real X_MAX     = 30;
  real H_MAX     = 0.25;
  int  N_SUB_MAX = 1000;

  // Uniformization truncation tolerance and a safety cap on the number of
  // terms
  real TOL       = 1e-14;
  int  MAX_TERMS = 1000;

  array[N] int n_steps;
  for (r in 1:N) {
    n_steps[r] = max(1, to_int(ceil((t_r[r] - t_l[r]) / MAX_STEP)));
  }

  // Each patient's distinct step lengths. pt_len[pt, 1:pt_n_len[pt]] lists
  // them, and step_len_idx[r] points a follow-up record into that list (0 for
  // records needing no evolution). Lengths within a relative 1e-12 count as
  // equal.
  array[N] int    step_len_idx = rep_array(0, N);
  array[N_pt] int pt_n_len     = rep_array(0, N_pt);
  {
    array[max(pt_records)] real found;
    for (pt in 1:N_pt) {
      for (j in 1:pt_records[pt]) {
        int r = pt_first[pt] + j - 1;
        if (recordtype[r] == 2 && t_r[r] > t_l[r]) {
          real h = (t_r[r] - t_l[r]) / n_steps[r];
          int  l = 0;
          for (i in 1:pt_n_len[pt]) {
            if (abs(found[i] - h) <= 1e-12 * h) {
              l = i;
            }
          }
          if (l == 0) {
            pt_n_len[pt] += 1;
            l = pt_n_len[pt];
            found[l] = h;
          }
          step_len_idx[r] = l;
        }
      }
    }
  }
  // Each patient's last follow-up time, for the hazard bound in step_setup
  array[N_pt] real pt_t_end = rep_array(0.0, N_pt);
  for (pt in 1:N_pt) {
    for (j in 1:pt_records[pt]) {
      pt_t_end[pt] = fmax(pt_t_end[pt], t_r[pt_first[pt] + j - 1]);
    }
  }

  int L_MAX = max(1, max(pt_n_len));
  array[N_pt, L_MAX] real pt_len = rep_array(0.0, N_pt, L_MAX);
  for (pt in 1:N_pt) {
    for (j in 1:pt_records[pt]) {
      int r = pt_first[pt] + j - 1;
      if (step_len_idx[r] > 0) {
        pt_len[pt, step_len_idx[r]] = (t_r[r] - t_l[r]) / n_steps[r];
      }
    }
  }

  // Unweighted patients (sampled random effect) and weighted patients
  // (random effect integrated out), each in data order
  int N_RE = 0;
  for (pt in 1:N_pt) {
    N_RE += wt[pt] == 1;
  }
  int N_MG = N_pt - N_RE;
  array[N_RE]    int re_pts;
  array[N_MG] int mg_pts;
  {
    int i_re = 0;
    int i_mg = 0;
    for (pt in 1:N_pt) {
      if (wt[pt] == 1) {
        i_re += 1;
        re_pts[i_re] = pt;
      } else {
        i_mg += 1;
        mg_pts[i_mg] = pt;
      }
    }
  }

  // Integration over the random effect for weighted patients (see
  // log_marginal_lik): eta truncated to [-T_ETA, T_ETA] (tail mass 3e-12),
  // node spacing at most H_ETA_MAX in eta and KAPPA in u = log z, likelihood
  // evaluated for u in [U_LO, U_HI] and extrapolated outside. On recovery
  // set 1 thinned to 200 weighted patients (wt 246.5), the total log density
  // was within 6e-5, and its gradient within 2e-4, of a fine-grid reference
  // (KAPPA = 0.1, u in [-14, 9]) for sigma_re from 0.27 to 6; Gauss-Hermite
  // with 20 nodes was off by 0.1 to 0.4 for sigma_re >= 2.
  real T_ETA     = 7;
  real H_ETA_MAX = 0.6;
  real KAPPA     = 0.5;
  real U_LO      = -6;
  real U_HI      = 4;

  // Lagrange basis on DEG + 1 equally spaced points rho on [0, 1], as
  // monomial coefficients: L_j(rho) = sum_m c_lagrange[m + 1, j + 1] rho^m
  vector[DEG + 1] rho = linspaced_vector(DEG + 1, 0, 1);
  matrix[DEG + 1, DEG + 1] c_lagrange;
  {
    matrix[DEG + 1, DEG + 1] V;
    for (i in 1:(DEG + 1)) {
      for (m in 1:(DEG + 1)) {
        V[i, m] = rho[i]^(m - 1);
      }
    }
    c_lagrange = inverse(V);
  }
}

parameters {

  // Sensitivity increases with stage, is separate for LDCT and CXR
  // Sensitivity is not perfect for any stage
  array[2] ordered[7] logit_sens;

  // Threshold effect for the final screen (log odds ratio, per arm)
  vector<lower=0>[2] log_or_sens_final;

  // Baseline (prevalence screen) multinomial logit over stages 1..7
  real<lower=0> b_NSCLC_plco;
  vector<offset=-8, multiplier=2>[7] b_NSCLC_cons;

  // Gompertz incidence
  real<offset=-7.5> c_NSCLC;
  real<lower=0> m_NSCLC;

  // Progression (stage k to k+1) and clinical presentation rates.
  // Presentation rates increase with stage (clinically justified).
  vector<lower=0>[6] lambda;
  positive_ordered[7] xi;

  // Random effect on progression: scale and non-centred standard normals
  real<lower=0> sigma_re;
  array[N_RE] real eta;

}

transformed parameters {

  array[2] vector[7] sens;
  array[2] vector[7] sens_final;

  for (g in 1:2) {
    sens[g] = inv_logit(logit_sens[g]);

    // We want to specify priors on sensitivity, so we need a Jacobian adjustment
    jacobian += sum(log_inv_logit(logit_sens[g]) + log1m_inv_logit(logit_sens[g]));

    sens_final[g] = inv_logit(logit_sens[g] + log_or_sens_final[g]);
  }

}

model {

  // Beta priors on sensitivity (the Jacobian of the logit transform is in
  // transformed parameters). beta_logit_lpdf(logit_sens[g, k] | a, b)
  // equals beta_lpdf(sens[g, k] | a, b) but stays finite if sens rounds to
  // 0 or 1, which matters for Beta(19, 1) (NaN in beta_lpdf at sens = 1).
  // 70% sensitivity for Stage I, effective sample size 20
  target += beta_logit_lpdf(logit_sens[1, 1:4] | 14.0, 6.0);
  target += beta_logit_lpdf(logit_sens[2, 1:4] | 14.0, 6.0);
  // 80% sensitivity for Stage II, effective sample size 20
  target += beta_logit_lpdf(logit_sens[1, 5:5] | 16.0, 4.0);
  target += beta_logit_lpdf(logit_sens[2, 5:5] | 16.0, 4.0);
  // 90% sensitivity for Stage III, effective sample size 20
  target += beta_logit_lpdf(logit_sens[1, 6:6] | 18.0, 2.0);
  target += beta_logit_lpdf(logit_sens[2, 6:6] | 18.0, 2.0);
  // 95% sensitivity for Stage IV, effective sample size 20
  target += beta_logit_lpdf(logit_sens[1, 7:7] | 19.0, 1.0);
  target += beta_logit_lpdf(logit_sens[2, 7:7] | 19.0, 1.0);

  target += gamma_lpdf(log_or_sens_final | 4.0, 4.0 / 0.082);

  target += gamma_lpdf(b_NSCLC_plco | 10.0, 10.0);
  target += normal_lpdf(b_NSCLC_cons | -8, 2);

  matrix[2, 2] cov_preclin_inc = [[1, -0.8], [-0.8, 25.0 / 16.0]];
  target += multi_normal_lpdf([m_NSCLC, c_NSCLC] | [2, -7.5], cov_preclin_inc);

  // Weakly informative gamma priors with shape 4 and the prior means below
  // (the xi means need not be ordered; the ordering comes from the
  // positive_ordered constraint)
  target += gamma_lpdf(lambda | 4, 4 ./ lambda_prior_mean);
  target += gamma_lpdf(xi     | 4, 4 ./ xi_prior_mean);

  target += gamma_lpdf(sigma_re | 16, 16 / 0.607);  // mean 0.607, shape 16
  target += std_normal_lpdf(eta);

  // LIKELIHOOD
  // Unweighted patients, given their sampled random effects
  if (N_RE > 0) {
    target += reduce_sum(
      partial_loglik_re, eta, grainsize, re_pts,
      sens, sens_final, b_NSCLC_cons, b_NSCLC_plco, m_NSCLC, c_NSCLC,
      lambda, xi, sigma_re,
      beta_former, beta_current, plco, cigsmok, rndgroup,
      pt_first, pt_records, recordtype, event, final_screen, t_l, t_r,
      n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange,
      X_MAX, H_MAX, N_SUB_MAX, TOL, MAX_TERMS
    );
  }
  // Weighted patients, with the random effect integrated out
  if (N_MG > 0) {
    target += reduce_sum(
      partial_loglik_mg, mg_pts, grainsize,
      sigma_re, wt, T_ETA, H_ETA_MAX, KAPPA, U_LO, U_HI,
      sens, sens_final, b_NSCLC_cons, b_NSCLC_plco, m_NSCLC, c_NSCLC,
      lambda, xi, beta_former, beta_current, plco, cigsmok, rndgroup,
      pt_first, pt_records, recordtype, event, final_screen, t_l, t_r,
      n_steps, step_len_idx, pt_n_len, pt_len, pt_t_end, rho, c_lagrange,
      X_MAX, H_MAX, N_SUB_MAX, TOL, MAX_TERMS
    );
  }

}

generated quantities {
  // Posterior predictive counts of NSCLC diagnoses in the cohort pp_*, by
  // arm (1 LDCT, 2 CXR), stage (1-7: IA1, IA2, IA3, IB, II, III, IV) and
  // category (1 pre-screening, 2 T0 screen, 3 T0-T1 interval, 4 T1 screen,
  // 5 T1-T2 interval, 6 T2 screen, 7 post-screening; see clinical_category)
  array[2, 7, 7] int pp_count = rep_array(0, 2, 7, 7);
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
      sens,
      sens_final,
      b_NSCLC_cons,
      b_NSCLC_plco,
      m_NSCLC,
      c_NSCLC,
      lambda,
      xi,
      sigma_re,
      beta_former,
      beta_current
    );
    if (dx.1 > 0) {
      pp_count[pp_rndgroup[i], dx.2, dx.1] += 1;
    }
  }
}
