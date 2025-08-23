use ndarray::{Array1, s};

use crate::duration::duration_errors::{ParamError, ParamResult};

/// Safety margin for strict stationarity in ACD models.
///
/// In an ACD(p, q), the stability condition requires
///   sum(alpha) + sum(beta) < 1.
/// This margin enforces the inequality *strictly* by reserving a small
/// buffer (default = 1e-6). Practically, the recursion always runs inside
/// the stable region, avoiding borderline cases that can cause blow-ups
/// in likelihood evaluation.
const STATIONARITY_MARGIN: f64 = 1e-6;

/// Small value to avoid numerical issues in logit transforms.
const LOGIT_EPS: f64 = 1e-15;

/// Model-space parameters for an ACD(p, q) model.
///
/// This is the "natural" constrained space used by the recursion and likelihood:
/// - `omega > 0`
/// - `alpha[i] >= 0` for i = 0..p-1  (coeffs on lagged durations)
/// - `beta[j]  >= 0` for j = 0..q-1  (coeffs on lagged ψ)
/// - `sum(alpha) + sum(beta) < 1`    (strict stationarity)
///
/// With unit-mean innovations, the unconditional mean of durations is
/// `mu = omega / (1.0 - sum(alpha) - sum(beta))`.
///
/// The struct stores values that already satisfy these constraints; use
/// [`ACDParams::new`] to validate an instance built by hand.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDParams {
    pub omega: f64,
    pub slack: f64,
    pub alpha: Array1<f64>,
    pub beta: Array1<f64>,
}

impl ACDParams {
    /// Construct validated ACD parameters in the model (natural) space.
    ///
    /// It checks basic numeric validity and the strict-stationarity
    /// constraint with a safety margin.
    ///
    /// # Arguments
    /// - `omega`: baseline duration (must be finite and `> 0`).
    /// - `alpha`: length-`p` non-negative coefficients on lagged durations.
    /// - `beta` : length-`q` non-negative coefficients on lagged conditional means.
    /// - `slack`: non-negative remainder so that
    ///            `sum(alpha) + sum(beta) + slack = 1 - STATIONARITY_MARGIN`.
    /// - `p`, `q`: the intended ACD orders (used to validate `alpha.len()` and `beta.len()`).
    ///
    /// # Returns
    /// A validated [`ACDParams`] instance that satisfies all constraints.
    ///
    /// # Errors
    /// - [`ParamError::InvalidOmega`] if `omega ≤ 0` or non-finite.
    /// - [`ParamError::AlphaLengthMismatch`] / [`ParamError::BetaLengthMismatch`] if lengths don’t match `p`/`q`.
    /// - [`ParamError::InvalidAlpha`] / [`ParamError::InvalidBeta`] if any entry is negative or non-finite.
    /// - [`ParamError::InvalidSlack`] if `slack < 0` or non-finite.
    /// - [`ParamError::StationarityViolated`] if
    ///   `sum(alpha)+sum(beta)+slack ≠ 1 - STATIONARITY_MARGIN` (within tolerance).
    pub fn new(
        omega: f64,
        alpha: Array1<f64>,
        beta: Array1<f64>,
        slack: f64,
        p: usize,
        q: usize,
    ) -> ParamResult<Self> {
        validate_omega(omega)?;
        validate_alpha(&alpha, p)?;
        validate_beta(&beta, q)?;
        validate_stationarity_and_slack(&alpha, &beta, slack)?;

        Ok(ACDParams {
            omega,
            slack,
            alpha,
            beta,
        })
    }

    /// Convert validated ACD parameters into an unconstrained optimization vector θ.
    ///
    /// This is the **backward transform** used for warm starts or for exporting
    /// fitted parameters into optimizer space. The mapping is:
    ///
    /// - θ[0] encodes ω (intercept) via the inverse softplus:  
    ///       θ[0] = ln(exp(ω) - 1)
    /// - θ[1..] are logits for the α (lagged durations) and β (lagged conditional means),
    ///   expressed relative to a slack term. Internally these logits get converted
    ///   back to positive weights that satisfy the strict stationarity condition.
    ///
    /// # Errors
    /// Fails if:
    /// - α or β lengths don’t match (p, q),
    /// - any coefficient is invalid (negative or NaN),
    /// - or the stationarity condition (sum α + sum β < 1) is violated.
    pub fn to_theta(&self, p: usize, q: usize) -> ACDTheta {
        let theta0 = safe_softplus_inv(self.omega);
        let denom = 1.0 - STATIONARITY_MARGIN;
        let mut pi: Vec<f64> = Vec::with_capacity(p + q + 1);
        pi.extend(self.alpha.iter().map(|&v| v / denom));
        pi.extend(self.beta.iter().map(|&v| v / denom));
        pi.push(self.slack / denom);

        // Clamp small values to avoid numerical issues in log
        pi.iter_mut().for_each(|x| {
            if *x < LOGIT_EPS {
                *x = LOGIT_EPS;
            }
        });

        let log_pi_slack = pi.last().unwrap().ln();
        let mut theta = ndarray::Array1::<f64>::zeros(1 + p + q);
        theta[0] = theta0;

        // α logits
        for i in 0..p {
            theta[1 + i] = pi[i].ln() - log_pi_slack;
        }
        // β logits
        for j in 0..q {
            theta[1 + p + j] = pi[p + j].ln() - log_pi_slack;
        }

        ACDTheta { theta }
    }

    /// Implied unconditional mean of durations:
    ///
    /// μ = ω / (1 - sum(α) - sum(β))
    ///
    /// This is the long-run average duration implied by the fitted parameters,
    /// assuming unit-mean innovations.
    pub fn uncond_mean(&self) -> f64 {
        let sum_alpha = self.alpha.sum();
        let sum_beta = self.beta.sum();
        self.omega / (1.0 - sum_alpha - sum_beta)
    }
}

/// Unconstrained parameter vector θ used internally by the optimizer.
///
/// Layout:
/// - θ[0] controls ω (baseline duration) through a softplus transform.
/// - θ[1..] represent α, β, and a slack component via a numerically stable
///   softmax. After scaling by `(1 - STATIONARITY_MARGIN)`, they become the
///   actual model coefficients.
///
/// # Why this matters
/// In estimation, we want the optimizer to search freely in ℝ^k without
/// worrying about positivity or stationarity. This transform guarantees that
/// whatever θ the optimizer proposes will map back to a *valid stationary ACD model*.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDTheta {
    pub theta: Array1<f64>,
}

impl ACDTheta {
    /// Convert an unconstrained vector θ back into model parameters (ω, α, β).
    ///
    /// This is the **forward transform**:  
    /// - ω = ln(1 + exp(θ[0]))  
    /// - α, β are obtained by applying a stable softmax to θ[1..], scaled by  
    ///   (1 - STATIONARITY_MARGIN) to enforce stationarity.
    ///
    /// # Errors
    /// - If θ has wrong length (`p + q + 2` expected),
    /// - If recovered coefficients fail basic validity checks.
    pub fn to_params(&self, p: usize, q: usize) -> ParamResult<ACDParams> {
        if self.theta.len() != p + q + 2 {
            return Err(ParamError::ThetaLengthMismatch {
                expected: p + q + 2,
                actual: self.theta.len(),
            });
        }

        let omega = safe_softplus(self.theta[0]);
        let coeffs = safe_softmax(&self.theta.slice(s![1..]).to_owned());
        let slack = coeffs.last().unwrap() * (1.0 - STATIONARITY_MARGIN);
        let alpha = coeffs
            .slice(s![0..p])
            .to_owned()
            .mapv_into(|x| x * (1.0 - STATIONARITY_MARGIN));
        let beta = coeffs
            .slice(s![p..p + q])
            .to_owned()
            .mapv_into(|x| x * (1.0 - STATIONARITY_MARGIN));

        ACDParams::new(omega, alpha, beta, slack, p, q)
    }
}

/// ---- Helper Methods ----

/// Numerically stable softplus: `softplus(x) = ln(1 + exp(x))`.
///
/// Computes softplus without overflow for large positive `x` and
/// with good precision for large negative `x`. This implementation
/// uses a simple piecewise guard:
///
/// - For sufficiently large `x`, `softplus(x) ≈ x + ln1p(exp(-x)) ≈ x`.
/// - Otherwise, it falls back to `ln1p(exp(x))`.
///
/// The cutoff used here (`x > 20.0`) is a practical threshold that
/// keeps the calculation in a well-conditioned regime for `f64`
/// (similar to the strategy used in common ML libraries like PyTorch).
///
/// # Parameters
/// - `x`: real input
///
/// # Returns
/// - `softplus(x)` as `f64`.
fn safe_softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Stable inverse of softplus on `(0, ∞)`: solves for `t` in
/// `softplus(t) = x`, returning `t = ln(exp(x) - 1)`.
///
/// Direct evaluation of `ln(exp(x) - 1)` can overflow or lose precision.
/// This implementation mirrors the guarded strategy of `safe_softplus`:
///
/// - For sufficiently large `x`, `exp(-x)` is tiny and
///   `ln(exp(x) - 1) ≈ x + ln(1 - exp(-x)) ≈ x`.
/// - Otherwise, it uses `ln(expm1(x))`.
///
/// The cutoff (`x > 20.0`) is chosen for numerical robustness with `f64`.
///
/// # Parameters
/// - `x`: a positive real (the softplus output), must be finite and `> 0`.
///
/// # Returns
/// - `t` such that `softplus(t) = x`.
fn safe_softplus_inv(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp_m1().ln() }
}

/// Numerically stable softmax (1-D).
///
/// Used to transform logits into positive weights that sum to 1.
/// Essential for mapping optimizer variables into valid (α, β, slack).
///
/// Implements the standard trick of subtracting max(x) before exponentiation
/// to avoid overflow. Returns a probability vector (non-negative entries,
/// sum = 1).
fn safe_softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_x = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max_x).exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}

/// Validate the ω coefficient (baseline duration).
fn validate_omega(omega: f64) -> ParamResult<()> {
    if omega <= 0.0 || !omega.is_finite() {
        return Err(ParamError::InvalidOmega { value: omega });
    }
    Ok(())
}

/// Validate the α coefficients (lags on durations).
///
/// Checks length = p, all entries finite and non-negative.
///
/// # Errors
/// - Length mismatch,
/// - Any coefficient < 0 or NaN.
fn validate_alpha(alpha: &Array1<f64>, p: usize) -> ParamResult<()> {
    if alpha.len() != p {
        return Err(ParamError::AlphaLengthMismatch {
            expected: p,
            actual: alpha.len(),
        });
    }
    if let Some((index, &value)) = alpha
        .iter()
        .enumerate()
        .find(|(_, v)| **v < 0.0 || !(**v).is_finite())
    {
        return Err(ParamError::InvalidAlpha { index, value });
    }
    Ok(())
}

/// Validate the β coefficients (lags on conditional means).
///
/// Checks length = q, all entries finite and non-negative.
///
/// # Errors
/// - Length mismatch,
/// - Any coefficient < 0 or NaN.
fn validate_beta(beta: &Array1<f64>, q: usize) -> ParamResult<()> {
    if beta.len() != q {
        return Err(ParamError::BetaLengthMismatch {
            expected: q,
            actual: beta.len(),
        });
    }
    if let Some((index, &value)) = beta
        .iter()
        .enumerate()
        .find(|(_, v)| **v < 0.0 || !(**v).is_finite())
    {
        return Err(ParamError::InvalidBeta { index, value });
    }
    Ok(())
}

/// Validate the stationarity condition and slack coefficient.
///
/// Checks:
/// - slack is finite and ≥ 0,
/// - sum(α) + sum(β) + slack ≈ 1 - STATIONARITY_MARGIN within tolerance.
///
/// # Errors
/// - If slack is invalid (negative / non-finite),
/// - If the stationarity condition is violated.
///
/// # Quant note
/// Stationarity here means the duration process won’t diverge. The “margin”
/// ensures you’re not right on the boundary, which would cause instability
/// in estimation or simulation.
fn validate_stationarity_and_slack(
    alpha: &Array1<f64>,
    beta: &Array1<f64>,
    slack: f64,
) -> ParamResult<()> {
    if slack < 0.0 {
        return Err(ParamError::InvalidSlack { value: slack });
    }
    if !slack.is_finite() {
        return Err(ParamError::InvalidSlack { value: slack });
    }
    let sum_alpha = alpha.sum();
    let sum_beta = beta.sum();
    let total_sum = sum_alpha + sum_beta + slack;

    const SUM_TOL: f64 = 1e-10;
    let target = 1.0 - STATIONARITY_MARGIN;
    if (total_sum - target).abs() > SUM_TOL {
        return Err(ParamError::StationarityViolated {
            coeff_sum: total_sum / target,
        });
    }
    Ok(())
}
