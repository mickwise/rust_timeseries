//! Parameterization for ACD(p, q) models.
//!
//! This module defines the **constrained model-space** parameters [`ACDParams`]
//! and the **unconstrained optimizer-space** parameters [`ACDTheta`] together
//! with stable, invertible mappings between the two.
//!
//! Key ideas
//! ---------
//! - **Model space (`ACDParams`)**: parameters live on their natural domain
//!   (ω > 0, αᵢ ≥ 0, βⱼ ≥ 0, ∑α + ∑β < 1). The ψ-recursion and likelihood are
//!   evaluated in this space.
//! - **Optimizer space (`ACDTheta`)**: parameters are mapped to ℝᵏ via
//!   softplus/softmax so numerical optimizers can search without constraints.
//! - **Stationarity margin**: we enforce strict stationarity with a tiny buffer
//!   (default 1e-6) to avoid boundary pathologies.
//!
//! The bidirectional mapping (`ACDParams` ↔ `ACDTheta`) guarantees every optimizer
//! iterate corresponds to a valid, strictly stationary ACD model.
//!
//! Invariants
//! ----------
//! - ω > 0
//! - α, β elementwise ≥ 0
//! - ∑α + ∑β < 1 − margin
//! - slack ≥ 0 and ∑α + ∑β + slack = 1 − margin
use crate::duration::{
    core::validation::{
        validate_alpha, validate_beta, validate_omega, validate_stationarity_and_slack,
    },
    errors::{ParamError, ParamResult},
};
use ndarray::{Array1, s};

/// Safety margin for strict stationarity in ACD models.
///
/// In an ACD(p, q), the stability condition requires
///   sum(alpha) + sum(beta) < 1.
/// This margin enforces the inequality *strictly* by reserving a small
/// buffer (default = 1e-6). Practically, the recursion always runs inside
/// the stable region, avoiding borderline cases that can cause blow-ups
/// in likelihood evaluation.
const STATIONARITY_MARGIN: f64 = 1e-6;

/// Lower clamp used in logit/softmax transforms.
///
/// Prevents log/exp underflow when converting extremely small probabilities or
/// weights during the optimizer-space ↔ model-space mapping.
const LOGIT_EPS: f64 = 1e-15;

/// Model-space parameters for an ACD(p, q) model.
///
/// This is the constrained set actually used by the ψ-recursion and the
/// likelihood:
/// - `omega > 0`
/// - `alpha[i] ≥ 0` for i = 0..p-1 (coeffs on lagged durations)
/// - `beta[j]  ≥ 0` for j = 0..q-1 (coeffs on lagged ψ)
/// - `∑alpha + ∑beta < 1` (strict stationarity, enforced with a small margin)
///
/// With unit-mean innovations, the unconditional mean of durations is
/// `mu = omega / (1 − ∑alpha − ∑beta)`.
///
/// Construct with [`ACDParams::new`] to validate inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDParams {
    pub omega: f64,
    pub slack: f64,
    pub alpha: Array1<f64>,
    pub beta: Array1<f64>,
}

impl ACDParams {
    /// Construct validated ACD parameters in model (natural) space.
    ///
    /// Checks numeric validity and strict stationarity (with a safety margin).
    ///
    /// # Arguments
    /// - `omega`: baseline duration (> 0, finite)
    /// - `alpha`: length-`q` non-negative coefficients on lagged durations
    /// - `beta` : length-`p` non-negative coefficients on lagged conditional means
    /// - `slack`: non-negative remainder so that
    ///   `∑alpha + ∑beta + slack = 1 − STATIONARITY_MARGIN`
    /// - `q`, `p`: intended orders (used to validate `alpha.len()` / `beta.len()`)
    ///
    /// # Returns
    /// A validated [`ACDParams`] instance.
    ///
    /// # Errors
    /// - [`ParamError::InvalidOmega`] if `omega ≤ 0` or non-finite
    /// - [`ParamError::AlphaLengthMismatch`] / [`ParamError::BetaLengthMismatch`]
    ///   if lengths don’t match `q`/`p`
    /// - [`ParamError::InvalidAlpha`] / [`ParamError::InvalidBeta`] if any entry
    ///   is negative or non-finite
    /// - [`ParamError::InvalidSlack`] if `slack < 0` or non-finite
    /// - [`ParamError::StationarityViolated`] if
    ///   `∑alpha + ∑beta + slack ≠ 1 − STATIONARITY_MARGIN` (within tolerance)
    pub fn new(
        omega: f64, alpha: Array1<f64>, beta: Array1<f64>, slack: f64, p: usize, q: usize,
    ) -> ParamResult<Self> {
        validate_omega(omega)?;
        validate_alpha(&alpha, p)?;
        validate_beta(&beta, q)?;
        validate_stationarity_and_slack(&alpha, &beta, slack)?;

        Ok(ACDParams { omega, slack, alpha, beta })
    }

    /// Convert validated model parameters into an unconstrained optimizer vector θ.
    ///
    /// **Backward transform** (model → optimizer):
    /// - θ[0] encodes ω via the inverse softplus:
    ///   `θ[0] = ln(exp(ω) − 1)`
    /// - θ[1..] are logits for α, β, and a slack component that will later be
    ///   converted via a numerically stable softmax and scaled to enforce
    ///   strict stationarity.
    ///
    /// This is useful for warm starts, exporting fits to optimizer space, or
    /// serializing an initial guess.
    ///
    /// # Notes
    /// Assumes this instance already satisfies length and stationarity checks
    /// (i.e., it was built by [`ACDParams::new`]).
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
/// - θ[0] controls ω (baseline) through a softplus transform
/// - θ[1..] parameterize α, β, and a slack component via a stable softmax
///   which is then scaled by `(1 − STATIONARITY_MARGIN)`
///
/// Why it exists:
/// Optimizers can freely search in ℝᵏ without manual bound handling, while the
/// forward map back to [`ACDParams`] guarantees a valid, strictly stationary model.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDTheta {
    pub theta: Array1<f64>,
}

impl ACDTheta {
    /// Convert an unconstrained θ back to model parameters (ω, α, β).
    ///
    /// **Forward transform** (optimizer → model):
    /// - `ω = ln(1 + exp(θ[0]))` (softplus)
    /// - `[α, β, slack] = softmax(θ[1..]) * (1 − STATIONARITY_MARGIN)`
    ///
    /// # Errors
    /// - [`ParamError::ThetaLengthMismatch`] if `θ.len() != p + q + 2`
    /// - Any of the validation errors emitted by [`ACDParams::new`]
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
        let alpha =
            coeffs.slice(s![0..p]).to_owned().mapv_into(|x| x * (1.0 - STATIONARITY_MARGIN));
        let beta =
            coeffs.slice(s![p..p + q]).to_owned().mapv_into(|x| x * (1.0 - STATIONARITY_MARGIN));

        ACDParams::new(omega, alpha, beta, slack, p, q)
    }
}

// ---- Helper Methods ----

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
