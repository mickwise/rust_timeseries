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
use crate::{
    duration::{
        core::validation::{
            validate_alpha, validate_beta, validate_omega, validate_stationarity_and_slack,
        },
        errors::ParamResult,
    },
    optimization::numerical_stability::transformations::{STATIONARITY_MARGIN, safe_softplus_inv},
};
use ndarray::Array1;

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
        validate_alpha(alpha.view(), p)?;
        validate_beta(beta.view(), q)?;
        validate_stationarity_and_slack(&alpha.view(), &beta.view(), slack)?;

        Ok(ACDParams { omega, slack, alpha, beta })
    }

    /// Convert validated model parameters into an unconstrained optimizer vector θ.
    ///
    /// **Backward transform** (model → optimizer):
    /// - `θ[0]` encodes `ω` via the inverse softplus: `θ[0] = ln(exp(ω) − 1)`.
    /// - `θ[1..]` are logits for **α** (length `p`) and **β** (length `q`).
    ///
    /// **Slack handling.** The slack component is **not** given its own logit in `θ`.
    /// Instead, during the forward transform (optimizer → model) the logits in
    /// `θ[1..]` are mapped through a softmax *with an implicit slack category*,
    /// then scaled by `1 − STATIONARITY_MARGIN`, ensuring:
    /// `sum(α) + sum(β) + slack = 1 − STATIONARITY_MARGIN` with `slack ≥ 0`.
    ///
    /// This mapping is useful for warm starts, exporting fits to optimizer space,
    /// or serializing an initial guess.
    ///
    /// # Length
    /// Returns an array of length `1 + p + q`.
    ///
    /// # Notes
    /// Assumes this instance already satisfies length and stationarity checks
    /// (i.e., it was built by [`ACDParams::new`]).
    pub fn to_theta(&self, p: usize, q: usize) -> Array1<f64> {
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

        theta
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
