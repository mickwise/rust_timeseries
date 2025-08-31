//! ACD(p, q) parameterization and scratch workspace.
//!
//! This module provides the **model-space** parameter container [`ACDParams`]
//! and a **zero-copy workspace** [`ACDScratch`] used by likelihood, gradient,
//! and forecasting routines elsewhere in the crate. It also implements a
//! **numerically stable mapping** from model space to an **optimizer-space
//! vector** θ (as `ndarray::Array1<f64>`).
//!
//! ## What this module defines
//! - [`ACDParams`]: validated model-space parameters `(ω, α, β, slack, ψ-lags)`,
//!   plus helpers such as the unconditional mean and a mapping to θ.
//! - [`ACDScratch`]: reusable buffers for α, β, ψ, duration lags, and
//!   derivatives so inner loops run allocation-free.
//!
//! ## Mapping conventions
//! - `θ₀ = softplus⁻¹(ω)` ensures `ω > 0`.
//! - `(α, β, slack)` live on a **scaled simplex** of total mass
//!   `(1 − margin)`, enforcing strict stationarity. The optimizer space stores
//!   **log-odds relative to slack** (slack is the softmax baseline).
//! - The forward map `ACDParams::to_theta` writes normalized probabilities
//!   to the α/β slots and then converts them in-place to log-odds using
//!   `ln(π_k) − ln(π_slack)`, clamping tiny probabilities to `LOGIT_EPS`
//!   before taking logs for numerical stability.
//!
//! ## Stationarity and slack
//! - Strict stationarity is enforced via a small **safety margin**
//!   (default `1e-6`), so `∑α + ∑β < 1 − margin`.
//! - `slack ≥ 0` completes the mass: `∑α + ∑β + slack = 1 − margin`.
//! - Using slack as the softmax baseline guarantees the constraint is
//!   respected for all optimizer iterates.
//!
//! ## Invariants validated by constructors
//! - `ω > 0`
//! - `α.len() == q` and `α ≥ 0` elementwise
//! - `β.len() == p` and `β ≥ 0` elementwise
//! - `∑α + ∑β + slack = 1 − margin` with `slack ≥ 0`
//! - `psi_lags.len() == p` and finite (used by forecasting)
//!
//! ## Scratch buffers (sizes)
//! - `alpha_buf`: length `q`
//! - `beta_buf`:  length `p`
//! - `psi_buf`:   length `n + p` (holds pre-sample ψ lags)
//! - `dur_buf`:   length `q`
//! - `deriv_buf`: shape `(n + p, 1 + q + p)`
//!
//! These buffers are zero-initialized once and then reused to avoid
//! allocations in hot paths.
use crate::{
    duration::{
        core::{
            shape::ACDShape,
            validation::{
                validate_alpha, validate_beta, validate_omega, validate_psi_lags,
                validate_stationarity_and_slack, validate_theta,
            },
        },
        errors::ParamResult,
    },
    optimization::numerical_stability::transformations::{
        LOGIT_EPS, STATIONARITY_MARGIN, safe_softmax, safe_softplus, safe_softplus_inv,
    },
};
use ndarray::{Array1, Array2, ArrayView1, Zip, s};
use std::cell::RefCell;

/// Zero-copy scratch workspace for ACD estimation and forecasting.
///
/// Holds mutable arrays reused across likelihood, gradient, and forecasting
/// steps so that hot loops run allocation-free.  Sizes are determined by the
/// sample length `n` and model orders `(p, q)`.
///
/// The buffers are zero-initialized at construction and then reused.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDScratch {
    /// Scratch buffer for α.
    pub alpha_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for β.
    pub beta_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for ψ.
    pub psi_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for initial durations.
    pub dur_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for derivatives.
    pub deriv_buf: RefCell<Array2<f64>>,
}

impl ACDScratch {
    /// Construct a new [`ACDScratch`] sized for a series of length `n`
    /// and orders `(p, q)`.
    ///
    /// - `alpha_buf` has length `q`
    /// - `beta_buf` has length `p`
    /// - `psi_buf` has length `n + p` (to hold pre-sample ψ lags)
    /// - `dur_buf` has length `q`
    /// - `deriv_buf` has shape `(n + p, 1 + q + p)`
    ///
    /// Returns a workspace with all arrays zero-initialized.  No further
    /// allocations are performed when reusing this workspace in inner loops.
    pub fn new(n: usize, p: usize, q: usize) -> ACDScratch {
        let alpha_buf = RefCell::new(Array1::zeros(q));
        let beta_buf = RefCell::new(Array1::zeros(p));
        let psi_buf = RefCell::new(Array1::zeros(n + p));
        let dur_buf = RefCell::new(Array1::zeros(q));
        let deriv_buf = RefCell::new(Array2::zeros((n + p, 1 + p + q)));
        ACDScratch { alpha_buf, beta_buf, psi_buf, dur_buf, deriv_buf }
    }
}

/// Constrained **model-space** parameters for an ACD(p, q) model.
///
/// Invariants are validated at construction; use this type to evaluate
/// the ψ-recursion, likelihood, and to generate forecasts in model space.
///
/// See [`ACDParams::to_theta`] for the optimizer-space mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDParams {
    /// ω > 0
    pub omega: f64,
    /// αᵢ ≥ 0
    pub alpha: Array1<f64>,
    /// βⱼ ≥ 0
    pub beta: Array1<f64>,
    /// slack ≥ 0
    pub slack: f64,
    /// Last p psi lags
    pub psi_lags: Array1<f64>,
}

impl ACDParams {
    /// Create validated model-space parameters.
    ///
    /// Validates:
    /// - `omega > 0`
    /// - `alpha.len() == q` and `alpha ≥ 0` elementwise
    /// - `beta.len()  == p` and `beta  ≥ 0` elementwise
    /// - `∑alpha + ∑beta + slack = 1 − margin` with `slack ≥ 0`
    /// - `psi_lags.len() == p`
    /// - `psi_lags finite and positive` elementwise
    ///
    /// Returns an error if any check fails.  On success the returned
    /// parameters satisfy strict stationarity (via the configured margin).
    pub fn new(
        omega: f64, alpha: Array1<f64>, beta: Array1<f64>, slack: f64, psi_lags: Array1<f64>,
        p: usize, q: usize,
    ) -> ParamResult<Self> {
        validate_omega(omega)?;
        validate_alpha(alpha.view(), q)?;
        validate_beta(beta.view(), p)?;
        validate_stationarity_and_slack(alpha.view(), beta.view(), slack)?;
        validate_psi_lags(&psi_lags, p)?;
        Ok(ACDParams { omega, alpha, beta, slack, psi_lags })
    }

    /// Build validated model-space parameters from an optimizer-space vector θ
    /// and the last `p` in-sample ψ-lags.
    ///
    /// ### Inputs
    /// - `theta`: optimizer-space parameters with layout
    ///   `θ = [θ₀ | α(1..q) | β(1..p)]`, where `θ₀ = softplus⁻¹(ω)` and the
    ///   α/β blocks are logits relative to the slack component (scaled softmax).
    /// - `shape`: model orders `(p, q)`
    /// - `psi_lags`: last `p` in-sample ψ values, used later for forecasting
    ///
    /// ### Behavior
    /// 1. Recovers `ω = softplus(θ₀)`.
    /// 2. Applies a numerically stable, max-shift softmax to `θ[1..]` to obtain
    ///    `(α, β, slack)` on the scaled simplex of total mass `1 − margin`.
    /// 3. Computes `slack = (1 − margin) − ∑α − ∑β`.
    /// 4. Validates parameter domains and stationarity, and verifies `psi_lags`.
    ///
    /// ### Requirements
    /// - `theta.len() == 1 + q + p`
    /// - `psi_lags.len() == p` and finite
    ///
    /// ### Returns
    /// A fully validated [`ACDParams`] containing owned `alpha`, `beta`, `omega`,
    /// `slack`, and `psi_lags`. On invalid input, returns a descriptive error.
    ///
    /// ### Notes
    /// - This constructor is allocation-minimal (one `q`-vector and one `p`-vector)
    ///   and performs no temporary array allocations beyond those.
    /// - It is intended for post-fit materialization: pass the optimizer’s final θ
    ///   and the cached ψ-lags to persist parameters for forecasting/inference.
    pub fn from_theta(
        theta: ArrayView1<f64>, shape: &ACDShape, psi_lags: Array1<f64>,
    ) -> ParamResult<Self> {
        let p = shape.p;
        let q = shape.q;
        validate_theta(theta, p, q)?;
        validate_psi_lags(&psi_lags, p)?;
        let mut alpha = Array1::zeros(q);
        let mut beta = Array1::zeros(p);
        let omega = safe_softplus(theta[0]);
        validate_omega(omega)?;
        safe_softmax(alpha.view_mut(), beta.view_mut(), &theta.slice(s![1..]), p, q);
        validate_alpha(alpha.view(), q)?;
        validate_beta(beta.view(), p)?;
        let slack = 1.0 - STATIONARITY_MARGIN - alpha.sum() - beta.sum();
        validate_stationarity_and_slack(alpha.view(), beta.view(), slack)?;
        Ok(ACDParams { omega, alpha, beta, slack, psi_lags })
    }

    /// Map model-space parameters to **optimizer-space** θ.
    ///
    /// Layout: `θ = [ θ₀ | α(1..q) | β(1..p) ]`, where:
    /// - `θ₀ = softplus⁻¹(ω)`
    /// - The α/β blocks are **log-odds relative to slack** under a
    ///   **scaled softmax** of total mass `(1 − margin)`.
    ///
    /// Implementation outline:
    /// 1. Write the normalized probabilities `π_i = component_i / (1 − margin)`
    ///    into the α/β slots (no extra allocations).
    /// 2. Clamp very small `π_i` to `LOGIT_EPS` for numerical safety.
    /// 3. Compute `log_slack = ln(slack / (1 − margin))`.
    /// 4. Convert each slot to `ln(π_i) − log_slack`.
    /// 5. Set `θ₀ = softplus_inv(ω)`.
    ///
    /// Returns a newly allocated `Array1<f64>` of length `1 + q + p`.
    ///
    /// ### Notes
    /// - Assumes this instance already satisfies the model-space invariants.
    pub fn to_theta(&self, p: usize, q: usize) -> Array1<f64> {
        let theta0 = safe_softplus_inv(self.omega);
        let denom_inv = 1.0 / (1.0 - STATIONARITY_MARGIN);
        let mut theta = ndarray::Array1::<f64>::zeros(1 + p + q);
        self.populate_simple_theta(&mut theta, denom_inv, p, q);

        // Clamp small values to avoid numerical issues in log
        theta.iter_mut().for_each(|x| {
            if *x < LOGIT_EPS {
                *x = LOGIT_EPS;
            }
        });

        let log_slack = (self.slack * denom_inv).ln();
        theta[0] = theta0;
        self.apply_inv_softmax(&mut theta, log_slack, p, q);
        theta
    }

    /// Unconditional mean of durations under the fitted ACD(p, q):
    ///
    /// ```text
    /// E[τ] = ω / (1 − ∑α − ∑β)
    /// ```
    ///
    /// Requires strict stationarity (`∑α + ∑β < 1`), which is enforced at
    /// construction.  Useful as a baseline forecast or for sanity checks.
    pub fn uncond_mean(&self) -> f64 {
        let sum_alpha = self.alpha.sum();
        let sum_beta = self.beta.sum();
        self.omega / (1.0 - sum_alpha - sum_beta)
    }

    /// Write normalized α and β into the θ buffer **in place**.
    ///
    /// Fills:
    /// - `θ[1 .. 1+q)`  ←  `alpha[i] / (1 − margin)`
    /// - `θ[1+q .. 1+q+p)` ← `beta[j] / (1 − margin)`
    ///
    /// This method:
    /// - **Does not** touch `θ[0]` (reserved for `softplus⁻¹(ω)`),
    /// - Performs **no allocations** and uses elementwise writes,
    /// - Assumes `alpha.len() == q` and `beta.len() == p`.
    ///
    /// The values written are *probabilities* (to be converted to
    /// log-odds later), not logits.
    fn populate_simple_theta(&self, theta: &mut Array1<f64>, denom_inv: f64, p: usize, q: usize) {
        let mut alpha_slice = theta.slice_mut(s![1..1 + q]);
        Zip::from(&mut alpha_slice).and(self.alpha.view()).for_each(|t, &a| *t = a * denom_inv);

        let mut beta_slice = theta.slice_mut(s![1 + q..1 + q + p]);
        Zip::from(&mut beta_slice).and(self.beta.view()).for_each(|t, &b| *t = b * denom_inv);
    }

    /// Convert normalized α/β **probabilities** in `θ` to **log-odds**
    /// relative to the slack probability.
    ///
    /// Expects:
    /// - The α block `θ[1 .. 1+q)` and the β block `θ[1+q .. 1+q+p)`
    ///   to currently store `π_i = component_i / (1 − margin)`.
    /// - `log_slack = ln(π_slack)`, where
    ///   `π_slack = slack / (1 − margin) = 1 − ∑π_α − ∑π_β`.
    ///
    /// Performs the in-place transform:
    /// `θ_k ← ln(π_k) − log_slack` for every α/β slot.
    ///
    /// Does not modify `θ[0]`.  No allocations.
    fn apply_inv_softmax(&self, theta: &mut Array1<f64>, log_slack: f64, p: usize, q: usize) {
        let mut alpha_slice = theta.slice_mut(s![1..1 + q]);
        alpha_slice.iter_mut().for_each(|t| *t = (*t).ln() - log_slack);

        let mut beta_slice = theta.slice_mut(s![1 + q..1 + q + p]);
        beta_slice.iter_mut().for_each(|t| *t = (*t).ln() - log_slack);
    }
}
