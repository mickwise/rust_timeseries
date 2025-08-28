//! Zero-copy parameter workspace for ACD(p, q).
//!
//! This module provides a `WorkSpace` that holds **mutable views** of the
//! coefficient vectors (α of length `q`, β of length `p`) plus scalar `ω` and
//! `slack`, and exposes an in-place transform from optimizer-space parameters
//! `θ` to model-space parameters `(ω, α, β, slack)` with **no heap allocation**.
//!
//! ## Why a workspace?
//! During optimization, each `θ` update should not allocate or clone. The
//! `WorkSpace` reuses caller-owned buffers for α and β and fills them in place.
//! After convergence, you can materialize an owned `ACDParams` **once** from the
//! final buffers if you want a storable, immutable object.
//!
//! ## Invariants enforced after `update_workspace`
//! - `alpha.len() == q` and `beta.len() == p`
//! - entries are finite and non-negative
//! - `ω > 0`
//! - `sum(α) + sum(β) + slack = 1 − STATIONARITY_MARGIN`
//!
//! ## Performance notes
//! - Uses a **three-pass max-shift softmax** over `θ[1..]`:
//!   1) find max logit, 2) accumulate denominator, 3) normalize+scale+write.
//! - Fills α and β using `ndarray::Zip` into mutable views.
//!
//! ## Lifetimes
//! The lifetime `'a` of the workspace ties α/β views to caller-owned buffers.
//! `update_workspace` accepts a `theta` view with an **independent** lifetime
//! so you can pass any temporary optimizer vector.
use crate::{
    duration::{
        core::{
            shape::ACDShape,
            validation::{
                validate_alpha, validate_alpha_beta_lengths, validate_beta, validate_omega,
                validate_stationarity_and_slack,
            },
        },
        errors::{ParamError, ParamResult},
    },
    optimization::numerical_stability::transformations::{STATIONARITY_MARGIN, safe_softplus},
};
use ndarray::{ArrayView1, ArrayViewMut1, s};

/// Mutable, zero-copy buffer for ACD parameters during optimization.
///
/// Holds α (`q`) and β (`p`) as **mutable ndarray views** borrowed from the
/// caller, plus scalar `ω` and `slack`. Each call to [`update_workspace`]
/// **overwrites** these buffers from an unconstrained parameter vector `θ` with
/// no allocation.
///
/// After optimization, you may copy these buffers once to build an owned
/// `ACDParams` for storage/serialization.
#[derive(Debug, PartialEq)]
pub struct WorkSpace<'a> {
    pub alpha: ArrayViewMut1<'a, f64>,
    pub beta: ArrayViewMut1<'a, f64>,
    pub omega: f64,
    pub slack: f64,
}

impl<'a> WorkSpace<'a> {
    /// Construct a workspace over caller-owned α/β buffers.
    ///
    /// Validates **lengths only** against the provided shape
    /// (α has length `q`, β has length `p`). `ω` and `slack` are initialized to
    /// `0.0`; they are set/validated by [`update_workspace`].
    ///
    /// # Errors
    /// - [`ParamError::AlphaLengthMismatch`] if `alpha.len() != q`
    /// - [`ParamError::BetaLengthMismatch`]  if `beta.len()  != p`
    pub fn new(
        alpha: ArrayViewMut1<'a, f64>, beta: ArrayViewMut1<'a, f64>, shape: &ACDShape,
    ) -> ParamResult<Self> {
        validate_alpha_beta_lengths(&alpha.view(), &beta.view(), shape.q, shape.p)?;
        Ok(WorkSpace { alpha, beta, omega: 0.0, slack: 0.0 })
    }

    /// Overwrite the workspace from an unconstrained optimizer vector `θ`.
    ///
    /// Mapping (optimizer → model):
    /// - `ω = softplus(θ[0])`
    /// - Split the logits slice `θ[1..]` into `[α (0..q), β (q..q+p)]`,
    ///   then apply a **three-pass max-shift softmax** and scale all weights by
    ///   `1 − STATIONARITY_MARGIN`. Writes α and β **in place** and sets `slack`.
    ///
    /// This function performs **no heap allocation** and validates the resulting
    /// parameters.
    ///
    /// # Errors
    /// - [`ParamError::ThetaLengthMismatch`] if `θ.len() != p + q + 1`
    /// - [`ParamError::InvalidOmega`] if `ω` is not finite or ≤ 0
    /// - [`ParamError::InvalidAlpha`] / [`ParamError::InvalidBeta`] if any entry is
    ///   negative or non-finite
    /// - [`ParamError::StationarityViolated`] or [`ParamError::InvalidSlack`] if
    ///   `sum(α) + sum(β) + slack` does not equal `1 − STATIONARITY_MARGIN`
    ///
    /// # Notes
    /// The logits normalization uses `m = max(logits)` to avoid overflow, computing
    /// `exp(li − m)` for stability.
    pub fn update_workspace<'b>(
        &mut self, theta: ArrayView1<'b, f64>, shape: &ACDShape,
    ) -> ParamResult<()> {
        let p = shape.p;
        let q = shape.q;
        let n = theta.len();
        if theta.len() != p + q + 1 {
            return Err(ParamError::ThetaLengthMismatch { expected: p + q + 1, actual: n });
        }
        let omega = safe_softplus(theta[0]);
        validate_omega(omega)?;
        self.omega = omega;
        self.safe_softmax(&theta.slice(s![1..]), p, q);
        validate_alpha(self.alpha.view(), q)?;
        validate_beta(self.beta.view(), p)?;
        self.slack = 1.0 - STATIONARITY_MARGIN - self.alpha.sum() - self.beta.sum();
        validate_stationarity_and_slack(&self.alpha.view(), &self.beta.view(), self.slack)?;
        Ok(())
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

    /// Fill α and β from the logits slice using a three-pass, max-shift softmax.
    ///
    /// Expects `theta.len() == q + p`, corresponding to `[α logits, β logits]`.
    /// Performs:
    /// 1) `m = max(theta)`
    /// 2) `den = Σ exp(theta[i] − m)`
    /// 3) Writes:
    ///    - `α[i] = exp(theta[i] − m) / den * (1 − STATIONARITY_MARGIN)` for `i in 0..q`
    ///    - `β[j] = exp(theta[q + j] − m) / den * (1 − STATIONARITY_MARGIN)` for `j in 0..p`
    ///
    /// Implementation is **allocation-free** and uses `ndarray::Zip` to write
    /// directly into the α/β buffers.
    ///
    /// This is an internal function; callers should use [`update_workspace`].
    fn safe_softmax(&mut self, theta: &ArrayView1<f64>, p: usize, q: usize) -> () {
        let max_x = theta.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp_x: f64 = theta.iter().map(|&v| (v - max_x).exp()).sum();
        let scale = 1.0 - STATIONARITY_MARGIN;

        ndarray::Zip::from(self.alpha.view_mut())
            .and(&theta.slice(ndarray::s![0..q]))
            .for_each(|a, &v| *a = ((v - max_x).exp() / sum_exp_x) * scale);

        ndarray::Zip::from(self.beta.view_mut())
            .and(&theta.slice(ndarray::s![q..q + p]))
            .for_each(|b, &v| *b = ((v - max_x).exp() / sum_exp_x) * scale);
    }
}
