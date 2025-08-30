//! ACD(p, q) model: analytic log-likelihood and gradient.
//!
//! This module wires an ACD(p, q) specification to the `LogLikelihood` trait.
//! It uses a zero-copy [`WorkSpace`] to transform optimizer parameters `θ` into
//! model parameters `(ω, α, β, slack)` without heap allocation, then evaluates
//! the log-likelihood and its **analytic gradient** w.r.t. `θ`.
//!
//! Key ideas:
//! - Parameters live in unconstrained space: `ω = softplus(θ₀)` and
//!   `(α, β, slack) = (1 − margin)·softmax(θ₁: )` (implicit slack).
//! - The ψ-recursion is computed allocation-free into a shared scratch buffer.
//! - The gradient uses the chain rule:
//!   1) accumulate ∂ℓ/∂ω, ∂ℓ/∂α, ∂ℓ/∂β via ψ-sensitivities;
//!   2) map to θ-space using the Jacobians of softplus and the scaled softmax.
//!
//! This impl is designed to be optimizer-friendly (Argmin), avoiding clones and
//! allocating only for the returned gradient vector.
use crate::{
    duration::{
        core::{
            data::ACDData,
            innovations::ACDInnovation,
            options::ACDOptions,
            params::ACDScratch,
            psi::{compute_derivative, compute_psi, likelihood_driver},
            shape::ACDShape,
            validation::validate_theta,
            workspace::WorkSpace,
        },
        errors::ACDError,
    },
    optimization::{
        errors::OptResult,
        loglik_optimizer::{Grad, LogLikelihood, OptimOutcome, Theta, maximize},
        numerical_stability::transformations::safe_logistic,
    },
};
use ndarray::{Array1, s};

/// ACD(p, q) model with analytic log-likelihood and gradient.
///
/// Encapsulates the model order (`shape`), unit-mean innovation family
/// (`innovation`), runtime options (`options`), and preallocated scratch
/// buffers (`scratch_bufs`) reused across evaluations. After fitting,
/// [`results`] stores the last optimization outcome.
///
/// # Notes
/// - Designed for allocation-free inner loops: parameter transforms, ψ recursion,
///   and sensitivity recursion operate in-place on `scratch_bufs`.
/// - Implements [`LogLikelihood`] so it plugs directly into Argmin-based optimizers.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDModel {
    /// ACD(p, q) model order.
    pub shape: ACDShape,
    /// Innovation distribution with unit-mean parametrization.
    pub innovation: ACDInnovation,
    /// Model options.
    pub options: ACDOptions,
    /// Workspace buffers.
    pub scratch_bufs: ACDScratch,
    /// Fit results (populated after `fit`).
    pub results: Option<OptimOutcome>,
}

impl ACDModel {
    /// Construct a new [`ACDModel`] with preallocated scratch buffers.
    ///
    /// # Arguments
    /// - `shape`: model order (p, q) with p, q ≥ 0 and p + q > 0.
    /// - `innovation`: unit-mean innovation family (e.g., Exponential, Weibull).
    /// - `options`: run-time options (guards, initialization).
    /// - `n`: number of observations; used to size internal buffers.
    ///
    /// # Returns
    /// A model instance with zero-copy scratch space sized for `n`, `p`, and `q`.
    pub fn new(
        shape: ACDShape, innovation: ACDInnovation, options: ACDOptions, n: usize,
    ) -> ACDModel {
        let p = shape.p;
        let q = shape.q;
        let scratch_bufs = ACDScratch::new(n, p, q);
        ACDModel { shape, innovation, options, scratch_bufs, results: None }
    }

    /// Fit ACD(p, q) by MLE (takes ownership of `theta0`, no clones).
    ///
    /// # Steps
    /// 1. Validate shapes/finiteness of `theta0`.
    /// 2. Wrap `(self, data)` in `ArgMinAdapter` (cost/grad are analytic).
    /// 3. Build L-BFGS (Hager–Zhang or More–Thuente) per options.
    /// 4. Move `theta0` into the executor (`state.param(theta0)`), run.
    /// 5. Store the `OptimOutcome` in `self.results`.
    ///
    /// # Arguments
    /// - `theta0`: initial unconstrained vector **taken by value** (consumed)
    /// - `data`: model data
    ///
    /// # Returns
    /// - `Ok(())` on success; outcome is available via `self.results`
    ///
    /// # Notes
    /// - Ownership of `theta0` is **consumed**; no internal clones are made.
    pub fn fit(&mut self, theta0: Array1<f64>, data: &ACDData) -> OptResult<()> {
        self.results = Some(maximize(self, theta0, data, &self.options.mle_opts)?);
        Ok(())
    }
}

impl LogLikelihood for ACDModel {
    type Data = ACDData;

    /// Log-likelihood evaluation at parameter vector `θ`.
    ///
    /// # Steps
    /// 1. Transform `θ` → `(ω, α, β, slack)` via [`WorkSpace`] (no allocation).
    /// 2. Run ψ-recursion into scratch buffer.
    /// 3. Accumulate log-likelihood from the innovation density.
    ///
    /// # Arguments
    /// - `theta`: unconstrained optimizer vector (len = 1 + p + q).
    /// - `data`: observed durations.
    ///
    /// # Returns
    /// - Scalar log-likelihood `ℓ(θ)`.
    ///
    /// # Errors
    /// - Returns [`ACDError`] if `θ` is invalid or data checks fail.
    fn value(&self, theta: &Theta, data: &Self::Data) -> OptResult<f64> {
        let mut workspace_alpha = self.scratch_bufs.alpha_buf.borrow_mut();
        let mut workspace_beta = self.scratch_bufs.beta_buf.borrow_mut();
        let mut workspace =
            WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &self.shape)?;
        workspace.update_workspace(theta.view(), &self.shape)?;
        Ok(likelihood_driver(&self, &workspace, &data)?)
    }

    /// Validate an unconstrained parameter vector `θ`.
    ///
    /// # Behavior
    /// - Checks `θ.len() == 1 + p + q`.
    /// - Ensures all entries are finite.
    ///
    /// # Arguments
    /// - `theta`: unconstrained optimizer vector.
    ///
    /// # Returns
    /// - `Ok(())` if valid, error otherwise.
    fn check(&self, theta: &Theta, _data: &Self::Data) -> OptResult<()> {
        validate_theta(theta.view(), self.shape.p, self.shape.q)?;
        Ok(())
    }

    /// Analytic gradient of log-likelihood w.r.t. unconstrained `θ`.
    ///
    /// # Steps
    /// 1. Transform `θ` → `(ω, α, β, slack)` via [`WorkSpace`].
    /// 2. Run ψ-recursion and derivative recursion into scratch buffers.
    /// 3. Accumulate ∂ℓ/∂ψ terms for each observation.
    /// 4. Chain rule to θ-space:
    ///    - multiply ∂ℓ/∂ω by `σ(θ₀)` (softplus’ derivative),
    ///    - map ∂ℓ/∂α, ∂ℓ/∂β via scaled softmax Jacobian.
    ///
    /// # Arguments
    /// - `theta`: unconstrained optimizer vector.
    /// - `data`: observed durations.
    ///
    /// # Returns
    /// - Gradient vector `∇ℓ(θ)` of length `1 + p + q`.
    ///
    /// # Errors
    /// - Returns [`ACDError`] if invalid params or derivative recursion fails.
    fn grad(&self, theta: &Theta, data: &Self::Data) -> OptResult<Grad> {
        let p = self.shape.p;
        let q = self.shape.q;
        let mut workspace_alpha = self.scratch_bufs.alpha_buf.borrow_mut();
        let mut workspace_beta = self.scratch_bufs.beta_buf.borrow_mut();
        let mut workspace =
            WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &self.shape)?;
        workspace.update_workspace(theta.view(), &self.shape)?;
        compute_psi(&workspace, data, &self);
        compute_derivative(&workspace, data, &self);
        let deriv_binding = self.scratch_bufs.deriv_buf.borrow();
        let mut start_idx = p;
        if data.t0.is_some() {
            start_idx += data.t0.unwrap();
        }
        let psi_derives = deriv_binding.slice(s![start_idx.., ..]);
        let psi_lags_binding = self.scratch_bufs.psi_buf.borrow();
        let psi_lags = psi_lags_binding.slice(s![start_idx..]);
        let mut grad = psi_derives
            .rows()
            .into_iter()
            .zip(data.data.slice(s![start_idx..]).iter())
            .zip(psi_lags.iter())
            .try_fold(
                ndarray::Array1::<f64>::zeros(1 + p + q),
                |mut acc, ((deriv_row, &x), &psi)| {
                    let innov_grad = self.innovation.one_d_loglik_grad(x, psi)?;
                    acc += &(&deriv_row * innov_grad);
                    Ok::<Array1<f64>, ACDError>(acc)
                },
            )?;
        grad[0] *= safe_logistic(theta[0]);
        workspace.safe_softmax_deriv(&mut grad.slice_mut(s![1..]), p, q);
        Ok(grad)
    }
}
