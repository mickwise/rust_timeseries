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
            forecasts::{ACDForecastResult, forecast_recursion},
            innovations::ACDInnovation,
            options::ACDOptions,
            params::{ACDParams, ACDScratch},
            psi::{compute_psi, likelihood_driver},
            shape::ACDShape,
            validation::validate_theta,
            workspace::WorkSpace,
        },
        errors::{ACDError, ACDResult},
        models::model_internals::{ObsEntry, calculate_scores, extract_theta, walk_observations},
    },
    inference::{HACOptions, calculate_avg_scores, hessian::calc_standard_errors},
    optimization::{
        errors::OptResult,
        loglik_optimizer::{Grad, LogLikelihood, OptimOutcome, Theta, maximize},
        numerical_stability::transformations::{safe_logistic, safe_softmax_deriv},
    },
};
use ndarray::{Array1, s};
use std::cell::RefCell;

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
    /// Fitted parameters (populated after `fit`).
    pub fitted_params: Option<ACDParams>,
    /// Forecasting results (populated after `predict`).
    pub forecast: Option<ACDForecastResult>,
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
        ACDModel {
            shape,
            innovation,
            options,
            scratch_bufs,
            results: None,
            fitted_params: None,
            forecast: None,
        }
    }

    /// Fit ACD(p, q) by maximum likelihood (consumes `theta0`) and cache results.
    ///
    /// ## Steps
    /// 1. Validate `theta0` (shape/finite) and set up the Argmin adapter
    ///    (analytic value/gradient).
    /// 2. Run L-BFGS per `options.mle_opts`, **moving** `theta0` into the executor.
    /// 3. Store the optimizer outcome (including `theta_hat`) in `self.results`.
    /// 4. **Recompute ψ at `theta_hat`** using the workspace (allocation-free) to
    ///    ensure the ψ buffer corresponds exactly to the best parameters.
    /// 5. Copy the **last `p` in-sample ψ-lags** from `psi_buf[len − p ..]`.
    /// 6. Map `(theta_hat, ψ-lags)` to model space via `ACDParams::from_theta` and
    ///    store in `self.fitted_params` (one small copy of `p` floats).
    ///
    /// ## Arguments
    /// - `theta0`: initial unconstrained parameter vector (owned; consumed)
    /// - `data`: observed duration series
    ///
    /// ## Returns
    /// - `Ok(())` on success; `self.results` and `self.fitted_params` are populated.
    ///
    /// ## Notes
    /// - This method performs **no internal clones** of `theta0`; Argmin takes ownership.
    /// - `self.results.theta_hat` is retained for warm starts, while
    ///   `self.fitted_params` is a self-contained snapshot (ω, α, β, slack, ψ-lags)
    ///   for forecasting/inference.
    pub fn fit(&mut self, theta0: Array1<f64>, data: &ACDData) -> OptResult<()> {
        let p = self.shape.p;
        self.results = Some(maximize(self, theta0, data, &self.options.mle_opts)?);
        let theta_hat = self.results.as_ref().unwrap().theta_hat.view();
        let mut workspace_alpha = self.scratch_bufs.alpha_buf.borrow_mut();
        let mut workspace_beta = self.scratch_bufs.beta_buf.borrow_mut();
        let mut workspace =
            WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &self.shape)?;
        workspace.update_workspace(theta_hat, &self.shape)?;
        compute_psi(&workspace, data, &self);
        let psi_lags = self.scratch_bufs.psi_buf.borrow();
        self.fitted_params = Some(ACDParams::from_theta(
            self.results.as_ref().unwrap().theta_hat.view(),
            &self.shape,
            psi_lags.slice(s![psi_lags.len() - p..]).to_owned(),
        )?);
        Ok(())
    }

    /// Forecast ψ (and τ, since E[τ] = ψ under unit-mean innovations) `horizon`
    /// steps ahead from the fitted ACD(p, q) model.
    ///
    /// ## Inputs
    /// - `horizon`: number of steps to forecast (H ≥ 1).
    /// - `data`: observed duration series; the last `q` values are used as lagged
    ///   durations in the recursion.
    ///
    /// ## Behavior
    /// 1. Requires that the model has been fitted (`self.fitted_params` present).
    /// 2. Extracts the last `q` durations from `data`.
    /// 3. Initializes a new [`ACDForecastResult`] of length `horizon`.
    /// 4. Runs [`forecast_recursion`] with the fitted parameters, the duration lags,
    ///    the cached ψ-lags in `fitted_params`, and the configured [`PsiGuards`].
    /// 5. Stores the full forecast path in `self.forecast` and returns the final
    ///    ψ forecast (at horizon H).
    ///
    /// ## Returns
    /// - `Ok(ψ̂_{T+H})` — the H-step-ahead forecast of the conditional mean duration.
    ///
    /// ## Side effects
    /// - Sets `self.forecast` to the new [`ACDForecastResult`] containing all
    ///   intermediate forecasts ψ̂[0..H).
    ///
    /// ## Errors
    /// - Returns [`ACDError::ModelNotFitted`] if called before fitting.
    /// - Propagates errors from [`forecast_recursion`] (e.g., lag length mismatch).
    ///
    /// ## Notes
    /// - Duration forecasts equal ψ forecasts under the unit-mean innovation
    ///   parameterization: `τ̂ = ψ̂`.
    /// - The full forecast path is retrievable via `self.forecast` after calling.
    pub fn predict(&mut self, horizon: usize, data: &ACDData) -> ACDResult<f64> {
        let forecast_result = ACDForecastResult::new(horizon);
        let n = data.data.len();
        let q = self.shape.q;
        let duration_lags = data.data.slice(s![n - q..]);
        let fitted_params = self.fitted_params.as_ref().ok_or(ACDError::ModelNotFitted)?;
        let h_forecast = forecast_recursion(
            &fitted_params,
            duration_lags,
            horizon,
            &forecast_result,
            &self.options.psi_guards,
        );
        self.forecast = Some(forecast_result);
        h_forecast
    }

    /// Compute classical or HAC-robust standard errors at the fitted ACD parameters.
    ///
    /// This routine builds standard errors for the **unconstrained parameter vector**
    /// θ = (θ₀, θ₁,…, θ_{p+q}) used by the optimizer. By default it returns
    /// *classical* (observed-information) SEs; with `HACOptions` it returns *robust*
    /// (sandwich/HAC) SEs that account for serial correlation and heteroskedasticity
    /// in the per-observation scores.
    ///
    /// ## Method
    /// - Let J(θ̂) denote the observed information (Hessian of the average
    ///   log-likelihood at the MLE θ̂). We obtain J(θ̂) by finite-differencing
    ///   the analytic gradient `self.grad(·, data)`; if configured, the Hessian is
    ///   regularized per `self.options.mle_opts`.
    /// - Classical SEs (no HAC):
    ///   Var(θ̂) ≈ J(θ̂)⁻¹
    ///   implemented via an eigendecomposition with a Moore–Penrose pseudoinverse
    ///   if J is singular or nearly singular; tiny eigenvalues are clipped at
    ///   `EIGEN_EPS`.
    /// - Robust/HAC SEs: form the covariance of the average score
    ///   S = Γ₀ + Σ (k=1..L) wₖ (Γₖ + Γₖᵀ)
    ///   using the kernel and bandwidth in `HACOptions` (with Newey–West small-sample
    ///   correction if enabled). Then the sandwich variance for component i is
    ///   Var(θ̂ᵢ) = wᵢᵀ S wᵢ, where wᵢ = J(θ̂)⁺ eᵢ.
    ///
    /// ## Arguments
    /// - `data`: The observed duration series used to compute per-observation scores.
    /// - `hac_opts`: `None` for classical SEs; `Some(opts)` for robust/HAC SEs.
    ///   - `opts.kernel`: Bartlett, Parzen, or Quadratic-Spectral (Bartlett ≈ Newey–West).
    ///   - `opts.bandwidth`: `None` = data-driven plug-in; otherwise clamped to n−1.
    ///   - `opts.center`: optionally demean scores before HAC (usually unnecessary at the MLE).
    ///   - `opts.small_sample_correction`: apply NW finite-sample scaling.
    ///
    /// ## Returns
    /// A length-(1 + p + q) vector of standard errors in θ-space (the optimizer’s
    /// unconstrained parameters).
    ///
    /// ## Notes
    /// - If you want SEs for model-space parameters (ω, α, β, slack), apply a delta-method
    ///   transform with the Jacobian of your softplus/softmax map.
    /// - The routine tolerates non–positive-definite Hessians via pseudoinversion; weak
    ///   identification will inflate SEs (by design).
    ///
    /// ## Errors
    /// - Propagates any runtime error from `self.grad` (captured during FD of the Hessian).
    /// - Returns an error if the model has not been fitted (no θ̂ available).
    /// - Robust branch may error if bandwidth plug-in fails and no fallback is allowed.
    pub fn standard_errors(
        &self, data: &ACDData, hac_opts: &Option<HACOptions>,
    ) -> ACDResult<Array1<f64>> {
        let runtime_error = RefCell::new(None);
        let theta_hat = extract_theta(self)?;
        let calc_grad = |th: &Array1<f64>| match self.grad(th, data) {
            Ok(g) => g,
            Err(e) => {
                *runtime_error.borrow_mut() = Some(e);
                Array1::from_elem(th.len(), f64::NAN)
            }
        };
        match hac_opts {
            Some(hac) => {
                let raw_scores = calculate_scores(&self, data)?;
                let theta_hat = extract_theta(self)?;
                let avg_scores = calculate_avg_scores(&hac, &raw_scores);
                Ok(calc_standard_errors(&calc_grad, theta_hat, Some(&avg_scores))?)
            }
            None => Ok(calc_standard_errors(&calc_grad, theta_hat, None)?),
        }
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
        let innov = &self.innovation;

        // Closure to accumulate gradient contributions from each observation
        let step_accumulate_grad = |row_triplet: ObsEntry<'_>, state: &mut Array1<f64>| {
            let (_, deriv_row, data_point, psi) = row_triplet;
            let innov_grad = innov.one_d_loglik_grad(data_point, psi)?;
            state.scaled_add(innov_grad, &deriv_row);
            Ok(())
        };

        let finish_map_grad = |workspace: &WorkSpace, state: &mut Array1<f64>| -> ACDResult<()> {
            state[0] *= safe_logistic(theta[0]);
            safe_softmax_deriv(
                &workspace.alpha,
                &workspace.beta,
                &mut state.slice_mut(s![1..]),
                p,
                q,
            );
            Ok(())
        };

        let mut state = Array1::zeros(1 + p + q);

        walk_observations(
            &self,
            data,
            theta.view(),
            &mut state,
            step_accumulate_grad,
            finish_map_grad,
        )?;
        Ok(state)
    }
}
