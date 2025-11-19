//! ACD model — fitting, forecasting, and inference for ACD(p, q).
//!
//! Purpose
//! -------
//! Provide a complete ACD(p, q) model API around durations, including
//! maximum-likelihood fitting, forecasting of ψ/τ, and classical or
//! HAC-robust standard errors. This module implements the [`LogLikelihood`]
//! trait so optimizers can work in unconstrained parameter space, while
//! internally mapping `θ` to model-space `(ω, α, β, slack)` using a
//! zero-copy [`WorkSpace`] and shared scratch buffers.
//!
//! Key behaviors
//! -------------
//! - Construct an [`ACDModel`] with model order (`shape`), innovation family
//!   (`innovation`), runtime options (`options`), and preallocated scratch
//!   buffers sized for a given in-sample length `n`.
//! - Expose log-likelihood and analytic gradient via the [`LogLikelihood`]
//!   trait (`value`, `check`, `grad`), using a zero-copy parameter mapping
//!   and allocation-free ψ recursion.
//! - Provide higher-level APIs:
//!   - [`ACDModel::fit`] for MLE in unconstrained θ-space,
//!   - [`ACDModel::forecast`] for ψ/τ forecasts from fitted parameters, and
//!   - [`ACDModel::standard_errors`] for classical or HAC-robust standard
//!     errors in θ-space.
//! - Cache optimizer outcomes, fitted model-space parameters, and forecast
//!   paths for downstream use (e.g., Python bindings, diagnostics).
//!
//! Invariants & assumptions
//! ------------------------
//! - Model order `shape` has been validated upstream via
//!   [`ACDShape::new(p, q, n)`], so that:
//!   - `p + q > 0` (no ACD(0, 0)),
//!   - `p < n` and `q < n`, where `n` is the in-sample length used to size
//!     scratch buffers.
//! - Duration data in [`ACDData`] are strictly positive, non-empty, and any
//!   burn-in index `t0` has been checked at construction time.
//! - Unconstrained parameter vectors `θ` passed into `value` / `grad` always
//!   satisfy `θ.len() == 1 + p + q` and have finite entries; this is enforced
//!   by [`LogLikelihood::check`] via [`validate_theta`].
//! - Parameter-space invariants (`ω > 0`, `αᵢ ≥ 0`, `βⱼ ≥ 0`,
//!   `∑α + ∑β < 1 − STATIONARITY_MARGIN`) are enforced by [`WorkSpace`] and
//!   `duration::core::validation` helpers; invalid user inputs are reported
//!   as [`ACDError`] / [`OptResult`] values rather than panics.
//!
//! Conventions
//! -----------
//! - Optimization is performed in unconstrained θ-space:
//!   - `ω = softplus(θ₀)`,
//!   - `(α, β, slack) = (1 − margin) · softmax(θ₁: )` with an implicit
//!     slack term added to enforce stationarity.
//! - Time indexing is 0-based in code, following the Engle–Russell ACD(p, q)
//!   recursion:
//!   `ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j}`.
//! - Scratch buffers ([`ACDScratch`]) and [`WorkSpace`] are reused across
//!   evaluations; ψ recursion and derivative recursion are allocation-free.
//! - Errors are surfaced as [`ACDResult`] or [`OptResult`]; panics indicate
//!   programming errors (e.g., inconsistent buffer sizes), not invalid user
//!   inputs.
//!
//! Downstream usage
//! ----------------
//! - Construct an [`ACDShape`] using [`ACDShape::new(p, q, n)`] with
//!   `n = data.len()` for the intended in-sample dataset.
//! - Build an [`ACDModel`] via [`ACDModel::new(shape, innovation, options, n)`],
//!   choosing an [`ACDInnovation`] family and [`ACDOptions`] for guards and MLE.
//! - Fit the model with [`ACDModel::fit(theta0, data)`], which:
//!   - runs an optimizer in θ-space using analytic value/gradient,
//!   - caches the resulting [`OptimOutcome`] in `results`, and
//!   - constructs [`ACDParams`] in model space (ω, α, β, slack, ψ-lags)
//!     into `fitted_params`.
//! - After fitting, call [`ACDModel::forecast`] to obtain ψ/τ forecasts and
//!   [`ACDModel::standard_errors`] to compute classical or HAC-robust SEs
//!   for θ̂. These outputs are intended to be wrapped by a higher-level
//!   Python API.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module cover:
//!   - `LogLikelihood` conformance for [`ACDModel`] (`check` length/finite
//!     validation, `value` vs. direct [`likelihood_driver`] calls, `grad`
//!     vs. finite-difference checks on small ACD examples).
//!   - [`ACDModel::fit`] behavior on synthetic data (successful MLE,
//!     populated `results` / `fitted_params`, θ/parameter roundtrips).
//!   - [`ACDModel::forecast`] behavior and error cases
//!     (e.g., [`ACDError::ModelNotFitted`]).
//!   - [`ACDModel::standard_errors`] shape and finiteness for classical and
//!     HAC configurations.
//! - Higher-level integration / Python tests should:
//!   - validate fitted parameters and forecasts against reference
//!     implementations on toy problems, and
//!   - exercise optimizer and HAC options end to end via the public API.
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
    inference::{HACOptions, calculate_avg_scores_cov, hessian::calc_standard_errors},
    optimization::{
        errors::OptResult,
        loglik_optimizer::{Grad, LogLikelihood, OptimOutcome, Theta, maximize},
        numerical_stability::transformations::{safe_logistic, safe_softmax_deriv},
    },
};
use ndarray::{Array1, s};
use std::cell::RefCell;

/// ACDModel — full ACD(p, q) model for durations.
///
/// Purpose
/// -------
/// Encapsulate the complete ACD(p, q) specification for a univariate duration
/// series, including model order, innovation family, run-time options, and the
/// scratch buffers / cached results needed for fitting, forecasting, and
/// inference.
///
/// Key behaviors
/// -------------
/// - Holds validated model metadata (`shape`, `innovation`, `options`) and
///   preallocated scratch buffers reused across likelihood/gradient evaluations.
/// - Caches the last optimization outcome (`results`), fitted model-space
///   parameters (`fitted_params`), and forecast paths (`forecast`).
/// - Implements [`LogLikelihood`] so it can be passed directly to
///   Argmin-based optimizers in unconstrained θ-space.
///
/// Parameters
/// ----------
/// Constructed via [`ACDModel::new`]:
/// - `shape`: [`ACDShape`]
///   Validated ACD(p, q) model order for the in-sample length `n`.
/// - `innovation`: [`ACDInnovation`]
///   Unit-mean innovation family (e.g., Exponential, Weibull).
/// - `options`: [`ACDOptions`]
///   Run-time options controlling guards, initialization, and MLE behavior.
/// - `n`: `usize`
///   In-sample length used to size internal scratch buffers.
///
/// Fields
/// ------
/// - `shape`: [`ACDShape`]
///   ACD(p, q) model order; treated as read-only after construction.
/// - `innovation`: [`ACDInnovation`]
///   Innovation distribution under a unit-mean parametrization.
/// - `options`: [`ACDOptions`]
///   Run-time configuration for likelihood evaluation and optimization.
/// - `scratch_bufs`: [`ACDScratch`]
///   Shared ψ / derivative buffers reused across evaluations to avoid
///   heap allocation in inner loops.
/// - `results`: `Option<OptimOutcome>`
///   Last optimizer outcome (θ̂, convergence info), populated by
///   [`ACDModel::fit`].
/// - `fitted_params`: `Option<ACDParams>`
///   Snapshot of model-space parameters (ω, α, β, slack, ψ-lags) at θ̂,
///   used for forecasting and reporting.
/// - `forecast`: `Option<ACDForecastResult>`
///   Last forecast path produced by [`ACDModel::forecast`].
///
/// Invariants
/// ----------
/// - When constructed via [`ACDModel::new`]:
///   - `scratch_bufs` is sized consistently with `shape` and `n`.
///   - `results`, `fitted_params`, and `forecast` are `None` until their
///     corresponding methods (`fit`, `forecast`) are called.
/// - Public APIs treat `shape` as already validated (via
///   [`ACDShape::new`]); this type does not re-check shape vs. data length.
///
/// Performance
/// -----------
/// - Designed so that repeated likelihood/gradient evaluations allocate
///   only for returned vectors (e.g., gradients), not for ψ recursion.
/// - Cloning an [`ACDModel`] clones scratch buffers and cached results; it is
///   intended for testing and small models rather than cheap duplication.
///
/// Notes
/// -----
/// - This struct is the natural backing type for a higher-level Python-facing
///   ACD model; the Python layer is expected to hold and orchestrate an
///   [`ACDModel`] instance internally.
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
    /// Forecasting results (populated after `forecast`).
    pub forecast: Option<ACDForecastResult>,
}

impl ACDModel {
    /// Construct a new [`ACDModel`] with preallocated scratch buffers.
    ///
    /// Parameters
    /// ----------
    /// - `shape`: [`ACDShape`]
    ///   Model order (p, q). Should be validated against the in-sample length
    ///   `n` (typically via [`ACDShape::new`]) before calling this constructor.
    /// - `innovation`: [`ACDInnovation`]
    ///   Unit-mean innovation family (e.g., Exponential, Weibull).
    /// - `options`: [`ACDOptions`]
    ///   Run-time options controlling guards, initialization, and optimizer
    ///   settings for MLE.
    /// - `n`: `usize`
    ///   In-sample length used to size internal ψ and derivative buffers.
    ///
    /// Returns
    /// -------
    /// [`ACDModel`]
    ///   A model instance with scratch space sized for `n`, `p = shape.p`,
    ///   and `q = shape.q`. All cached result fields (`results`,
    ///   `fitted_params`, `forecast`) are initialized to `None`.
    ///
    /// Errors
    /// ------
    /// - This constructor does not return errors. Consistency between `shape`
    ///   and `n` is assumed to have been enforced upstream (via
    ///   [`ACDShape::new`]).
    ///
    /// Panics
    /// ------
    /// - May panic if [`ACDScratch::new`] panics due to an internal
    ///   inconsistency between `shape` and `n`. This shouldn't happen in practice
    ///   since in the python wrappings `shape` is created using [`ACDShape::new`].
    ///
    /// Notes
    /// -----
    /// - Callers are expected to keep `n` consistent with the data used for
    ///   fitting and inference (typically `n = data.len()` for the training
    ///   sample).
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

    /// Fit ACD(p, q) by maximum likelihood and cache the results.
    ///
    /// Parameters
    /// ----------
    /// - `theta0`: `Array1<f64>`
    ///   Initial unconstrained parameter vector of length `1 + q + p`, where:
    ///   - `theta0[0]` is the log-parameter mapped to `ω` via softplus, and
    ///   - `theta0[1..]` are logits for the α/β/slack weights in the scaled
    ///     softmax mapping.
    ///   Ownership is moved into the optimizer; no internal clone is taken.
    /// - `data`: `&ACDData`
    ///   Observed duration series used to evaluate the log-likelihood.
    ///
    /// Returns
    /// -------
    /// `OptResult<()>`
    ///   - `Ok(())` if the optimizer terminates successfully and the model
    ///     caches both the optimizer outcome (`results`) and fitted parameters
    ///     (`fitted_params`).
    ///   - `Err(..)` if validation, likelihood evaluation, gradient evaluation,
    ///     or the optimizer itself reports a failure.
    ///
    /// Errors
    /// ------
    /// - `<optimizer / ACDError>`
    ///   Propagates any error from:
    ///   - [`validate_theta`] (shape / finiteness of `theta0`),
    ///   - [`LogLikelihood::value`] / [`LogLikelihood::grad`] (parameter
    ///     mapping, ψ recursion, innovation density / gradient), or
    ///   - the optimizer runner ([`maximize`]).
    ///
    /// Panics
    /// ------
    /// - Never panics on invalid user inputs; such conditions are surfaced as
    ///   `Err(..)` from the returned [`OptResult`].
    ///
    /// Notes
    /// -----
    /// - On success, this method:
    ///   - stores the optimizer outcome (including θ̂) in [`ACDModel::results`],
    ///   - recomputes ψ at θ̂ using a [`WorkSpace`] backed by
    ///     `self.scratch_bufs`,
    ///   - extracts the last `p` ψ-lags from the shared buffer, and
    ///   - constructs an [`ACDParams`] snapshot in model space
    ///     `(ω, α, β, slack, ψ-lags)` stored in [`ACDModel::fitted_params`].
    /// - The cached θ̂ and fitted parameters are intended for reuse in
    ///   forecasting, diagnostics, and Python bindings.
    pub fn fit(&mut self, theta0: Array1<f64>, data: &ACDData) -> OptResult<()> {
        let p = self.shape.p;
        self.results = Some(maximize(self, theta0, data, &self.options.mle_opts)?);
        let theta_hat = self.results.as_ref().unwrap().theta_hat.view();
        let mut workspace_alpha = self.scratch_bufs.alpha_buf.borrow_mut();
        let mut workspace_beta = self.scratch_bufs.beta_buf.borrow_mut();
        let mut workspace =
            WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &self.shape)?;
        workspace.update(theta_hat)?;
        compute_psi(&workspace, data, &self);
        let psi_lags = self.scratch_bufs.psi_buf.borrow();
        self.fitted_params = Some(ACDParams::from_theta(
            self.results.as_ref().unwrap().theta_hat.view(),
            &self.shape,
            psi_lags.slice(s![psi_lags.len() - p..]).to_owned(),
        )?);
        Ok(())
    }

    /// Forecast ψ (and τ, since E[τ_{t + i} | F_t] = E[ψ _{t + i} | F_t] under unit-mean innovations)
    /// `horizon` steps ahead from the fitted ACD(p, q) model.
    ///
    /// Parameters
    /// ----------
    /// - `horizon`: `usize`
    ///   Number of steps to forecast (H ≥ 1).
    /// - `data`: `&ACDData`
    ///   Observed duration series; the last `q = self.shape.q` values are used
    ///   as lagged durations in the recursion. Callers are expected to pass a
    ///   series whose length is at least the in-sample `n` used at model
    ///   construction.
    ///
    /// Returns
    /// -------
    /// `ACDResult<f64>`
    ///   - `Ok(ψ̂_{T+H})` — the H-step-ahead forecast of the conditional mean
    ///     duration.
    ///   - `Err(ACDError)` if the model has not been fitted or if forecast
    ///     recursion fails.
    ///
    /// Errors
    /// ------
    /// - [`ACDError::ModelNotFitted`]
    ///   Returned if called before [`ACDModel::fit`] has populated
    ///   [`ACDModel::fitted_params`].
    /// - Other [`ACDError`] variants
    ///   Propagated from [`forecast_recursion`] (e.g., lag length mismatch,
    ///   guard violations).
    ///
    /// Panics
    /// ------
    /// - Never panics on invalid user inputs; all such conditions are surfaced
    ///   as [`ACDError`] values.
    ///
    /// Notes
    /// -----
    /// - Duration forecasts equal ψ forecasts under the unit-mean innovation
    ///   parametrization: `τ̂ = ψ̂`.
    /// - The full forecast path is stored in [`ACDModel::forecast`] as an
    ///   [`ACDForecastResult`] and is available for inspection after calling
    ///   this method.
    pub fn forecast(&mut self, horizon: usize, data: &ACDData) -> ACDResult<f64> {
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
    /// This routine builds standard errors for the **unconstrained parameter
    /// vector** θ = (θ₀, θ₁,…, θ_{p+q}) used by the optimizer. By default it
    /// returns *classical* (observed-information) SEs; with [`HACOptions`] it
    /// returns *robust* (sandwich/HAC) SEs that account for serial correlation
    /// and heteroskedasticity in the per-observation scores.
    ///
    /// Parameters
    /// ----------
    /// - `data`: `&ACDData`
    ///   Observed duration series used to compute per-observation scores and
    ///   to re-evaluate the gradient as needed for Hessian approximation.
    /// - `hac_opts`: `Option<&HACOptions>`
    ///   - `None` for classical SEs based on the observed information.
    ///   - `Some(opts)` for robust/HAC SEs that use `opts.kernel`,
    ///     `opts.bandwidth`, `opts.center`, and optional small-sample
    ///     corrections.
    ///
    /// Returns
    /// -------
    /// `ACDResult<Array1<f64>>`
    ///   A length-(1 + p + q) vector of standard errors in θ-space (the
    ///   optimizer’s unconstrained parameters).
    ///
    /// Errors
    /// ------
    /// - [`ACDError::ModelNotFitted`]
    ///   Returned if no θ̂ is available (model has not been fitted).
    /// - Other [`ACDError`] variants
    ///   Propagated from:
    ///   - [`ACDModel::grad`] (during finite-differenced Hessian construction),
    ///   - [`calculate_scores`] / [`calculate_avg_scores`] (for HAC SEs),
    ///   - [`calc_standard_errors`] (e.g., Hessian regularization failures).
    ///
    /// Panics
    /// ------
    /// - Never panics on invalid user inputs; such conditions are surfaced as
    ///   [`ACDError`] values.
    ///
    /// Notes
    /// -----
    /// - Classical SEs (no HAC) use the observed information J(θ̂) obtained by
    ///   finite-differencing the analytic gradient `self.grad(·, data)`.
    ///   Inversion uses an eigendecomposition with a Moore–Penrose
    ///   pseudoinverse and eigenvalue clipping at `EIGEN_EPS` if needed.
    /// - Robust/HAC SEs form a covariance estimator S of the average score
    ///   using the kernel and bandwidth in [`HACOptions`], optionally with
    ///   Newey–West small-sample corrections, and then apply a sandwich
    ///   variance formula with J(θ̂)⁺.
    /// - If any gradient evaluation fails inside the Hessian or HAC machinery,
    ///   the internal callback records the last runtime error and this method
    ///   returns that [`ACDError`] instead of silently propagating NaNs.
    /// - To obtain SEs for model-space parameters (ω, α, β, slack), apply a
    ///   delta-method transform using the Jacobian of the softplus/softmax map.
    pub fn standard_errors(
        &self, data: &ACDData, hac_opts: Option<&HACOptions>,
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
        let se_result = match hac_opts {
            Some(hac) => {
                let raw_scores = calculate_scores(&self, data)?;
                let avg_scores = calculate_avg_scores_cov(&hac, &raw_scores);
                calc_standard_errors(&calc_grad, theta_hat, Some(&avg_scores))
            }
            None => calc_standard_errors(&calc_grad, theta_hat, None),
        };

        if let Some(e) = runtime_error.into_inner() {
            return Err(e.into());
        }

        Ok(se_result?)
    }
}

impl LogLikelihood for ACDModel {
    type Data = ACDData;

    /// Log-likelihood evaluation at parameter vector `θ`.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `&Theta`
    ///   Unconstrained optimizer vector of length `1 + p + q`.
    /// - `data`: `&ACDData`
    ///   Observed duration series.
    ///
    /// Returns
    /// -------
    /// `OptResult<f64>`
    ///   - `Ok(ℓ(θ))` — scalar log-likelihood at `θ`.
    ///   - `Err(..)` if parameter mapping, ψ recursion, or innovation
    ///     evaluation fails.
    ///
    /// Errors
    /// ------
    /// - Propagates any [`ACDError`] produced by:
    ///   - [`WorkSpace::new`] / [`WorkSpace::update`] (invalid θ, stationarity
    ///     violations, non-finite mapped parameters), or
    ///   - [`likelihood_driver`] (data validation, innovation density issues).
    ///
    /// Panics
    /// ------
    /// - Never panics on invalid user inputs; such conditions are reported as
    ///   `Err(..)` from the returned [`OptResult`].
    fn value(&self, theta: &Theta, data: &Self::Data) -> OptResult<f64> {
        let mut workspace_alpha = self.scratch_bufs.alpha_buf.borrow_mut();
        let mut workspace_beta = self.scratch_bufs.beta_buf.borrow_mut();
        let mut workspace =
            WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &self.shape)?;
        workspace.update(theta.view())?;
        Ok(likelihood_driver(&self, &workspace, &data)?)
    }

    /// Validate an unconstrained parameter vector `θ`.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `&Theta`
    ///   Candidate unconstrained optimizer vector.
    /// - `_data`: `&ACDData`
    ///   Unused; included to satisfy the [`LogLikelihood`] trait.
    ///
    /// Returns
    /// -------
    /// `OptResult<()>`
    ///   - `Ok(())` if `theta.len() == 1 + p + q` and all entries are finite.
    ///   - `Err(..)` otherwise.
    ///
    /// Errors
    /// ------
    /// - Propagates any [`ACDError`] from [`validate_theta`] when:
    ///   - the length of `theta` does not match `1 + p + q`, or
    ///   - any entry is non-finite.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    fn check(&self, theta: &Theta, _data: &Self::Data) -> OptResult<()> {
        validate_theta(theta.view(), self.shape.p, self.shape.q)?;
        Ok(())
    }

    /// Analytic gradient of log-likelihood w.r.t. unconstrained `θ`.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `&Theta`
    ///   Unconstrained optimizer vector of length `1 + p + q`.
    /// - `data`: `&ACDData`
    ///   Observed duration series.
    ///
    /// Returns
    /// -------
    /// `OptResult<Grad>`
    ///   - `Ok(∇ℓ(θ))` — gradient vector of length `1 + p + q`.
    ///   - `Err(..)` if parameter mapping, ψ recursion, derivative recursion,
    ///     or innovation gradient evaluation fails.
    ///
    /// Errors
    /// ------
    /// - Propagates any [`ACDError`] from:
    ///   - [`WorkSpace::new`] / [`WorkSpace::update`] (mapping θ → (ω, α, β, slack)),
    ///   - derivative recursion via [`walk_observations`], or
    ///   - [`ACDInnovation::one_d_loglik_grad`].
    ///
    /// Panics
    /// ------
    /// - Never panics on invalid user inputs; such conditions are surfaced as
    ///   `Err(..)` via the returned [`OptResult`].
    ///
    /// Notes
    /// -----
    /// - The gradient is computed in two stages:
    ///   1) accumulate ∂ℓ/∂ω, ∂ℓ/∂α, ∂ℓ/∂β via ψ-sensitivities, and
    ///   2) map to θ-space using:
    ///      - softplus’ derivative for θ₀ (`safe_logistic`), and
    ///      - a scaled softmax Jacobian (`safe_softmax_deriv`) for θ₁:.
    fn grad(&self, theta: &Theta, data: &Self::Data) -> OptResult<Grad> {
        let p = self.shape.p;
        let q = self.shape.q;
        let innov = &self.innovation;

        // Closure to accumulate gradient contributions from each observation
        let step_accumulate_grad = |entry: ObsEntry<'_>, state: &mut Array1<f64>| {
            let innov_grad = innov.one_d_loglik_grad(entry.data_point, entry.psi)?;
            state.scaled_add(innov_grad, &entry.deriv_row);
            Ok(())
        };

        let finish_map_grad = |workspace: &WorkSpace, state: &mut Array1<f64>| -> ACDResult<()> {
            state[0] *= safe_logistic(theta[0]);
            safe_softmax_deriv(&workspace.alpha, &workspace.beta, &mut state.slice_mut(s![1..]));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::core::{
        data::{ACDData, ACDMeta},
        guards::PsiGuards,
        init::Init,
        innovations::ACDInnovation,
        options::ACDOptions,
        shape::ACDShape,
        units::ACDUnit,
    };
    use crate::optimization::loglik_optimizer::traits::{LineSearcher, MLEOptions, Tolerances};
    use approx::assert_relative_eq;
    use ndarray::{Array1, array};

    /// Build a minimal ACD(1, 1) fixture: data, options, innovation, and model.
    fn make_acd11_fixture() -> (ACDModel, ACDData) {
        // Simple positive durations
        let durations = array![1.0, 0.9, 1.1, 1.2, 0.95, 1.05];

        let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
        let data = ACDData::new(durations.clone(), None, meta)
            .expect("ACDData::new should accept positive finite durations");

        let n = data.data.len();
        let shape =
            ACDShape::new(1, 1, n).expect("ACDShape::new(1, 1, n) should be valid for this n");

        // Estimation options: Init, MLEOptions, PsiGuards
        let init = Init::uncond_mean();

        let tols = Tolerances::new(Some(1e-6), None, Some(100))
            .expect("Tolerances::new should succeed with these arguments");
        let mle_opts = MLEOptions::new(tols, LineSearcher::MoreThuente, Some(5))
            .expect("MLEOptions::new should succeed with these arguments");

        let psi_guards =
            PsiGuards::new((1e-6, 1e6)).expect("PsiGuards::new should accept a wide positive band");

        let options = ACDOptions::new(init, mle_opts, psi_guards);

        // Use a concrete innovation family instead of a made-up Default
        let innovation = ACDInnovation::exponential();

        let model = ACDModel::new(shape, innovation, options, n);
        (model, data)
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDModel::forecast` returns `ACDError::ModelNotFitted` when
    // called before `fit` has been run.
    //
    // Given
    // -----
    // - A freshly constructed ACD(1, 1) model with valid data but no fit.
    //
    // Expect
    // ------
    // - `forecast(1, &data)` returns `Err(ACDError::ModelNotFitted)`.
    fn acdmodel_forecast_errors_when_model_not_fitted() {
        // Arrange
        let (mut model, data) = make_acd11_fixture();

        // Act
        let result = model.forecast(1, &data);

        // Assert
        match result {
            Err(ACDError::ModelNotFitted) => {}
            other => panic!("Expected ModelNotFitted, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDModel::standard_errors` returns `ACDError::ModelNotFitted`
    // when called before the model has been fitted.
    //
    // Given
    // -----
    // - A freshly constructed ACD(1, 1) model with valid data but no fit.
    //
    // Expect
    // ------
    // - `standard_errors(&data, None)` returns `Err(ACDError::ModelNotFitted)`.
    fn acdmodel_standard_errors_errors_when_model_not_fitted() {
        // Arrange
        let (model, data) = make_acd11_fixture();

        // Act
        let result = model.standard_errors(&data, None);

        // Assert
        match result {
            Err(ACDError::ModelNotFitted) => {}
            other => panic!("Expected ModelNotFitted, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDModel::fit` succeeds on a simple positive duration series
    // and populates `results` and `fitted_params`.
    //
    // Given
    // -----
    // - A valid ACD(1, 1) model and small positive duration series.
    // - A zero-initialized theta0 of length 1 + p + q.
    //
    // Expect
    // ------
    // - `fit(theta0, &data)` returns `Ok(())`.
    // - `model.results.is_some()` and `model.fitted_params.is_some()` afterwards.
    fn acdmodel_fit_populates_results_and_fitted_params() {
        // Arrange
        let (mut model, data) = make_acd11_fixture();
        let p = model.shape.p;
        let q = model.shape.q;
        let theta0: Array1<f64> = Array1::zeros(1 + p + q);

        // Act
        let fit_result = model.fit(theta0, &data);

        // Assert
        assert!(fit_result.is_ok(), "Expected fit to succeed on simple data, got {fit_result:?}");
        assert!(
            model.results.is_some(),
            "model.results should be populated after a successful fit"
        );
        assert!(
            model.fitted_params.is_some(),
            "model.fitted_params should be populated after a successful fit"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDModel::standard_errors` returns a finite SE vector of
    // length 1 + p + q after a successful fit (classical, non-HAC case).
    //
    // Given
    // -----
    // - A successfully fitted ACD(1, 1) model on a simple positive duration
    //   series.
    //
    // Expect
    // ------
    // - `standard_errors(&data, None)` returns `Ok(se)` where:
    //   - `se.len() == 1 + p + q`,
    //   - all entries of `se` are finite.
    fn acdmodel_standard_errors_returns_finite_vector_after_fit() {
        // Arrange
        let (mut model, data) = make_acd11_fixture();
        let p = model.shape.p;
        let q = model.shape.q;
        let theta0: Array1<f64> = Array1::zeros(1 + p + q);

        model.fit(theta0, &data).expect("fit should succeed before computing standard errors");

        // Act
        let se_result = model.standard_errors(&data, None);

        // Assert
        let se = se_result.expect("standard_errors should succeed in the classical case");
        assert_eq!(
            se.len(),
            1 + p + q,
            "SE vector length should match number of unconstrained parameters"
        );
        for (i, val) in se.iter().enumerate() {
            assert!(val.is_finite(), "SE[{i}] should be finite, got {val}",);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDModel::check` rejects a theta vector with the wrong length.
    //
    // Given
    // -----
    // - An ACD(1, 1) model (so θ should have length 1 + p + q = 3).
    // - A candidate θ of length 2.
    //
    // Expect
    // ------
    // - `check` returns an error (we only assert is_err, not the variant).
    fn loglik_check_rejects_incorrect_length_theta() {
        // Arrange
        let (model, data) = make_acd11_fixture();
        let bad_theta: Array1<f64> = Array1::zeros(2);

        // Act
        let result = model.check(&bad_theta, &data);

        // Assert
        assert!(result.is_err(), "Expected check to fail for wrong-length theta");
    }

    #[test]
    // Purpose
    // -------
    // Verify that the analytic gradient from `ACDModel::grad` matches a
    // central finite-difference approximation of `ACDModel::value` on a
    // small ACD(1, 1) example.
    //
    // Given
    // -----
    // - A valid ACD(1, 1) model and simple duration data.
    // - θ₀ = 0 vector of length 1 + p + q.
    //
    // Expect
    // ------
    // - `grad(θ₀)` is close to finite-difference ∂ℓ/∂θ at θ₀.
    fn loglik_grad_matches_finite_difference() {
        // Arrange
        let (model, data) = make_acd11_fixture();
        let p = model.shape.p;
        let q = model.shape.q;
        let theta0: Array1<f64> = Array1::zeros(1 + p + q);

        // Sanity: theta0 passes check
        model.check(&theta0, &data).expect("theta0 should be valid for ACD(1, 1)");

        let grad = model.grad(&theta0, &data).expect("grad should succeed at theta0");

        let h = 1e-6;
        let mut fd_grad = Array1::zeros(theta0.len());

        for i in 0..theta0.len() {
            let mut theta_plus = theta0.clone();
            let mut theta_minus = theta0.clone();
            theta_plus[i] += h;
            theta_minus[i] -= h;

            let val_plus =
                model.value(&theta_plus, &data).expect("value(theta_plus) should succeed");
            let val_minus =
                model.value(&theta_minus, &data).expect("value(theta_minus) should succeed");

            fd_grad[i] = (val_plus - val_minus) / (2.0 * h);
        }

        // Assert
        assert_relative_eq!(
            grad.as_slice().unwrap(),
            fd_grad.as_slice().unwrap(),
            max_relative = 1e-4
        );
    }
}
