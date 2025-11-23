//! loglik_optimizer::traits — log-likelihood traits and optimizer config.
//!
//! Purpose
//! -------
//! Expose the public traits and configuration types used to maximize
//! user-supplied log-likelihoods with L-BFGS. This module defines the
//! `LogLikelihood` contract, line-search choices, optimizer tolerances,
//! and the canonical optimization outcome returned by `maximize`.
//!
//! Key behaviors
//! -------------
//! - Provide the [`LogLikelihood`] trait that models implement to expose
//!   `value`, `check`, and optionally `grad` for a log-likelihood `ℓ(θ)`.
//! - Represent optimizer configuration via [`Tolerances`] and
//!   [`MLEOptions`], including line-search selection and L-BFGS memory.
//! - Normalize raw solver output into an [`OptimOutcome`] that reports
//!   θ̂, ℓ(θ̂), gradient norm, and iteration / evaluation counts.
//!
//! Invariants & assumptions
//! ------------------------
//! - The optimizer always *maximizes* a log-likelihood `ℓ(θ)` by
//!   minimizing the cost `c(θ) = -ℓ(θ)`.
//! - Any analytic gradient supplied by a [`LogLikelihood`] implementation
//!   must be the gradient of the log-likelihood (`∇ℓ(θ)`); the adapter
//!   is responsible for flipping the sign to match the cost.
//! - All configuration types ([`Tolerances`], [`MLEOptions`]) are validated
//!   on construction and are assumed to be internally consistent by the
//!   solver layer.
//!
//! Conventions
//! -----------
//! - [`Theta`] and [`Grad`] are type aliases for `ndarray` vectors and are
//!   treated as column vectors in the optimizer.
//! - Error handling is expressed in [`OptResult`] / [`OptError`]; user
//!   code implementing [`LogLikelihood`] should *not* panic on invalid
//!   inputs and should instead return descriptive errors.
//! - This module is pure Rust; there is no I/O, logging, or FFI at this
//!   layer. Optional logging is attached elsewhere via observers.
//!
//! Downstream usage
//! ----------------
//! - Model crates implement [`LogLikelihood`] for their types and pass
//!   those models, together with data and [`MLEOptions`], into the
//!   high-level `maximize` API.
//! - High-level APIs (e.g., Python bindings) construct [`Tolerances`] and
//!   [`MLEOptions`] from user-facing configuration, convert any
//!   human-readable line-search names into [`LineSearcher`], and surface
//!   [`OptimOutcome`] as the normalized optimization result.
//! - Other optimizer modules in this crate reuse [`Tolerances`] and
//!   [`MLEOptions`] to maintain consistent configuration semantics.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module:
//!   - verify [`LineSearcher::from_str`] parsing and error cases,
//!   - validate [`Tolerances::new`] invariants and error reporting,
//!   - exercise [`MLEOptions::new`] on valid and invalid `lbfgs_mem`
//!     values, and
//!   - check [`OptimOutcome::new`] behavior with valid / invalid θ̂ and
//!     log-likelihood values.
//! - Integration tests elsewhere exercise [`LogLikelihood`] implementations
//!   together with the `maximize` entrypoint on toy models (e.g.,
//!   simple quadratic log-likelihoods).
use crate::optimization::{
    errors::{OptError, OptResult},
    loglik_optimizer::{
        Cost, FnEvalMap, Grad, Theta,
        validation::{validate_theta_hat, validate_value, verify_tol_cost, verify_tol_grad},
    },
};
use argmin::core::TerminationStatus;
use argmin_math::ArgminL2Norm;
use std::str::FromStr;

/// LogLikelihood — user-provided log-likelihood for optimization.
///
/// Purpose
/// -------
/// Abstract the minimal interface a model must implement in order to be
/// optimized by the `maximize` API. Implementors expose evaluation,
/// validation, and (optionally) an analytic gradient for a scalar
/// log-likelihood `ℓ(θ)` defined over parameter vectors `Theta`.
///
/// Required methods
/// ----------------
/// - `value(&self, theta: &Theta, data: &Self::Data) -> OptResult<Cost>`
///   Evaluates the log-likelihood at `theta`. The return value is the
///   *log-likelihood* `ℓ(θ)`; the optimizer takes care of negating this
///   to form a cost.
/// - `check(&self, theta: &Theta, data: &Self::Data) -> OptResult<()>`
///   Performs cheap validation of `theta` and `data` (e.g., length and
///   finiteness checks) before optimization begins. Called once at the
///   start of `maximize`.
///
/// Optional methods
/// ----------------
/// - `grad(&self, theta: &Theta, data: &Self::Data) -> OptResult<Grad>`
///   Returns the analytic log-likelihood gradient `∇ℓ(θ)`. The default
///   implementation returns `OptError::GradientNotImplemented`, in which
///   case the optimizer falls back to finite differences.
///
/// Blanket impls / auto traits
/// ---------------------------
/// - Implementors are typically used behind a borrowed reference
///   (`&M`) where `M: LogLikelihood<Data = D>`. There are no blanket
///   implementations beyond the default `grad` method.
/// - The associated type `Data` is required to be `'static` to satisfy
///   the optimizer’s lifetime bounds.
///
/// Notes
/// -----
/// - Implementations should treat all invalid-user-input conditions
///   (e.g., non-finite data, violated model constraints) as recoverable
///   and return descriptive [`OptError`] variants rather than panicking.
/// - The gradient, when implemented, must be consistent with `value`:
///   it should represent `∇ℓ(θ)`, not `∇c(θ)`. The adapter that builds
///   the cost function takes care of sign conventions.
pub trait LogLikelihood {
    type Data: 'static;

    // Required methods
    fn value(&self, theta: &Theta, data: &Self::Data) -> OptResult<Cost>;
    fn check(&self, theta: &Theta, data: &Self::Data) -> OptResult<()>;

    // Optional methods
    fn grad(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<Grad> {
        Err(OptError::GradientNotImplemented)
    }
}

/// LineSearcher — choice of line search for L-BFGS.
///
/// Purpose
/// -------
/// Capture the line-search algorithm used inside the L-BFGS optimizer
/// when stepping along search directions. This enum is configured once
/// via [`MLEOptions`] and passed through to the underlying solver.
///
/// Variants
/// --------
/// - `MoreThuente`
///   Use the More–Thuente line search, which enforces strong Wolfe
///   conditions and is widely used in quasi-Newton methods.
/// - `HagerZhang`
///   Use the Hager–Zhang line search, which can offer improved
///   globalization properties on some problems.
///
/// Invariants
/// ----------
/// - This enum is purely a configuration knob; it does not perform any
///   logic on its own. The underlying solver is responsible for
///   implementing the chosen algorithm correctly.
///
/// Notes
/// -----
/// - [`LineSearcher`] implements [`FromStr`] with case-insensitive
///   parsing of `"MoreThuente"` and `"HagerZhang"`. Unknown names are
///   mapped to `OptError::InvalidLineSearch` so that user-facing layers
///   can surface clear error messages.
/// - Downstream code pattern-matches on this enum once when
///   building the argmin solver state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearcher {
    MoreThuente,
    HagerZhang,
}

impl FromStr for LineSearcher {
    type Err = OptError;

    /// Parse a line-search choice from a string (case-insensitive).
    ///
    /// Accepts:
    /// - `"MoreThuente"`
    /// - `"HagerZhang"`
    /// - Any case variant (e.g., `"morethuente"`, `"HAGERZHANG"`).
    ///
    /// Any other value returns `OptError::InvalidLineSearch` with a helpful message.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "morethuente" => Ok(LineSearcher::MoreThuente),
            "hagerzhang" => Ok(LineSearcher::HagerZhang),
            _ => Err(OptError::InvalidLineSearch {
                name: s.to_string(),
                reason: "Valid options are case insensitive 'MoreThuente' or 'HagerZhang'.",
            }),
        }
    }
}

/// MLEOptions — optimizer configuration for log-likelihood maximization.
///
/// Purpose
/// -------
/// Bundle the numerical tolerances, line-search choice, and optional
/// L-BFGS memory setting used by the `maximize` API. This type provides
/// a single, validated configuration object that can be constructed from
/// user-facing options and passed into the optimizer.
///
/// Key behaviors
/// -------------
/// - Carries [`Tolerances`] that control termination criteria and maximum
///   iterations.
/// - Encodes which line-search algorithm to use via [`LineSearcher`].
/// - Optionally specifies L-BFGS memory (`lbfgs_mem`); when `None`, the
///   solver’s default (e.g., 7) is used.
///
/// Parameters
/// ----------
/// Constructed via [`MLEOptions::new`]:
/// - `tols`: [`Tolerances`]
///   Validated numerical tolerances and iteration limits.
/// - `line_searcher`: [`LineSearcher`]
///   Choice of line search for L-BFGS steps.
/// - `lbfgs_mem`: `Option<usize>`
///   Optional number of correction pairs to store in L-BFGS. `None` means
///   “use the solver default”; `Some(m)` requires `m > 0`.
///
/// Fields
/// ------
/// - `tols`: [`Tolerances`]
///   Per-run termination tolerances and iteration cap.
/// - `line_searcher`: [`LineSearcher`]
///   Line-search algorithm used by the optimizer.
/// - `lbfgs_mem`: `Option<usize>`
///   Optional memory parameter forwarded to the L-BFGS implementation.
///   When `Some(m)`, `m` must be greater than zero.
///
/// Invariants
/// ----------
/// - `lbfgs_mem`, if present, is strictly positive; zero is rejected by
///   [`MLEOptions::new`] with `OptError::InvalidLBFGSMem`.
/// - `tols` is always a validated [`Tolerances`] instance; any invalid
///   tolerances are rejected at construction time.
///
/// Performance
/// -----------
/// - `MLEOptions` is cheap to copy and pass by value; it contains only a
///   small struct, an enum, and an `Option<usize>`.
///
/// Notes
/// -----
/// - A `Default` implementation is provided and corresponds to a typical
///   quasi-Newton configuration (`tol_grad = 1e-6`, `max_iter = 300`,
///   More–Thuente line search, solver-default L-BFGS memory).
#[derive(Debug, Clone, PartialEq)]
pub struct MLEOptions {
    pub tols: Tolerances,
    pub line_searcher: LineSearcher,
    pub lbfgs_mem: Option<usize>,
}

impl MLEOptions {
    /// Construct validated optimizer options.
    ///
    /// Parameters
    /// ----------
    /// - `tols`: [`Tolerances`]
    ///   Numerical tolerances and iteration limits. Must already have been
    ///   constructed via [`Tolerances::new`] so that it is internally
    ///   consistent.
    /// - `line_searcher`: [`LineSearcher`]
    ///   Line-search variant to use in L-BFGS steps.
    /// - `lbfgs_mem`: `Option<usize>`
    ///   Optional number of correction pairs to keep in L-BFGS. When
    ///   `Some(m)`, `m` must be strictly greater than zero; `None` delegates
    ///   to the solver’s default.
    ///
    /// Returns
    /// -------
    /// `OptResult<MLEOptions>`
    ///   - `Ok(opts)` if all arguments are valid and `lbfgs_mem` satisfies
    ///     its invariants.
    ///   - `Err(e)` if `lbfgs_mem` is invalid.
    ///
    /// Errors
    /// ------
    /// - `OptError::InvalidLBFGSMem`
    ///   Returned when `lbfgs_mem` is `Some(0)` or otherwise violates the
    ///   L-BFGS memory constraint.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - No `unsafe` code is used.
    ///
    /// Notes
    /// -----
    /// - This constructor does *not* re-validate numerical tolerances; any
    ///   checks on `tol_grad`, `tol_cost`, or `max_iter` must be performed
    ///   when creating the [`Tolerances`] object.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::optimization::loglik_optimizer::traits::{
    /// #     LineSearcher, MLEOptions, Tolerances
    /// # };
    /// let tols = Tolerances::new(Some(1e-6), None, Some(200)).unwrap();
    /// let opts = MLEOptions::new(tols, LineSearcher::MoreThuente, Some(7)).unwrap();
    /// assert_eq!(opts.lbfgs_mem, Some(7));
    /// ```
    pub fn new(
        tols: Tolerances, line_searcher: LineSearcher, lbfgs_mem: Option<usize>,
    ) -> OptResult<Self> {
        if let Some(m) = lbfgs_mem
            && m == 0
        {
            return Err(OptError::InvalidLBFGSMem {
                mem: m,
                reason: "L-BFGS memory must be greater than zero.",
            });
        }
        Ok(Self { tols, line_searcher, lbfgs_mem })
    }
}

impl Default for MLEOptions {
    /// Default optimizer configuration.
    ///
    /// Parameters
    /// ----------
    /// - None  
    ///   This is a zero-argument convenience constructor.
    ///
    /// Returns
    /// -------
    /// `MLEOptions`
    ///   An options struct with:
    ///   - `tols = Tolerances::new(Some(1e-6), None, Some(300))`
    ///   - `line_searcher = LineSearcher::MoreThuente`
    ///   - `lbfgs_mem = None` (use solver default).
    ///
    /// Errors
    /// ------
    /// - Never returns an error; the default tolerances are hard-coded to a
    ///   valid configuration.
    ///
    /// Panics
    /// ------
    /// - Panics only if the hard-coded default tolerances become invalid due
    ///   to future changes in [`Tolerances::new`]. Under current invariants,
    ///   this cannot happen.
    ///
    /// Safety
    /// ------
    /// - No `unsafe` code is used.
    ///
    /// Notes
    /// -----
    /// - The default configuration is intended as a sensible starting point
    ///   for quasi-Newton MLE problems.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::optimization::loglik_optimizer::traits::MLEOptions;
    /// let opts = MLEOptions::default();
    /// assert!(opts.tols.max_iter.is_some());
    /// ```
    fn default() -> Self {
        Self {
            tols: Tolerances::new(Some(1e-6), None, Some(300))
                .expect("hard-coded defaults are valid"),
            line_searcher: LineSearcher::MoreThuente,
            lbfgs_mem: None,
        }
    }
}

/// Tolerances — numerical stopping criteria and iteration limits.
///
/// Purpose
/// -------
/// Represent the termination conditions for the optimizer in a single
/// struct. Callers specify gradient and/or cost tolerances and an
/// iteration cap; `Tolerances::new` validates that at least one stopping
/// mechanism is enabled and that all values are sensible.
///
/// Key behaviors
/// -------------
/// - Encapsulate optional gradient and cost tolerances (`tol_grad`,
///   `tol_cost`) and an optional iteration limit (`max_iter`).
/// - Enforce that at least one of the three termination criteria is
///   provided.
/// - Validate the finiteness and positivity of any supplied tolerances
///   and iteration counts.
///
/// Parameters
/// ----------
/// Constructed via [`Tolerances::new`]:
/// - `tol_grad`: `Option<f64>`
///   Optional gradient-norm tolerance. When `Some(τ)`, requires `τ` to
///   be finite and strictly positive.
/// - `tol_cost`: `Option<f64>`
///   Optional cost-delta tolerance. When `Some(τ)`, requires `τ` to be
///   finite and strictly positive.
/// - `max_iter`: `Option<usize>`
///   Optional maximum iteration count. When `Some(m)`, requires
///   `m > 0`.
///
/// Fields
/// ------
/// - `tol_grad`: `Option<f64>`
///   Terminate when the gradient norm falls below this threshold, if
///   provided.
/// - `tol_cost`: `Option<f64>`
///   Terminate when the change in cost falls below this threshold, if
///   provided.
/// - `max_iter`: `Option<usize>`
///   Hard cap on the number of iterations, if provided.
///
/// Invariants
/// ----------
/// - At least one of `tol_grad`, `tol_cost`, or `max_iter` is `Some(..)`;
///   the all-`None` configuration is rejected.
/// - Any provided tolerances are finite and strictly positive.
/// - Any provided `max_iter` is strictly greater than zero.
///
/// Performance
/// -----------
/// - This is a small, `Copy`-friendly struct; cloning is trivial.
///
/// Notes
/// -----
/// - `Tolerances` is intended to be an internal optimizer primitive; most
///   users interact with it via higher-level configuration builders or
///   via [`MLEOptions`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerances {
    pub tol_grad: Option<f64>,
    pub tol_cost: Option<f64>,
    pub max_iter: Option<usize>,
}

impl Tolerances {
    /// Construct and validate optimizer tolerances.
    ///
    /// Parameters
    /// ----------
    /// - `tol_grad`: `Option<f64>`
    ///   Optional gradient-norm tolerance. When `Some(τ)`, `τ` must be
    ///   finite and strictly positive.
    /// - `tol_cost`: `Option<f64>`
    ///   Optional cost-delta tolerance. When `Some(τ)`, `τ` must be finite
    ///   and strictly positive.
    /// - `max_iter`: `Option<usize>`
    ///   Optional maximum iteration count. When `Some(m)`, `m` must be
    ///   strictly greater than zero.
    ///
    /// Returns
    /// -------
    /// `OptResult<Tolerances>`
    ///   - `Ok(tols)` if at least one stopping condition is provided and all
    ///     supplied values satisfy their constraints.
    ///   - `Err(e)` if no tolerances are provided or any value is invalid.
    ///
    /// Errors
    /// ------
    /// - `OptError::NoTolerancesProvided`
    ///   Returned when all of `tol_grad`, `tol_cost`, and `max_iter` are
    ///   `None`.
    /// - `OptError::InvalidTolGrad`
    ///   Returned when `tol_grad` is `Some` but non-finite or non-positive.
    /// - `OptError::InvalidTolCost`
    ///   Returned when `tol_cost` is `Some` but non-finite or non-positive.
    /// - `OptError::InvalidMaxIter`
    ///   Returned when `max_iter` is `Some(0)`.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - No `unsafe` code is used.
    ///
    /// Notes
    /// -----
    /// - This constructor centralizes all tolerance validation so that
    ///   downstream code can rely on [`Tolerances`] being internally
    ///   consistent.
    /// - Callers are free to leave some fields as `None` so long as at least
    ///   one termination mechanism is enabled.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::optimization::loglik_optimizer::traits::Tolerances;
    /// let tols = Tolerances::new(Some(1e-6), None, Some(100)).unwrap();
    /// assert_eq!(tols.max_iter, Some(100));
    /// ```
    pub fn new(
        tol_grad: Option<f64>, tol_cost: Option<f64>, max_iter: Option<usize>,
    ) -> OptResult<Self> {
        if tol_grad.is_none() && tol_cost.is_none() && max_iter.is_none() {
            return Err(OptError::NoTolerancesProvided);
        }
        verify_tol_cost(tol_cost)?;
        verify_tol_grad(tol_grad)?;
        if let Some(max_iter) = max_iter
            && max_iter == 0
        {
            return Err(OptError::InvalidMaxIter {
                max_iter,
                reason: "Maximum iterations must be greater than zero.",
            });
        }
        Ok(Self { tol_grad, tol_cost, max_iter })
    }
}

/// OptimOutcome — normalized result of log-likelihood maximization.
///
/// Purpose
/// -------
/// Capture the final parameter vector, log-likelihood value, and basic
/// diagnostics returned by the `maximize` API. This struct normalizes
/// argmin’s raw solver state into a crate-specific result type that is
/// easy to surface in higher-level APIs.
///
/// Key behaviors
/// -------------
/// - Store the best parameter vector θ̂ and corresponding log-likelihood
///   value `ℓ(θ̂)`.
/// - Record convergence status, termination message, iteration count, and
///   function-evaluation counters.
/// - Optionally report the final gradient norm, if a gradient was
///   available from the solver.
///
/// Parameters
/// ----------
/// Constructed via [`OptimOutcome::new`]:
/// - `theta_hat_opt`: `Option<Theta>`
///   Candidate best parameter vector. Must be `Some` with all finite
///   entries; otherwise an error is returned.
/// - `value`: `f64`
///   Log-likelihood value at θ̂. Must be finite.
/// - `converged`: `TerminationStatus`
///   Argmin termination status; mapped into a boolean convergence flag
///   and a human-readable status string.
/// - `iterations`: `u64`
///   Number of iterations performed by the optimizer.
/// - `fn_evals`: [`FnEvalMap`]
///   Map of evaluation counters (cost, gradient, Hessian, etc.).
/// - `grad`: `Option<Grad>`
///   Optional gradient at θ̂, used to compute the final gradient norm.
///
/// Fields
/// ------
/// - `theta_hat`: [`Theta`]
///   Best parameter vector found by the solver.
/// - `value`: `f64`
///   Best log-likelihood value `ℓ(θ̂)` (not the cost).
/// - `term_status`: `bool`
///   `true` if the termination status is not `NotTerminated`.
/// - `status`: `String`
///   Human-readable termination status derived from `TerminationStatus`.
/// - `iterations`: `usize`
///   Number of iterations executed (downcast from `u64`).
/// - `fn_evals`: [`FnEvalMap`]
///   Evaluation counters as reported by the underlying solver.
/// - `grad_norm`: `Option<f64>`
///   L2 norm of the final gradient, if available.
///
/// Invariants
/// ----------
/// - `theta_hat` is always present and finite; invalid or missing vectors
///   are rejected by [`OptimOutcome::new`].
/// - `value` is always finite; NaN or infinite values are rejected.
/// - `iterations` is the integer number of steps actually taken; it may
///   be zero if the solver terminated immediately.
///
/// Performance
/// -----------
/// - Cloning an [`OptimOutcome`] is cheap; it consists of a parameter
///   vector and a handful of scalars and small containers.
///
/// Notes
/// -----
/// - This type is the primary bridge between the optimization core and
///   user-facing layers (e.g., Python bindings). It is intentionally
///   verbose in order to surface diagnostics without requiring access to
///   argmin’s internal types.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimOutcome {
    pub theta_hat: Theta,
    pub value: f64,
    pub term_status: bool,
    pub status: String,
    pub iterations: usize,
    pub fn_evals: FnEvalMap,
    pub grad_norm: Option<f64>,
}

impl OptimOutcome {
    /// Construct a validated optimization outcome from raw solver state.
    ///
    /// Parameters
    /// ----------
    /// - `theta_hat_opt`: `Option<Theta>`
    ///   Candidate best parameter vector. Must be `Some` with all finite
    ///   components; otherwise an error is returned.
    /// - `value`: `f64`
    ///   Log-likelihood value at the candidate θ̂. Must be finite; non-finite
    ///   values are reported as `OptError::NonFiniteCost`.
    /// - `term_status`: `TerminationStatus`
    ///   Argmin termination status; mapped into a boolean convergence flag
    ///   and a human-readable status string.
    /// - `iterations`: `u64`
    ///   Number of iterations performed by the optimizer.
    /// - `fn_evals`: [`FnEvalMap`]
    ///   Raw evaluation counters (cost, gradient, etc.) from the solver.
    /// - `grad`: `Option<Grad>`
    ///   Gradient at θ̂, if available; used to compute the final gradient
    ///   norm reported in `grad_norm`.
    ///
    /// Returns
    /// -------
    /// `OptResult<OptimOutcome>`
    ///   - `Ok(outcome)` if `theta_hat_opt` is present and finite and `value`
    ///     is finite.
    ///   - `Err(e)` if either θ̂ or `value` fail validation.
    ///
    /// Errors
    /// ------
    /// - Propagates any `OptError` from:
    ///   - `validate_theta_hat` when `theta_hat_opt` is `None` or contains
    ///     non-finite entries, and
    ///   - `validate_value` when `value` is non-finite.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - No `unsafe` code is used.
    ///
    /// Notes
    /// -----
    /// - The boolean `term_status` flag is set to `false` only when the
    ///   termination status is `NotTerminated`; all other statuses are
    ///   treated as “converged” and are reported verbatim in `status`.
    /// - `grad_norm` is computed using the L2 norm provided by
    ///   `ArgminL2Norm`; when `grad` is `None`, `grad_norm` is left as
    ///   `None`.
    /// - `iterations` is downcast from `u64` to `usize`; extremely large
    ///   iteration counts are not expected in typical use.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use argmin::core::TerminationStatus;
    /// # use rust_timeseries::optimization::loglik_optimizer::traits::{
    /// #     OptimOutcome, Theta
    /// # };
    /// # use std::collections::HashMap;
    /// let theta_hat = Theta::from(vec![0.0_f64, 1.0]);
    /// let fn_evals = HashMap::new();
    /// let outcome = OptimOutcome::new(
    ///     Some(theta_hat.clone()),
    ///     -1.23,
    ///     TerminationStatus::Terminated,
    ///     10,
    ///     fn_evals,
    ///     None,
    /// ).unwrap();
    /// assert!(outcome.term_status);
    /// assert_eq!(outcome.theta_hat, theta_hat);
    /// ```
    pub fn new(
        theta_hat_opt: Option<Theta>, value: f64, converged: TerminationStatus, iterations: u64,
        fn_evals: FnEvalMap, grad: Option<Grad>,
    ) -> OptResult<Self> {
        let theta_hat = validate_theta_hat(theta_hat_opt)?;
        validate_value(value)?;
        let status: String;
        let term_status = match converged {
            TerminationStatus::NotTerminated => {
                status = "Not terminated".to_string();
                false
            }
            TerminationStatus::Terminated(reason) => {
                status = format!("{reason:?}");
                true
            }
        };
        let iterations = iterations as usize;
        let grad_norm = grad.map(|g| g.l2_norm());
        Ok(Self { theta_hat, value, term_status, status, iterations, fn_evals, grad_norm })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin::core::{TerminationReason, TerminationStatus};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Parsing and error handling for `LineSearcher::from_str`.
    // - Validation invariants enforced by `Tolerances::new`.
    // - Construction and defaults for `MLEOptions`.
    // - Validation and normalization behavior of `OptimOutcome::new`.
    //
    // They intentionally DO NOT cover:
    // - End-to-end optimizer behavior via `maximize` (handled in higher-level
    //   integration tests).
    // - Concrete `LogLikelihood` implementations for specific models
    //   (covered in model-specific test modules).
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Ensure that `LineSearcher::from_str` parses known variants in a
    // case-insensitive way.
    //
    // Given
    // -----
    // - Several string representations of the supported line-search variants
    //   in different casings.
    //
    // Expect
    // ------
    // - All known names parse successfully to the correct `LineSearcher`
    //   variant.
    fn linesearcher_from_str_parses_known_variants_case_insensitive() {
        // Arrange
        let more_variants = ["MoreThuente", "morethuente", "MORETHUENTE"];
        let hager_variants = ["HagerZhang", "hagerzhang", "HAGERZHANG"];

        // Act / Assert
        for s in more_variants {
            let ls = LineSearcher::from_str(s).expect("MoreThuente should parse");
            assert_eq!(ls, LineSearcher::MoreThuente);
        }

        for s in hager_variants {
            let ls = LineSearcher::from_str(s).expect("HagerZhang should parse");
            assert_eq!(ls, LineSearcher::HagerZhang);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `LineSearcher::from_str` returns an error for unknown
    // strings.
    //
    // Given
    // -----
    // - A string that does not correspond to any supported line-search
    //   variant.
    //
    // Expect
    // ------
    // - `from_str` returns `Err(OptError::InvalidLineSearch { .. })`.
    fn linesearcher_from_str_rejects_unknown_value() {
        // Arrange
        let bad = "UnknownSearch";

        // Act
        let result = LineSearcher::from_str(bad);

        // Assert
        let err = result.expect_err("Unknown line search should be rejected");
        match err {
            OptError::InvalidLineSearch { name, .. } => {
                assert_eq!(name, bad);
            }
            other => panic!("Expected InvalidLineSearch, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `Tolerances::new` rejects the configuration where all
    // three stopping criteria are `None`.
    //
    // Given
    // -----
    // - `tol_grad = None`, `tol_cost = None`, `max_iter = None`.
    //
    // Expect
    // ------
    // - `Tolerances::new` returns `Err(OptError::NoTolerancesProvided)`.
    fn tolerances_new_errors_when_all_none() {
        // Arrange
        let tol_grad = None;
        let tol_cost = None;
        let max_iter = None;

        // Act
        let result = Tolerances::new(tol_grad, tol_cost, max_iter);

        // Assert
        let err = result.expect_err("All-None tolerances should be rejected");
        match err {
            OptError::NoTolerancesProvided => {}
            other => panic!("Expected NoTolerancesProvided, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Tolerances::new` rejects an invalid gradient tolerance.
    //
    // Given
    // -----
    // - `tol_grad = Some(-1.0)` (non-positive), `tol_cost = None`,
    //   `max_iter = Some(10)`.
    //
    // Expect
    // ------
    // - `Tolerances::new` returns `Err(OptError::InvalidTolGrad { .. })`.
    fn tolerances_new_errors_on_invalid_tol_grad() {
        // Arrange
        let tol_grad = Some(-1.0);
        let tol_cost = None;
        let max_iter = Some(10);

        // Act
        let result = Tolerances::new(tol_grad, tol_cost, max_iter);

        // Assert
        let err = result.expect_err("Negative tol_grad should be rejected");
        match err {
            OptError::InvalidTolGrad { .. } => {}
            other => panic!("Expected InvalidTolGrad, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Tolerances::new` rejects an invalid cost tolerance.
    //
    // Given
    // -----
    // - `tol_grad = None`, `tol_cost = Some(0.0)` (non-positive),
    //   `max_iter = Some(10)`.
    //
    // Expect
    // ------
    // - `Tolerances::new` returns `Err(OptError::InvalidTolCost { .. })`.
    fn tolerances_new_errors_on_invalid_tol_cost() {
        // Arrange
        let tol_grad = None;
        let tol_cost = Some(0.0);
        let max_iter = Some(10);

        // Act
        let result = Tolerances::new(tol_grad, tol_cost, max_iter);

        // Assert
        let err = result.expect_err("Non-positive tol_cost should be rejected");
        match err {
            OptError::InvalidTolCost { .. } => {}
            other => panic!("Expected InvalidTolCost, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Tolerances::new` rejects a zero `max_iter`.
    //
    // Given
    // -----
    // - `tol_grad = Some(1e-6)`, `tol_cost = None`, `max_iter = Some(0)`.
    //
    // Expect
    // ------
    // - `Tolerances::new` returns `Err(OptError::InvalidMaxIter { .. })`.
    fn tolerances_new_errors_on_zero_max_iter() {
        // Arrange
        let tol_grad = Some(1e-6);
        let tol_cost = None;
        let max_iter = Some(0);

        // Act
        let result = Tolerances::new(tol_grad, tol_cost, max_iter);

        // Assert
        let err = result.expect_err("max_iter = 0 should be rejected");
        match err {
            OptError::InvalidMaxIter { max_iter, .. } => {
                assert_eq!(max_iter, 0);
            }
            other => panic!("Expected InvalidMaxIter, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `Tolerances::new` accepts a valid combination of
    // tolerances and max_iter.
    //
    // Given
    // -----
    // - `tol_grad = Some(1e-6)`, `tol_cost = None`, `max_iter = Some(100)`.
    //
    // Expect
    // ------
    // - `Tolerances::new` returns `Ok(tols)` with the same fields echoed
    //   back.
    fn tolerances_new_accepts_valid_configuration() {
        // Arrange
        let tol_grad = Some(1e-6);
        let tol_cost = None;
        let max_iter = Some(100);

        // Act
        let result = Tolerances::new(tol_grad, tol_cost, max_iter);

        // Assert
        let tols = result.expect("Valid tolerances should construct");
        assert_eq!(tols.tol_grad, tol_grad);
        assert_eq!(tols.tol_cost, tol_cost);
        assert_eq!(tols.max_iter, max_iter);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `MLEOptions::new` rejects an invalid L-BFGS memory
    // value of zero.
    //
    // Given
    // -----
    // - A valid `Tolerances` value.
    // - `line_searcher = LineSearcher::MoreThuente`.
    // - `lbfgs_mem = Some(0)`.
    //
    // Expect
    // ------
    // - `MLEOptions::new` returns `Err(OptError::InvalidLBFGSMem { .. })`.
    fn mleoptions_new_errors_on_zero_lbfgs_mem() {
        // Arrange
        let tols =
            Tolerances::new(Some(1e-6), None, Some(10)).expect("Tolerances::new should succeed");
        let line_searcher = LineSearcher::MoreThuente;
        let lbfgs_mem = Some(0);

        // Act
        let result = MLEOptions::new(tols, line_searcher, lbfgs_mem);

        // Assert
        let err = result.expect_err("lbfgs_mem = 0 should be rejected");
        match err {
            OptError::InvalidLBFGSMem { mem, .. } => {
                assert_eq!(mem, 0);
            }
            other => panic!("Expected InvalidLBFGSMem, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `MLEOptions::new` accepts a valid configuration and
    // preserves its arguments.
    //
    // Given
    // -----
    // - Valid `Tolerances`.
    // - `line_searcher = LineSearcher::HagerZhang`.
    // - `lbfgs_mem = Some(7)`.
    //
    // Expect
    // ------
    // - `MLEOptions::new` returns `Ok(opts)` with matching fields.
    fn mleoptions_new_accepts_valid_configuration() {
        // Arrange
        let tols =
            Tolerances::new(Some(1e-6), None, Some(50)).expect("Tolerances::new should succeed");
        let line_searcher = LineSearcher::HagerZhang;
        let lbfgs_mem = Some(7);

        // Act
        let result = MLEOptions::new(tols, line_searcher, lbfgs_mem);

        // Assert
        let opts = result.expect("Valid MLEOptions should construct");
        assert_eq!(opts.tols, tols);
        assert_eq!(opts.line_searcher, line_searcher);
        assert_eq!(opts.lbfgs_mem, lbfgs_mem);
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `MLEOptions::default` matches the documented default
    // configuration.
    //
    // Given
    // -----
    // - The `MLEOptions::default()` value.
    //
    // Expect
    // ------
    // - `tols.tol_grad == Some(1e-6)`.
    // - `tols.tol_cost == None`.
    // - `tols.max_iter == Some(300)`.
    // - `line_searcher == LineSearcher::MoreThuente`.
    // - `lbfgs_mem == None`.
    fn mleoptions_default_matches_documented_values() {
        // Arrange
        let opts = MLEOptions::default();

        // Assert
        assert_eq!(opts.tols.tol_grad, Some(1e-6));
        assert_eq!(opts.tols.tol_cost, None);
        assert_eq!(opts.tols.max_iter, Some(300));
        assert_eq!(opts.line_searcher, LineSearcher::MoreThuente);
        assert_eq!(opts.lbfgs_mem, None);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `OptimOutcome::new` rejects a missing `theta_hat_opt`.
    //
    // Given
    // -----
    // - `theta_hat_opt = None`.
    // - A finite log-likelihood value.
    //
    // Expect
    // ------
    // - `OptimOutcome::new` returns an error via `validate_theta_hat`.
    fn optimoutcome_new_errors_when_theta_hat_missing() {
        // Arrange
        let theta_hat_opt = None;
        let value = -1.23;
        let term_status = TerminationStatus::NotTerminated;
        let iterations = 0_u64;
        let fn_evals = FnEvalMap::default();
        let grad = None;

        // Act
        let result =
            OptimOutcome::new(theta_hat_opt, value, term_status, iterations, fn_evals, grad);

        // Assert
        assert!(result.is_err(), "Missing theta_hat should cause OptimOutcome::new to fail");
    }

    #[test]
    // Purpose
    // -------
    // Verify that `OptimOutcome::new` rejects a non-finite log-likelihood
    // value.
    //
    // Given
    // -----
    // - A finite `theta_hat`.
    // - `value = f64::NAN`.
    //
    // Expect
    // ------
    // - `OptimOutcome::new` returns an error via `validate_value`.
    fn optimoutcome_new_errors_on_non_finite_value() {
        // Arrange
        let theta_hat = Theta::from(vec![0.0_f64, 1.0]);
        let theta_hat_opt = Some(theta_hat);
        let value = f64::NAN;
        let termination_reason = TerminationReason::SolverConverged;
        let term_status: TerminationStatus = TerminationStatus::Terminated(termination_reason);
        let iterations = 5_u64;
        let fn_evals = FnEvalMap::default();
        let grad = None;

        // Act
        let result =
            OptimOutcome::new(theta_hat_opt, value, term_status, iterations, fn_evals, grad);

        // Assert
        assert!(result.is_err(), "Non-finite value should cause OptimOutcome::new to fail");
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `OptimOutcome::new` populates fields correctly for a
    // valid solver state, including the boolean `term_status` flag and
    // `grad_norm`.
    //
    // Given
    // -----
    // - A finite `theta_hat` and log-likelihood value.
    // - `TerminationStatus::Terminated`.
    // - A small number of iterations.
    // - A non-zero gradient vector.
    //
    // Expect
    // ------
    // - `OptimOutcome::new` returns `Ok(outcome)`.
    // - `outcome.term_status == true` and `outcome.status == "Terminated"`.
    // - `outcome.iterations` equals the provided iteration count.
    // - `outcome.theta_hat` equals the input `theta_hat`.
    // - `outcome.grad_norm` matches the L2 norm of the provided gradient.
    fn optimoutcome_new_populates_fields_for_valid_input() {
        // Arrange
        let theta_hat = Theta::from(vec![0.0_f64, 3.0]);
        let theta_hat_opt = Some(theta_hat.clone());
        let value = -2.5;
        let termination_reason = TerminationReason::SolverConverged;
        let term_status: TerminationStatus = TerminationStatus::Terminated(termination_reason);
        let iterations = 7_u64;
        let fn_evals = FnEvalMap::default();
        let grad = Some(Grad::from(vec![1.0_f64, 2.0]));

        // Expected gradient norm (sqrt(1^2 + 2^2) = sqrt(5)).
        let expected_grad_norm = (1.0_f64 + 4.0_f64).sqrt();

        // Act
        let outcome =
            OptimOutcome::new(theta_hat_opt, value, term_status, iterations, fn_evals, grad)
                .expect("Valid inputs should construct an OptimOutcome");

        // Assert
        assert_eq!(outcome.theta_hat, theta_hat);
        assert_eq!(outcome.value, value);
        assert!(outcome.term_status, "term_status should be true for Terminated");
        assert_eq!(outcome.status, "SolverConverged");
        assert_eq!(outcome.iterations, iterations as usize);
        let grad_norm =
            outcome.grad_norm.expect("grad_norm should be present when a gradient is provided");
        assert!((grad_norm - expected_grad_norm).abs() < 1e-12);
    }
}
