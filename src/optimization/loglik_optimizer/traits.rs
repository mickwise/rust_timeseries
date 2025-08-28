//! Public API surface for log-likelihood maximization.
//!
//! - [`LogLikelihood`]: trait users implement for their model.
//! - [`MLEOptions`] and [`Tolerances`]: configuration for the optimizer.
//! - [`LineSearcher`]: choice of line search used by L-BFGS.
//! - [`OptimOutcome`]: normalized result returned by the high-level `maximize` API.
//!
//! Convention: we *maximize* a user log-likelihood `ℓ(θ)` by minimizing the cost
//! `c(θ) = -ℓ(θ)`. If an analytic gradient is provided, it should be the gradient
//! of the log-likelihood (`∇ℓ(θ)`); the adapter flips the sign as needed.
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

/// User-implemented log-likelihood interface.
///
/// You maximize `ℓ(θ)`; internally we minimize the cost `c(θ) = -ℓ(θ)`.
/// If you provide an analytic gradient, return the gradient of the
/// log-likelihood `∇ℓ(θ)` (the adapter flips the sign to match the cost).
///
/// - `type Data`: per-model data carried into `value`/`grad`/`check`.
///
/// Required:
/// - `value(&Theta, &Data) -> OptResult<Cost>`: evaluate `ℓ(θ)`.
///   - Errors: return a descriptive `OptError` for invalid inputs or model failures.
/// - `check(&Theta, &Data) -> OptResult<()>`: validation hook to reject
///   obviously invalid `θ`/`data` pairs. Called once before optimization.
///
/// Optional:
/// - `grad(&Theta, &Data) -> OptResult<Grad>`: analytic gradient `∇ℓ(θ)`.
///   If not implemented, robust finite differences are used automatically.
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

/// Choice of line search used inside the L-BFGS solver.
///
/// Variants:
/// - `MoreThuente`: More–Thuente line search.
/// - `HagerZhang`: Hager–Zhang line search.
///
/// Parsing:
/// This enum implements `FromStr` and accepts case-insensitive names
/// (`"MoreThuente"`, `"HagerZhang"`). Unknown names return
/// `OptError::InvalidLineSearch`.
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

/// Optimizer-level configuration.
///
/// Fields:
/// - `tols: Tolerances` — numerical tolerances and iteration limits.
/// - `line_searcher: LineSearcher` — line-search algorithm used by L-BFGS.
/// - `verbose: bool` — if `true`, attaches an observer (behind the `obs_slog`
///   feature) and prints progress.
///
/// Constructor:
/// - `new(tols, line_searcher, verbose) -> Self` — builds options; validation of
///   numeric values is handled in `Tolerances::new`.
///
/// Default:
/// - `tols`: `tol_grad = 1e-6`, `tol_cost = None`, `max_iter = 300`
/// - `line_searcher`: `MoreThuente`
/// - `verbose`: `false`
/// - `lbfgs_mem`: `None` (uses default of 7)
#[derive(Debug, Clone, PartialEq)]
pub struct MLEOptions {
    pub tols: Tolerances,
    pub line_searcher: LineSearcher,
    pub verbose: bool,
    pub lbfgs_mem: Option<usize>,
}

impl MLEOptions {
    /// Create a new set of optimizer options.
    ///
    /// This constructor does not mutate values; validation of numeric fields is
    /// performed inside [`Tolerances::new`].
    pub fn new(
        tols: Tolerances, line_searcher: LineSearcher, verbose: bool, lbfgs_mem: Option<usize>,
    ) -> OptResult<Self> {
        if let Some(m) = lbfgs_mem {
            if m == 0 {
                return Err(OptError::InvalidLBFGSMem {
                    mem: m,
                    reason: "L-BFGS memory must be greater than zero.",
                });
            }
        }
        Ok(Self { tols, line_searcher, verbose, lbfgs_mem })
    }
}

impl Default for MLEOptions {
    fn default() -> Self {
        Self {
            tols: Tolerances::new(Some(1e-6), None, Some(300)).unwrap(),
            line_searcher: LineSearcher::MoreThuente,
            verbose: false,
            lbfgs_mem: None,
        }
    }
}

/// Numerical tolerances and iteration limits used by the optimizer.
///
/// - `tol_grad`: terminate when the gradient norm falls below this threshold.
/// - `tol_cost`: terminate when the change in cost falls below this threshold.
/// - `max_iter`: hard cap on the number of iterations.
///
/// Any field can be `None` but **at least one** of the three must be provided
/// (see [`Tolerances::new`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerances {
    pub tol_grad: Option<f64>,
    pub tol_cost: Option<f64>,
    pub max_iter: Option<usize>,
}

impl Tolerances {
    /// Construct validated tolerances.
    ///
    /// # Rules
    /// - At least one of `tol_grad`, `tol_cost`, or `max_iter` must be `Some`.
    /// - If provided, tolerances must be **finite and strictly positive**.
    /// - If provided, `max_iter` must be `> 0`.
    ///
    /// # Errors
    /// - [`OptError::NoTolerancesProvided`] if all three are `None`.
    /// - [`OptError::InvalidTolGrad`] / [`OptError::InvalidTolCost`] for non-finite or non-positive tolerances.
    /// - `OptError::InvalidMaxIter` if `max_iter == 0`.
    pub fn new(
        tol_grad: Option<f64>, tol_cost: Option<f64>, max_iter: Option<usize>,
    ) -> OptResult<Self> {
        if tol_grad.is_none() && tol_cost.is_none() && max_iter.is_none() {
            return Err(OptError::NoTolerancesProvided);
        }
        verify_tol_cost(tol_cost)?;
        verify_tol_grad(tol_grad)?;
        if let Some(max_iter) = max_iter {
            if max_iter == 0 {
                return Err(OptError::InvalidMaxIter {
                    max_iter,
                    reason: "Maximum iterations must be greater than zero.",
                });
            }
        }
        Ok(Self { tol_grad, tol_cost, max_iter })
    }
}

/// Canonical result returned by `maximize`.
///
/// - `theta_hat`: best parameter vector found.
/// - `value`: best **log-likelihood** value `ℓ(θ)` (not the cost).
/// - `converged`: `true` if the solver reported a terminating status other
///   than `NotTerminated`.
/// - `status`: human-readable termination status string.
/// - `iterations`: number of optimizer iterations performed.
/// - `fn_evals`: function-evaluation counters reported by `argmin`.
/// - Keys follow argmin’s counters, e.g., cost_count, gradient_count, etc.
/// - `grad_norm`: norm of the last available gradient, if present.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimOutcome {
    pub theta_hat: Theta,
    pub value: f64,
    pub converged: bool,
    pub status: String,
    pub iterations: usize,
    pub fn_evals: FnEvalMap,
    pub grad_norm: Option<f64>,
}

impl OptimOutcome {
    /// Build a validated [`OptimOutcome`] from raw solver state.
    ///
    /// Performs:
    /// - `theta_hat` check via `validate_theta_hat` (present and all finite).
    /// - `value` check via `validate_value` (finite).
    /// - Maps `TerminationStatus` into `(converged, status)`.
    /// - Computes `grad_norm` if a gradient was provided.
    ///
    /// # Errors
    /// - Propagates any validation errors for `theta_hat` or `value`.
    pub fn new(
        theta_hat_opt: Option<Theta>, value: f64, converged: TerminationStatus, iterations: u64,
        fn_evals: FnEvalMap, grad: Option<Grad>,
    ) -> OptResult<Self> {
        let theta_hat = validate_theta_hat(theta_hat_opt)?;
        validate_value(value)?;
        let status: String;
        let converged = match converged {
            TerminationStatus::NotTerminated => {
                status = "Not terminated".to_string();
                false
            }
            _ => {
                status = format!("{converged:?}");
                true
            }
        };
        let iterations = iterations as usize;
        let grad_norm = grad.map(|g| g.l2_norm());
        Ok(Self { theta_hat, value, converged, status, iterations, fn_evals, grad_norm })
    }
}
