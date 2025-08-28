//! Builders for L-BFGS solvers used by the log-likelihood optimizer.
//!
//! This module exposes two concrete constructors and one generic helper:
//!
//! - [`build_optimizer_hager_zhang`] — L-BFGS with **Hager–Zhang** line search.
//! - [`build_optimizer_more_thuente`] — L-BFGS with **More–Thuente** line search.
//! - [`configure_lbfgs`] — generic helper that applies optional tolerances from [`MLEOptions`].
//!
//! These functions only **configure the solver**. They do **not** set the initial
//! parameter vector or `max_iters`; that is handled by the runner (e.g. [`run_optimizer`])
//! after the problem is wrapped via [`ArgMinAdapter`].
//!
//! # Type aliases used here
//! - [`HZ`], [`MT`]: line-search types.
//! - [`LbfgsHz`], [`LbfgsMt`]: L-BFGS solvers specialized for our [`Theta`] / [`Grad`].
//! - [`LBFGS_MEM`]: number of correction pairs kept by L-BFGS.
//!
//! # Errors
//! Any invalid tolerance passed to `with_tolerance_grad` / `with_tolerance_cost` is
//! surfaced as `argmin::core::Error` and is automatically converted to your crate’s
//! [`OptError`] via `From`, so callers can simply use `?`.
//!
//! # Example
//! ```ignore
//! use crate::optimization::loglik_optimizer::{
//!     loglik_adapter::ArgMinAdapter,
//!     loglik_traits::{LineSearcher, MLEOptions},
//!     loglik_builders::{build_optimizer_hager_zhang, build_optimizer_more_thuente},
//!     loglik_run::run_optimizer,
//! };
//!
//! let problem = ArgMinAdapter::new(&model, &data);
//!
//! let solver = match opts.line_searcher {
//!     LineSearcher::HagerZhang   => build_optimizer_hager_zhang(&opts)?,
//!     LineSearcher::MoreThuente  => build_optimizer_more_thuente(&opts)?,
//! };
//!
//! let outcome = run_optimizer(theta0.clone(), &opts, problem, solver)?;
//! ```
//!
//! # Notes
//! - [`configure_lbfgs`] is generic over the line-search type.
//! - Builders are pure configuration; they don’t mutate `theta0` or touch the executor.
use argmin::solver::quasinewton::LBFGS;

use crate::optimization::{
    errors::OptResult,
    loglik_optimizer::{
        Grad, LBFGS_MEM, Theta,
        traits::MLEOptions,
        types::{HZ, LbfgsHz, LbfgsMt, MT},
    },
};

/// Build an L-BFGS solver that uses **Hager–Zhang** line search.
///
/// This allocates `LBFGS_MEM` correction pairs, creates the Hager–Zhang
/// line-search object, and applies any optional tolerances from `opts.tols`
/// (i.e., `tol_grad`, `tol_cost`) via `with_tolerance_grad` /
/// `with_tolerance_cost`.
///
/// # Parameters
/// - `opts`: Optimizer options. Only `opts.tols.tol_grad` and
///   `opts.tols.tol_cost` are used here.
///
/// # Returns
/// A configured [`LbfgsHz`] solver ready to be passed to `run_optimizer`.
///
/// # Errors
/// Propagates any error returned by `with_tolerance_grad` /
/// `with_tolerance_cost` (e.g., invalid or non-finite tolerances). These
/// originate as `argmin::core::Error` and are converted into your `OptError`.
///
/// # Examples
/// ```ignore
/// let solver = build_optimizer_hager_zhang(&opts)?;
/// ```
pub fn build_optimizer_hager_zhang(opts: &MLEOptions) -> OptResult<LbfgsHz> {
    let hager_zhang = HZ::new();
    let lbfgs = match opts.lbfgs_mem {
        Some(m) => LbfgsHz::new(hager_zhang, m),
        None => LbfgsHz::new(hager_zhang, LBFGS_MEM),
    };
    configure_lbfgs(lbfgs, opts)
}

/// Build an L-BFGS solver that uses **More–Thuente** line search.
///
/// This allocates `LBFGS_MEM` correction pairs, creates the More–Thuente
/// line-search object, and applies any optional tolerances from `opts.tols`
/// (i.e., `tol_grad`, `tol_cost`) via `with_tolerance_grad` /
/// `with_tolerance_cost`.
///
/// This function does **not** set an initial parameter vector or `max_iters`;
/// those are applied later by the runner (`run_optimizer`).
///
/// # Parameters
/// - `opts`: Optimizer options. Only `opts.tols.tol_grad` and
///   `opts.tols.tol_cost` are used here.
///
/// # Returns
/// A configured [`LbfgsMt`] solver ready to be passed to `run_optimizer`.
///
/// # Errors
/// Propagates any error returned by `with_tolerance_grad` /
/// `with_tolerance_cost` (e.g., invalid or non-finite tolerances). These
/// originate as `argmin::core::Error` and are converted into your `OptError`.
///
/// # Examples
/// ```ignore
/// let solver = build_optimizer_more_thuente(&opts)?;
/// ```
pub fn build_optimizer_more_thuente(opts: &MLEOptions) -> OptResult<LbfgsMt> {
    let more_thuente = MT::new();
    let lbfgs = match opts.lbfgs_mem {
        Some(m) => LbfgsMt::new(more_thuente, m),
        None => LbfgsMt::new(more_thuente, LBFGS_MEM),
    };
    configure_lbfgs(lbfgs, opts)
}

/// Apply optional tolerance settings from `opts` to an L-BFGS solver.
///
/// This helper is **generic over the line-search type `L`** and simply wires
/// `opts.tols.tol_grad` and `opts.tols.tol_cost` into the solver using
/// `with_tolerance_grad` / `with_tolerance_cost`. It returns the updated solver
/// so it can be used fluently by builder functions.
///
/// It intentionally does **not** set initial parameters or `max_iters`.
/// Do that in your executor/runner.
///
/// # Type Parameters
/// - `L`: The line-search type used by the L-BFGS solver (e.g., Hager–Zhang,
///   More–Thuente).
///
/// # Parameters
/// - `solver`: The L-BFGS instance to configure.
/// - `opts`: Source of optional tolerances (`opts.tols.tol_grad`, `opts.tols.tol_cost`).
///
/// # Returns
/// The same solver with tolerances applied (if present).
///
/// # Errors
/// Propagates any error produced by the underlying `with_tolerance_*` calls.
/// These originate as `argmin::core::Error` and are converted into your `OptError`.
///
/// # Examples
/// ```ignore
/// let lbfgs = LBFGS::new(HZ::new(), LBFGS_MEM);
/// let lbfgs = configure_lbfgs(lbfgs, &opts)?;
/// ```
pub fn configure_lbfgs<L>(
    mut solver: LBFGS<L, Theta, Grad, f64>, opts: &MLEOptions,
) -> OptResult<LBFGS<L, Theta, Grad, f64>> {
    if let Some(g) = opts.tols.tol_grad {
        solver = solver.with_tolerance_grad(g)?;
    }
    if let Some(c) = opts.tols.tol_cost {
        solver = solver.with_tolerance_cost(c)?;
    }
    Ok(solver)
}
