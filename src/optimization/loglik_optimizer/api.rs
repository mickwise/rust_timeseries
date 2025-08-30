//! High-level entry point for maximizing a user-provided `LogLikelihood`.
//!
//! This selects an L-BFGS solver with either Hager–Zhang or More–Thuente line
//! search, wraps the model in an `ArgMinAdapter` (which *minimizes* `-ℓ(θ)`),
//! and delegates the run to `run_lbfgs`.
use crate::optimization::{
    errors::OptResult,
    loglik_optimizer::{
        OptimOutcome, Theta,
        adapter::ArgMinAdapter,
        builders::{build_optimizer_hager_zhang, build_optimizer_more_thuente},
        run::run_lbfgs,
        traits::{LineSearcher, LogLikelihood, MLEOptions},
    },
};

/// Maximize a log-likelihood `ℓ(θ)` using L-BFGS with the chosen line search.
///
/// # Behavior
/// - Validates the initial guess via `f.check(theta0, data)`.
/// - Wraps `(f, data)` in an `ArgMinAdapter` that exposes a *minimization*
///   problem `c(θ) = -ℓ(θ)` to `argmin`.
/// - Builds an L-BFGS solver with either **Hager–Zhang** or **More–Thuente**
///   line search based on `opts.line_searcher`.
/// - Calls `run_lbfgs`, which configures the executor (initial params,
///   max iters, optional observers) and returns an `OptimOutcome`.
///
/// # Parameters
/// - `f`: Your model implementing [`LogLikelihood`].
/// - `theta0`: Initial parameter vector.
/// - `data`: Model data passed through to `value`/`grad`.
/// - `opts`: Optimizer options (tolerances, line search choice, verbosity, etc.).
///
/// # Errors
/// - Propagates any error from `f.check`.
/// - Propagates builder errors from `build_optimizer_*`.
/// - Propagates runtime errors from `run_lbfgs` (e.g., line search failures).
///
/// # Returns
/// An [`OptimOutcome`] containing `theta_hat`, best value `ℓ(θ̂)`,
/// termination status, iteration counts, function evaluation counts, and
/// optionally the gradient norm.
///
/// # Example
/// ```no_run
/// use ndarray::array;
/// use your_crate::optimization::loglik_optimizer::{
///     maximize, MLEOptions, Tolerances, LineSearcher, LogLikelihood
/// };
///
/// struct MyLL;
/// impl LogLikelihood for MyLL {
///     type Data = ();
///     fn value(&self, theta: &ndarray::Array1<f64>, _: &()) -> your_crate::optimization::opt_errors::OptResult<f64> {
///         // Simple concave log-likelihood: -(θ·θ)
///         Ok(-theta.dot(theta))
///     }
///     fn check(&self, _: &ndarray::Array1<f64>, _: &()) -> your_crate::optimization::opt_errors::OptResult<()> {
///         Ok(())
///     }
/// }
///
/// let f = MyLL;
/// let theta0 = array![0.1, -0.2, 0.3];
/// let data = ();
/// let opts = MLEOptions {
///     tols: Tolerances { tol_grad: Some(1e-6), tol_cost: None, max_iter: Some(200) },
///     line_searcher: LineSearcher::HagerZhang,
///     verbose: false,
/// };
///
/// let out = maximize(&f, &theta0, &data, &opts)?;
/// println!("θ̂ = {:?}", out.theta_hat);
/// # Ok::<(), your_crate::optimization::opt_errors::OptError>(())
/// ```
pub fn maximize<F: LogLikelihood>(
    f: &F, theta0: Theta, data: &F::Data, opts: &MLEOptions,
) -> OptResult<OptimOutcome> {
    f.check(&theta0, data)?;
    let problem = ArgMinAdapter::new(f, data);
    match opts.line_searcher {
        LineSearcher::MoreThuente => {
            let solver = build_optimizer_more_thuente(opts)?;
            run_lbfgs(theta0, opts, problem, solver)
        }
        LineSearcher::HagerZhang => {
            let solver = build_optimizer_hager_zhang(opts)?;
            run_lbfgs(theta0, opts, problem, solver)
        }
    }
}
