//! Execution helper that runs an `argmin` solver on a log-likelihood problem and
//! returns a crate-friendly [`OptimOutcome`].
use crate::optimization::{
    errors::OptResult,
    loglik_optimizer::{
        Grad, LogLikelihood, MLEOptions, OptimOutcome, Theta, adapter::ArgMinAdapter,
    },
};
#[cfg(feature = "obs_slog")]
use argmin::core::{CostFunction, Gradient};
use argmin::core::{Executor, State};
#[cfg(feature = "obs_slog")]
use argmin_math::ArgminL2Norm;

/// Run an `argmin` optimization for a log-likelihood problem.
///
/// This is the shared runner used by both line-search variants. It wires up:
/// - the user model via [`ArgMinAdapter`],
/// - the chosen `Solver` (e.g. L-BFGS with Hager–Zhang/More–Thuente),
/// - initial parameter `theta0`,
/// - optional observers (behind the `obs_slog` feature),
/// - optional `max_iters`,
///   then executes the solver and converts the result into [`OptimOutcome`].
///
/// # Type Parameters
/// - `F`: Your log-likelihood type implementing [`LogLikelihood`].
/// - `S`: Any `argmin` solver whose `Problem` is `ArgMinAdapter<'a, F>` and whose
///   `IterState` matches the aliases `Theta` (parameters), `Grad` (gradient),
///   and `f64` as the float type.
///
/// # Arguments
/// - `theta0`: Initial parameter vector. It is **consumed** and set on the optimizer
///   state via `state.param(theta0)`.
/// - `opts`: Optimizer options (tolerances, verbosity, max iters, etc.).
/// - `problem`: An [`ArgMinAdapter`] wrapping the user’s model and data.
/// - `solver`: A fully constructed solver (e.g. from
///   [`build_optimizer_hager_zhang`](crate::optimization::loglik_optimizer::loglik_builders::build_optimizer_hager_zhang)
///   or
///   [`build_optimizer_more_thuente`](crate::optimization::loglik_optimizer::loglik_builders::build_optimizer_more_thuente)).
///
/// # Feature flags
/// If the `obs_slog` feature is enabled and `opts.verbose == true`, a terminal
/// slog observer is attached with `ObserverMode::Always` and a one-time pre-iteration
/// line logs ℓ(θ₀) and, if available, ||grad|| before the first iteration.
///
/// # Returns
/// An [`OptimOutcome`] containing the best parameter found, best log-likelihood value ℓ(θ̂),
/// termination status, iteration count, function-evaluation counts, and the last
/// available gradient's norm if it can be calculated.
///
/// # Errors
/// - Propagates any `argmin` runtime error (observer failures, solver errors,
///   line-search failures, etc.) via your crate’s `From<argmin::core::Error>`
///   conversion.
/// - Propagates any validation errors encountered when constructing
///   [`OptimOutcome`].
///
/// # Examples
/// ```ignore
/// let problem = ArgMinAdapter::new(&model, &data);
/// let solver  = build_optimizer_hager_zhang(&opts)?;
/// let out     = run_lbfgs(theta0.clone(), &opts, problem, solver)?;
/// println!("done in {} iters, status: {}", out.iterations, out.status);
/// ```
pub fn run_lbfgs<'a, F, S>(
    theta0: Theta, opts: &MLEOptions, problem: ArgMinAdapter<'a, F>, solver: S,
) -> OptResult<OptimOutcome>
where
    F: LogLikelihood,
    S: argmin::core::Solver<
            ArgMinAdapter<'a, F>,
            argmin::core::IterState<Theta, Grad, (), (), (), f64>,
        > + Send
        + 'static,
{
    #[cfg(feature = "obs_slog")]
    if opts.verbose {
        log_initial_state(&theta0, &problem)?;
    }
    let mut optimizer = Executor::new(problem, solver);
    optimizer = optimizer.configure(|state| state.param(theta0));
    #[cfg(feature = "obs_slog")]
    if opts.verbose {
        let observer = argmin_observer_slog::SlogLogger::term_noblock();
        optimizer = optimizer.add_observer(observer, argmin::core::observers::ObserverMode::Always);
    }
    if let Some(max_iter) = opts.tols.max_iter {
        optimizer = optimizer.configure(|state| state.max_iters(max_iter as u64));
    }

    let mut result = optimizer.run()?.state().clone();
    let iterations = result.get_iter();
    let function_counts = result.get_func_counts().clone();
    let termination = result.get_termination_status().clone();
    let grad = result.take_gradient();
    OptimOutcome::new(
        result.take_best_param(),
        -result.get_best_cost(),
        termination,
        iterations,
        function_counts,
        grad,
    )
}

// ---- Helper Methods ----

#[cfg(feature = "obs_slog")]
fn log_initial_state<F>(theta0: &Theta, problem: &ArgMinAdapter<'_, F>) -> OptResult<()>
where
    F: LogLikelihood,
{
    let ll0 = -problem.cost(theta0)?;
    let g0n = problem.gradient(theta0).ok().map(|g| g.l2_norm());

    eprintln!(
        "init: ell(theta0) = {:.6}{}",
        ll0,
        g0n.map(|n| format!(", ||grad|| = {:.6}", n)).unwrap_or_default()
    );
    Ok(())
}
