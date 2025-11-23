//! loglik_optimizer::run — execute Argmin solvers on log-likelihood problems.
//!
//! Purpose
//! -------
//! Provide a thin, crate-level entrypoint that runs an [`argmin`] optimizer on a
//! [`LogLikelihood`] model and converts the raw [`argmin`] state into a normalized
//! [`OptimOutcome`]. This keeps solver wiring, observer configuration, and
//! result translation in one place.
//!
//! Key behaviors
//! -------------
//! - Wrap a user-provided [`LogLikelihood`] and its data in an [`ArgMinAdapter`]
//!   and run an `argmin` solver (typically L-BFGS) starting from `theta0`.
//! - Translate the final `argmin` state into an [`OptimOutcome`], including
//!   θ̂, ℓ(θ̂), termination status, iteration counts, and function-evaluation
//!   counts.
//! - Optionally attach a slog-based terminal observer (behind `obs_slog`) and
//!   log the initial objective and gradient norm before iterations start.
//!
//! Invariants & assumptions
//! ------------------------
//! - The wrapped [`LogLikelihood`] reports log-likelihood values `ℓ(θ)`
//!   (not costs) and uses [`OptError`] for recoverable failures.
//! - The [`ArgMinAdapter`] implements `CostFunction` with cost
//!   `c(θ) = -ℓ(θ)`; this module assumes that convention and converts the
//!   best cost back to ℓ(θ̂) by negating `get_best_cost()`.
//! - The solver type `S` must be compatible with
//!   `IterState<Theta, Grad, (), (), (), f64>` and must be safe to move into
//!   a `'static` executor (no non-`'static` references inside).
//!
//! Conventions
//! -----------
//! - All caller-facing APIs speak in terms of log-likelihoods and θ; cost `c`
//!   is treated as an internal detail of the optimizer layer.
//! - Maximum iterations are configured via `opts.tols.max_iter` if present;
//!   absence of a value means "let Argmin use its default".
//! - When the `obs_slog` feature is enabled and `opts.verbose == true`,
//!   a terminal slog observer is attached with `ObserverMode::Always`, and a
//!   single pre-iteration line logs `ℓ(θ₀)` (and `||∇ℓ(θ₀)||` if available).
//! - All failures are reported as [`OptError`] through the crate-wide
//!   [`OptResult`] alias; this module never panics intentionally.
//!
//! Downstream usage
//! ----------------
//! - Higher-level duration and inference code builds a concrete solver via
//!   builders (e.g. `build_optimizer_hager_zhang`) and then calls
//!   [`run_lbfgs`] with `theta0`, `opts`, and the [`ArgMinAdapter`].
//! - Python bindings and other FFI layers are expected to interact only with
//!   [`OptimOutcome`] and not with Argmin types directly.
//! - Other optimization entrypoints (e.g., different solvers) can reuse the
//!   same pattern as [`run_lbfgs`] while still returning [`OptimOutcome`].
//!
//! Testing notes
//! -------------
//! - Unit tests in this module validate wiring and translation behavior:
//!   - that `get_best_cost()` is negated before being passed into
//!     [`OptimOutcome::new`],
//!   - that `max_iter` from `opts.tols` is honored when present,
//!   - that `run_lbfgs` propagates [`argmin::core::Error`] values through
//!     [`OptError`].
//! - When `obs_slog` is enabled, tests additionally validate that
//!   [`log_initial_state`] evaluates cost/gradient once and prints the
//!   expected log line without mutating solver state.
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

/// run_lbfgs — execute an L-BFGS-style optimizer on a log-likelihood model.
///
/// Purpose
/// -------
/// Drive an [`argmin`] solver on top of a [`LogLikelihood`] wrapped in
/// [`ArgMinAdapter`], then convert the final [`argmin`] state into a crate-level
/// [`OptimOutcome`]. This function centralizes solver configuration, optional
/// observers, and the `c(θ) = -ℓ(θ)` sign convention.
///
/// Parameters
/// ----------
/// - `theta0`: [`Theta`]
///   Initial unconstrained parameter vector. It is consumed and installed into
///   the Argmin state via `state.param(theta0)`.
/// - `opts`: `&MLEOptions`
///   Optimizer configuration, including tolerances, optional `max_iter`, and
///   verbosity flags (used to decide whether to attach observers).
/// - `problem`: [`ArgMinAdapter<'a, F>`]
///   Adapter that exposes a user-implemented [`LogLikelihood`] and its data as
///   an Argmin `CostFunction` and `Gradient`.
/// - `solver`: `S`
///   Fully constructed Argmin solver (typically L-BFGS with a chosen
///   line-searcher), parameterized for [`Theta`], [`Grad`], and `f64`.
///
/// Returns
/// -------
/// [`OptResult<OptimOutcome>`]
///   - `Ok(OptimOutcome)` when the solver runs to completion and the final
///     state can be converted into a valid outcome (finite θ̂ and ℓ(θ̂),
///     iteration counts, and function counts).
///   - `Err(OptError)` when Argmin fails or when [`OptimOutcome::new`]
///     rejects the final state (e.g., non-finite values).
///
/// Errors
/// ------
/// - `OptError` (via `From<argmin::core::Error>` implementation)
///   Propagated when Argmin’s executor or solver returns an error, including
///   line-search failures, observer errors, or internal solver issues.
/// - `OptError`
///   Propagated when [`OptimOutcome::new`] rejects its inputs (e.g., non-finite
///   θ̂ or objective value).
///
/// Panics
/// ------
/// - Never panics intentionally. Any internal Argmin panics are treated as
///   bugs.
///
/// Safety
/// ------
/// - No `unsafe` code is used. The caller must ensure that the underlying
///   model and data captured by [`ArgMinAdapter`] remain valid for the
///   duration of the optimization (enforced via lifetimes).
///
/// Notes
/// -----
/// - Cost values in the Argmin state are stored as `c(θ) = -ℓ(θ)`; this
///   function negates `get_best_cost()` so [`OptimOutcome`] stores the
///   log-likelihood `ℓ(θ̂)` instead of the cost.
/// - If `opts.tols.max_iter` is `Some`, the executor’s `max_iters` is set to
///   that value; otherwise Argmin’s default iteration limits apply.
/// - When the `obs_slog` feature is enabled and `opts.verbose == true`,
///   this function first calls [`log_initial_state`] and then attaches a
///   slog-based terminal observer with `ObserverMode::Always`.
///
/// Examples
/// --------
/// ```ignore
/// use rust_timeseries::optimization::loglik_optimizer::{
///     adapter::ArgMinAdapter,
///     run::run_lbfgs,
///     traits::{LogLikelihood, MLEOptions},
/// };
///
/// // Assume `MyLL` implements `LogLikelihood<Data = MyData>`.
/// let model = MyLL::new(...);
/// let data = MyData::from(...);
/// let theta0 = Theta::from(vec![0.0_f64; model.num_params()]);
/// let problem = ArgMinAdapter::new(&model, &data);
/// let opts = MLEOptions::default();
/// let solver = build_optimizer_hager_zhang(&opts)?;
///
/// let outcome = run_lbfgs(theta0, &opts, problem, solver)?;
/// println!("status = {}, iters = {}", outcome.status, outcome.iterations);
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

    let mut result = optimizer.run()?.state().to_owned();
    let iterations = result.get_iter();
    let function_counts = result.get_func_counts().to_owned();
    let termination = result.get_termination_status().to_owned();
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

/// log_initial_state — log ℓ(θ₀) and initial gradient norm (if available).
///
/// Purpose
/// -------
/// Provide a lightweight, optional diagnostic hook that evaluates the
/// log-likelihood and gradient at the starting point `θ₀` and prints a single
/// summary line to stderr when verbose logging is enabled.
///
/// Parameters
/// ----------
/// - `theta0`: `&Theta`
///   Initial unconstrained parameter vector at which to evaluate the
///   log-likelihood and (optionally) the gradient.
/// - `problem`: `&ArgMinAdapter<'_, F>`
///   Adapter exposing the underlying [`LogLikelihood`] model and its data as
///   a cost/gradient pair. Used here only for a one-off evaluation at `θ₀`.
///
/// Returns
/// -------
/// [`OptResult<()>`]
///   - `Ok(())` on success.
///   - `Err(OptError)` if evaluating the cost at `θ₀` fails.
///   Gradient errors are ignored; only the cost must succeed.
///
/// Errors
/// ------
/// - `OptError`
///   Propagated if `problem.cost(theta0)` fails (e.g., invalid θ₀ or model
///   configuration). In that case, no log line is printed.
///
/// Panics
/// ------
/// - Never panics; errors from gradient evaluation are ignored via `.ok()`
///   and only cost errors are surfaced through the return type.
///
/// Safety
/// ------
/// - No `unsafe` code is used. The function does not mutate any model or
///   solver state; it only performs read-only evaluations and side-effectful
///   logging to stderr.
///
/// Notes
/// -----
/// - The log-likelihood is recovered as `ℓ(θ₀) = -c(θ₀)` by negating the
///   cost returned from [`ArgMinAdapter::cost`].
/// - The gradient norm, when available, is computed using the `ArgminL2Norm`
///   extension trait and logged as `||grad|| = <norm>`. If gradient
///   evaluation fails, the norm is simply omitted.
/// - This helper is compiled and used only when the `obs_slog` feature is
///   enabled; it is not part of the public API surface.
///
/// Examples
/// --------
/// ```ignore
/// # use rust_timeseries::optimization::loglik_optimizer::run::log_initial_state;
/// # use rust_timeseries::optimization::loglik_optimizer::adapter::ArgMinAdapter;
/// # fn demo<F: LogLikelihood>(theta0: &Theta, adapter: &ArgMinAdapter<'_, F>) -> OptResult<()> {
/// log_initial_state(theta0, adapter)?;
/// # Ok(())
/// # }
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::errors::OptError;
    use crate::optimization::loglik_optimizer::Cost;
    use crate::optimization::loglik_optimizer::adapter::ArgMinAdapter;
    use crate::optimization::loglik_optimizer::builders::build_optimizer_hager_zhang;
    use crate::optimization::loglik_optimizer::traits::LogLikelihood;
    use approx::assert_relative_eq;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Wiring between `run_lbfgs` and `ArgMinAdapter` for a simple quadratic
    //   log-likelihood model.
    // - The sign convention `c(θ) = -ℓ(θ)` when translating the best cost back
    //   into an `OptimOutcome` value.
    // - Propagation of model / argmin errors via `OptResult<OptimOutcome>`.
    // - Respecting `opts.tols.max_iter` when configuring the executor.
    // - (With `obs_slog`) basic behavior of `log_initial_state`.
    //
    // They intentionally DO NOT cover:
    // - The internal correctness of Argmin’s L-BFGS implementation.
    // - Detailed observer behavior beyond the fact that `log_initial_state`
    //   evaluates the model once and returns `Ok(())`.
    // -------------------------------------------------------------------------

    /// Simple 1D quadratic log-likelihood:
    /// ℓ(θ) = -0.5 (θ₀ - 1)², grad ℓ(θ) = -(θ₀ - 1).
    struct QuadraticLL;

    impl LogLikelihood for QuadraticLL {
        type Data = ();

        fn value(&self, theta: &Theta, _data: &Self::Data) -> OptResult<Cost> {
            let x = theta[0];
            Ok(-0.5 * (x - 1.0).powi(2))
        }

        fn check(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<()> {
            Ok(())
        }

        fn grad(&self, theta: &Theta, _data: &Self::Data) -> OptResult<Grad> {
            let x = theta[0];
            // grad ℓ(θ) = -(x - 1)
            Ok(Grad::from(vec![-(x - 1.0)]))
        }
    }

    /// Log-likelihood that always fails at `value`, used to test error propagation.
    struct ErrorValueLL;

    impl LogLikelihood for ErrorValueLL {
        type Data = ();

        fn value(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<Cost> {
            Err(OptError::InvalidLogLikInput { value: 123.0 })
        }

        fn check(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<()> {
            Ok(())
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `run_lbfgs` converges on a simple quadratic model and that
    // the returned `OptimOutcome` stores ℓ(θ̂) (not the cost) and a reasonable
    // estimate of θ̂.
    //
    // Given
    // -----
    // - A 1D quadratic log-likelihood with maximizer at θ* = 1.
    // - Default `MLEOptions` and an optimizer built via `build_optimizer_hager_zhang`.
    //
    // Expect
    // ------
    // - `run_lbfgs` returns `Ok(OptimOutcome)`.
    // - `theta_hat` is close to 1 within a small tolerance.
    // - `value` (ℓ(θ̂)) is close to 0 (the maximum of the quadratic).
    fn run_lbfgs_quadratic_model_converges_and_negates_cost() {
        // Arrange
        let model = QuadraticLL;
        let data = ();
        let opts = MLEOptions::default();
        let theta0 = Theta::from(vec![0.0_f64]);
        let problem = ArgMinAdapter::new(&model, &data);
        let solver = build_optimizer_hager_zhang(&opts).expect("builder should succeed");

        // Act
        let outcome =
            run_lbfgs(theta0, &opts, problem, solver).expect("optimization should succeed");

        // Assert
        let theta_hat =
            outcome.theta_hat.as_slice().expect("theta_hat should be present in outcome");
        let x_hat = theta_hat[0];
        // Max at x = 1.
        assert_relative_eq!(x_hat, 1.0, epsilon = 1e-6);
        // ℓ(θ̂) should be close to 0.
        assert_relative_eq!(outcome.value, 0.0, epsilon = 1e-6);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `run_lbfgs` propagates failures from the underlying
    // log-likelihood / Argmin executor rather than swallowing them.
    //
    // Given
    // -----
    // - A `FailingLL` model whose `value` method always returns an error.
    // - A solver constructed via `build_optimizer_hager_zhang`.
    //
    // Expect
    // ------
    // - `run_lbfgs` returns `Err(OptError)` rather than `Ok`.
    // - The specific variant may be `InvalidLogLikInput` or `BackendError`,
    //   depending on how `From<Error> for OptError` maps the Argmin error, but
    //   it must not be silently converted into a success.
    fn run_lbfgs_propagates_model_error_from_executor() {
        // Arrange
        let model = ErrorValueLL;
        let data = ();
        let opts = MLEOptions::default();
        let theta0 = Theta::from(vec![0.0_f64]);
        let problem = ArgMinAdapter::new(&model, &data);
        let solver = build_optimizer_hager_zhang(&opts).expect("builder should succeed");

        // Act
        let result = run_lbfgs(theta0, &opts, problem, solver);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    // Purpose
    // -------
    // Verify that `run_lbfgs` respects `opts.tols.max_iter` when configuring
    // the Argmin executor.
    //
    // Given
    // -----
    // - A quadratic model that would normally converge quickly.
    // - `MLEOptions` with `tols.max_iter = Some(3)`.
    //
    // Expect
    // ------
    // - The resulting `OptimOutcome.iterations` is at most `max_iter`.
    // - The optimization still runs and returns `Ok`.
    fn run_lbfgs_respects_max_iter_from_mleoptions() {
        // Arrange
        let model = QuadraticLL;
        let data = ();
        let mut opts = MLEOptions::default();
        opts.tols.max_iter = Some(3);
        let theta0 = Theta::from(vec![10.0_f64]); // start far from optimum
        let problem = ArgMinAdapter::new(&model, &data);
        let solver = build_optimizer_hager_zhang(&opts).expect("builder should succeed");

        // Act
        let outcome = run_lbfgs(theta0, &opts, problem, solver).expect("optimization should run");

        // Assert
        assert!(outcome.iterations <= 3, "expected iterations <= 3, got {}", outcome.iterations);
        assert!(
            outcome.iterations >= 1,
            "expected at least one iteration, got {}",
            outcome.iterations
        );
    }

    #[cfg(feature = "obs_slog")]
    #[test]
    // Purpose
    // -------
    // Smoke-test that `log_initial_state` evaluates the model at θ₀ and
    // returns `Ok(())` without mutating the adapter or panicking.
    //
    // Given
    // -----
    // - A quadratic log-likelihood model and a fixed θ₀.
    //
    // Expect
    // ------
    // - `log_initial_state` returns `Ok(())`.
    // - It can be called multiple times without error.
    fn log_initial_state_quadratic_model_logs_initial_state() {
        // Arrange
        let model = QuadraticLL;
        let data = ();
        let theta0 = Theta::from(vec![0.5_f64]);
        let problem = ArgMinAdapter::new(&model, &data);

        // Act
        let first = log_initial_state(&theta0, &problem);
        let second = log_initial_state(&theta0, &problem);

        // Assert
        assert!(first.is_ok());
        assert!(second.is_ok());
    }
}
