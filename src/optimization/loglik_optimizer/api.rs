//! loglik_optimizer::api — high-level entry point for MLE.
//!
//! Purpose
//! -------
//! Provide a single user-facing helper, [`maximize`], that runs L-BFGS on any
//! model implementing [`LogLikelihood`]. This module hides the Argmin wiring
//! (`Executor`, `IterState`, observers) and returns a crate-friendly
//! [`OptimOutcome`] struct.
//!
//! Key behaviors
//! -------------
//! - Validate a user’s initial guess via [`LogLikelihood::check`] before
//!   touching the optimizer.
//! - Wrap `(f, data)` in an [`ArgMinAdapter`] so Argmin *minimizes* the cost
//!   `c(θ) = -ℓ(θ)` while the public API remains phrased in terms of
//!   maximizing the log-likelihood `ℓ(θ)`.
//! - Select an L-BFGS solver with either **Hager–Zhang** or
//!   **More–Thuente** line search based on [`LineSearcher`].
//! - Delegate actual execution to [`run_lbfgs`], which configures the
//!   `Executor`, optional observers, maximum iterations, and converts the
//!   Argmin result into [`OptimOutcome`].
//!
//! Invariants & assumptions
//! ------------------------
//! - Callers must provide a well-behaved [`LogLikelihood`] implementation
//!   whose [`LogLikelihood::check`] method can detect obvious input problems
//!   (dimension, finiteness, stationarity) and return an [`OptError`] rather
//!   than panicking.
//! - The initial parameter vector `theta0` is treated as an *unconstrained*
//!   point in parameter space; any mapping from constrained to unconstrained
//!   space is expected to happen in the model or elsewhere.
//! - [`MLEOptions`] is assumed to be internally validated by its constructor;
//!   this module does not re-verify tolerances or memory settings.
//!
//! Conventions
//! -----------
//! - Public APIs talk exclusively in terms of *log-likelihood* values `ℓ(θ)`
//!   and unconstrained parameters `θ`; the negative-cost convention
//!   `c(θ) = -ℓ(θ)` is an internal implementation detail.
//! - The choice of line search is controlled by [`LineSearcher`] on
//!   [`MLEOptions`]; if additional line-searchers are added, this module is
//!   the central dispatch point.
//! - All errors are reported via [`OptResult`] and [`OptError`]; this module
//!   never panics under normal operation.
//!
//! Downstream usage
//! ----------------
//! - Most callers construct a model implementing [`LogLikelihood`], an
//!   initial parameter vector, data, and [`MLEOptions`], then call
//!   [`maximize`] and inspect the returned [`OptimOutcome`].
//! - Higher-level front-ends (e.g., Python bindings) are expected to call
//!   [`maximize`] rather than building Argmin solvers directly.
//! - Other optimizer utilities (builders, finite-difference helpers) are kept
//!   in sibling modules; this module is the “one-stop shop” for MLE runs.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module verify:
//!   - That [`maximize`] dispatches to the correct builder based on
//!     [`LineSearcher`] and succeeds on a simple quadratic model.
//!   - That errors from [`LogLikelihood::check`] are propagated unchanged.
//! - End-to-end integration tests for more complex models live elsewhere and
//!   exercise this module indirectly.
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

/// `maximize` — run L-BFGS MLE for a user-provided log-likelihood.
///
/// Purpose
/// -------
/// Maximize a log-likelihood `ℓ(θ)` by wrapping a [`LogLikelihood`] model
/// in an [`ArgMinAdapter`], selecting an L-BFGS solver with the configured
/// line search, and delegating execution to [`run_lbfgs`]. The function
/// returns a crate-friendly [`OptimOutcome`] containing θ̂, ℓ(θ̂), and
/// metadata.
///
/// Parameters
/// ----------
/// - `f`: `&F`
///   Model implementing [`LogLikelihood`]. Responsible for providing
///   `value`, `grad` (or signaling that gradients are not implemented),
///   and `check`.
/// - `theta0`: [`Theta`]
///   Initial *unconstrained* parameter vector. This value is consumed and
///   passed by value into the optimizer.
/// - `data`: `&F::Data`
///   Borrowed data associated with the model; forwarded to all `value`/`grad`
///   calls. Lifetime must outlive the optimization run.
/// - `opts`: `&MLEOptions`
///   Optimizer configuration, including tolerances, maximum iterations,
///   verbosity, and line-search choice via [`LineSearcher`].
///
/// Returns
/// -------
/// [`OptResult<OptimOutcome>`]
///   - `Ok(OptimOutcome)` on successful completion of the Argmin run and
///     validation of the resulting solution.
///   - `Err(OptError)` if validation, builder construction, or the optimizer
///     itself fails.
///
/// Errors
/// ------
/// - [`OptError`]
///   - Propagates any error returned by [`LogLikelihood::check`] on the
///     initial guess `theta0`.
///   - Propagates builder errors from
///     [`build_optimizer_hager_zhang`] or [`build_optimizer_more_thuente`]
///     when constructing the L-BFGS solver.
///   - Propagates runtime failures from [`run_lbfgs`], including line-search
///     failures and Argmin-internal errors.
///
/// Panics
/// ------
/// - Never panics under normal operation. Any failure path is expressed as an
///   [`OptError`] via [`OptResult`].
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that references `f` and
///   `data` remain valid for the duration of the optimization.
///
/// Notes
/// -----
/// - The function always maximizes the log-likelihood even though the
///   underlying optimizer *minimizes* a cost; the sign flip is handled
///   internally by [`ArgMinAdapter`] and [`run_lbfgs`].
/// - The caller is free to reuse the same model and data with different
///   starting points or [`MLEOptions`] configurations.
///
/// Examples
/// --------
/// ```no_run
/// use ndarray::array;
/// use rust_timeseries::optimization::errors::OptResult;
/// use rust_timeseries::optimization::loglik_optimizer::{
///     maximize,
///     traits::{LogLikelihood, MLEOptions, LineSearcher},
///     types::{Cost, Theta},
/// };
///
/// struct MyLL;
///
/// impl LogLikelihood for MyLL {
///     type Data = ();
///
///     fn value(&self, theta: &Theta, _: &Self::Data) -> OptResult<Cost> {
///         // Simple concave log-likelihood: ℓ(θ) = -||θ||².
///         Ok(-theta.dot(theta))
///     }
///
///     fn check(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<()> {
///         Ok(())
///     }
/// }
///
/// fn main() -> OptResult<()> {
///     let f = MyLL;
///     let theta0 = Theta::from(vec![0.1, -0.2, 0.3]);
///     let data = ();
///
///     let mut opts = MLEOptions::default();
///     opts.line_searcher = LineSearcher::HagerZhang;
///
///     let out = maximize(&f, theta0, &data, &opts)?;
///     println!("θ̂ = {:?}", out.theta_hat);
///     Ok(())
/// }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::errors::OptError;
    use crate::optimization::loglik_optimizer::types::{Cost, Grad, Theta};
    use approx::assert_relative_eq;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - That `maximize` dispatches to the correct line-search builder based on
    //   `MLEOptions.line_searcher`.
    // - That `maximize` successfully maximizes a simple quadratic
    //   log-likelihood and returns a sensible `OptimOutcome`.
    // - That errors from `LogLikelihood::check` are propagated unchanged.
    //
    // They intentionally DO NOT cover:
    // - The internal correctness of the L-BFGS implementation.
    // - Detailed convergence diagnostics, observer wiring, or `run_lbfgs`
    //   internals (those are tested in `run.rs` and related modules).
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

    /// Log-likelihood whose `check` always fails, to verify that `maximize`
    /// propagates the error from `f.check` without touching Argmin.
    struct FailingCheckLL;

    impl LogLikelihood for FailingCheckLL {
        type Data = ();

        fn value(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<Cost> {
            // Should never be called in this test.
            Ok(0.0)
        }

        fn check(&self, _theta: &Theta, _data: &Self::Data) -> OptResult<()> {
            Err(OptError::InvalidLogLikInput { value: 42.0 })
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `maximize` with `LineSearcher::HagerZhang` converges on a
    // simple quadratic model and returns θ̂ ≈ 1 with ℓ(θ̂) close to zero.
    //
    // Given
    // -----
    // - A 1D quadratic log-likelihood with maximizer at θ* = 1.
    // - `MLEOptions` whose `line_searcher` is `HagerZhang`.
    //
    // Expect
    // ------
    // - `maximize` returns `Ok(OptimOutcome)`.
    // - `theta_hat` is close to 1 within a small tolerance.
    // - `value` (ℓ(θ̂)) is close to 0 (the maximum).
    fn maximize_quadratic_hagerzhang_converges() {
        // Arrange
        let f = QuadraticLL;
        let data = ();
        let theta0 = Theta::from(vec![0.0_f64]);

        let mut opts = MLEOptions::default();
        opts.line_searcher = LineSearcher::HagerZhang;

        // Act
        let outcome = maximize(&f, theta0, &data, &opts).expect("optimization should succeed");

        // Assert
        let theta_hat = outcome.theta_hat.as_slice().expect("theta_hat should be present");
        let x_hat = theta_hat[0];

        assert_relative_eq!(x_hat, 1.0, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(outcome.value, 0.0, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `maximize` with `LineSearcher::MoreThuente` also converges
    // on the quadratic model and returns a sensible θ̂ and ℓ(θ̂).
    //
    // Given
    // -----
    // - The same quadratic log-likelihood as above.
    // - `MLEOptions` whose `line_searcher` is `MoreThuente`.
    //
    // Expect
    // ------
    // - `maximize` returns `Ok(OptimOutcome)`.
    // - `theta_hat` is close to 1.
    // - `value` (ℓ(θ̂)) is close to 0.
    fn maximize_quadratic_morethuente_converges() {
        // Arrange
        let f = QuadraticLL;
        let data = ();
        let theta0 = Theta::from(vec![2.5_f64]); // start on the other side

        let mut opts = MLEOptions::default();
        opts.line_searcher = LineSearcher::MoreThuente;

        // Act
        let outcome = maximize(&f, theta0, &data, &opts).expect("optimization should succeed");

        // Assert
        let theta_hat = outcome.theta_hat.as_slice().expect("theta_hat should be present");
        let x_hat = theta_hat[0];

        assert_relative_eq!(x_hat, 1.0, epsilon = 1e-6, max_relative = 1e-6);
        assert_relative_eq!(outcome.value, 0.0, epsilon = 1e-6, max_relative = 1e-6);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `maximize` propagates errors from `LogLikelihood::check`
    // without attempting to build a solver or run optimization.
    //
    // Given
    // -----
    // - A `FailingCheckLL` model whose `check` method always returns
    //   `InvalidLogLikInput`.
    //
    // Expect
    // ------
    // - `maximize` returns `Err(OptError::InvalidLogLikInput { .. })`.
    fn maximize_propagates_check_error() {
        // Arrange
        let f = FailingCheckLL;
        let data = ();
        let theta0 = Theta::from(vec![0.0_f64]);
        let opts = MLEOptions::default();

        // Act
        let result = maximize(&f, theta0, &data, &opts);

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            OptError::InvalidLogLikInput { value } => {
                assert_eq!(value, 42.0);
            }
            other => panic!("expected InvalidLogLikInput, got {:?}", other),
        }
    }
}
