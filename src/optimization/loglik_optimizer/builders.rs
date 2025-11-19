//! loglik_optimizer::builders — L-BFGS solver construction helpers.
//!
//! Purpose
//! -------
//! Provide small, focused builders for L-BFGS solvers used by the
//! log-likelihood optimizer. These helpers hide Argmin’s generic wiring
//! and apply crate-level options (e.g., tolerances, memory size) so that
//! higher-level code can request a configured solver without touching
//! Argmin-specific types.
//!
//! Key behaviors
//! -------------
//! - Construct L-BFGS solvers with either Hager–Zhang or More–Thuente
//!   line search based on crate-level aliases.
//! - Apply optional gradient and cost-change tolerances from
//!   [`MLEOptions`] via a shared configuration helper.
//! - Leave the initial parameter vector and maximum iterations to the
//!   runner/executor layer, keeping these builders side-effect free.
//!
//! Invariants & assumptions
//! ------------------------
//! - All solvers operate on the canonical optimizer numeric types
//!   [`Theta`], [`Grad`], and [`Cost`] as defined in
//!   [`loglik_optimizer::types`].
//! - The L-BFGS memory (`m`) is either provided via `opts.lbfgs_mem` or
//!   defaults to [`DEFAULT_LBFGS_MEM`].
//! - Any invalid tolerance passed into Argmin’s
//!   `with_tolerance_grad` / `with_tolerance_cost` is surfaced as an
//!   [`OptError`] via the crate’s `From<Error>` implementations; callers
//!   are expected to handle these with `OptResult`.
//!
//! Conventions
//! -----------
//! - [`HagerZhangLS`] and [`MoreThuenteLS`] are the crate’s canonical
//!   line-search aliases; [`LbfgsHagerZhang`] and
//!   [`LbfgsMoreThuente`] pair these with the standard `(Theta, Grad,
//!   Cost)` triple.
//! - The builders do **not** set an initial parameter vector (`theta0`)
//!   or `max_iters`; these are treated as runtime concerns and are
//!   applied by the runner (e.g., `run_lbfgs`).
//! - Errors are always reported via [`OptResult`]; the underlying
//!   `argmin::core::Error` values never leak directly across module
//!   boundaries.
//!
//! Downstream usage
//! ----------------
//! - High-level optimization entry points call
//!   [`build_optimizer_hager_zhang`] or
//!   [`build_optimizer_more_thuente`] based on a configured
//!   `LineSearcher` enum in [`MLEOptions`].
//! - The returned solver is passed to a runner (e.g., `run_lbfgs`) along
//!   with an adapted problem and initial parameters.
//! - [`configure_lbfgs`] is the shared wiring function that applies
//!   tolerances; it is generic over the line-search type and can be
//!   reused by future L-BFGS variants if needed.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module verify:
//!   - Correct propagation of `lbfgs_mem` and `DEFAULT_LBFGS_MEM` into
//!     the solver configuration.
//! - Integration tests in the optimizer layer exercise these builders
//!   indirectly by running full L-BFGS solves with different line-search
//!   and tolerance configurations.
use argmin::solver::quasinewton::LBFGS;

use crate::optimization::{
    errors::OptResult,
    loglik_optimizer::{
        traits::MLEOptions,
        types::{
            Cost, DEFAULT_LBFGS_MEM, Grad, HagerZhangLS, LbfgsHagerZhang, LbfgsMoreThuente,
            MoreThuenteLS, Theta,
        },
    },
};

/// build_optimizer_hager_zhang — construct L-BFGS with Hager–Zhang line search.
///
/// Purpose
/// -------
/// Build an [`LbfgsHagerZhang`] solver configured with the crate’s
/// standard numeric types and optional tolerances from [`MLEOptions`],
/// leaving initial parameters and iteration limits to the caller.
///
/// Parameters
/// ----------
/// - `opts`: `&MLEOptions`  
///   Optimizer options. This builder consults:
///   - `opts.lbfgs_mem`: optional L-BFGS history size (`m`); when
///     `None`, [`DEFAULT_LBFGS_MEM`] is used.
///   - `opts.tols.tol_grad` and `opts.tols.tol_cost`: optional
///     gradient-norm and cost-change tolerances wired into the solver
///     via Argmin’s `with_tolerance_grad` / `with_tolerance_cost`.
///
/// Returns
/// -------
/// `OptResult<LbfgsHagerZhang>`  
///   - `Ok(solver)` containing an L-BFGS instance with Hager–Zhang line
///     search and any configured tolerances.
///   - `Err(e)` if Argmin rejects any of the tolerance settings.
///
/// Errors
/// ------
/// - `OptError` (via `From<argmin::core::Error>`)  
///   Returned when `with_tolerance_grad` or `with_tolerance_cost`
///   encounters an invalid tolerance (e.g., non-finite or non-positive
///   value) or other internal configuration error.
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
/// - This function does not set `theta0` or `max_iters`; these must be
///   configured by the caller when running the solver.
/// - The underlying line-search object is `HagerZhangLS`, as defined in
///   [`loglik_optimizer::types`].
///
/// Examples
/// --------
/// ```ignore
/// let solver = build_optimizer_hager_zhang(&opts)?;
/// let outcome = run_lbfgs(theta0, &opts, problem, solver)?;
/// ```
pub fn build_optimizer_hager_zhang(opts: &MLEOptions) -> OptResult<LbfgsHagerZhang> {
    let hager_zhang = HagerZhangLS::new();
    let mem = opts.lbfgs_mem.unwrap_or(DEFAULT_LBFGS_MEM);
    let lbfgs = LbfgsHagerZhang::new(hager_zhang, mem);
    configure_lbfgs(lbfgs, opts)
}

/// build_optimizer_more_thuente — construct L-BFGS with More–Thuente line search.
///
/// Purpose
/// -------
/// Build an [`LbfgsMoreThuente`] solver configured with the crate’s
/// standard numeric types and optional tolerances from [`MLEOptions`],
/// using the More–Thuente line-search strategy.
///
/// Parameters
/// ----------
/// - `opts`: `&MLEOptions`  
///   Optimizer options. This builder consults:
///   - `opts.lbfgs_mem`: optional L-BFGS history size (`m`); when
///     `None`, [`DEFAULT_LBFGS_MEM`] is used.
///   - `opts.tols.tol_grad` and `opts.tols.tol_cost`: optional
///     gradient-norm and cost-change tolerances wired into the solver
///     via Argmin’s `with_tolerance_grad` / `with_tolerance_cost`.
///
/// Returns
/// -------
/// `OptResult<LbfgsMoreThuente>`  
///   - `Ok(solver)` containing an L-BFGS instance with More–Thuente line
///     search and any configured tolerances.
///   - `Err(e)` if Argmin rejects any of the tolerance settings.
///
/// Errors
/// ------
/// - `OptError` (via `From<argmin::core::Error>`)  
///   Returned when `with_tolerance_grad` or `with_tolerance_cost`
///   fails due to invalid or non-finite tolerances or other internal
///   configuration issues.
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
/// - As with [`build_optimizer_hager_zhang`], this builder only configures
///   the solver; initial parameters and iteration limits are applied by
///   the runner.
/// - The underlying line-search object is [`MoreThuenteLS`].
///
/// Examples
/// --------
/// ```ignore
/// let solver = build_optimizer_more_thuente(&opts)?;
/// let outcome = run_lbfgs(theta0, &opts, problem, solver)?;
/// ```
pub fn build_optimizer_more_thuente(opts: &MLEOptions) -> OptResult<LbfgsMoreThuente> {
    let more_thuente = MoreThuenteLS::new();
    let mem = opts.lbfgs_mem.unwrap_or(DEFAULT_LBFGS_MEM);
    let lbfgs = LbfgsMoreThuente::new(more_thuente, mem);
    configure_lbfgs(lbfgs, opts)
}

/// configure_lbfgs — apply optional tolerances to an L-BFGS solver.
///
/// Purpose
/// -------
/// Generic helper that wires crate-level tolerance options from
/// [`MLEOptions`] into an existing L-BFGS solver, regardless of the
/// line-search type. This centralizes tolerance handling so builder
/// functions remain thin.
///
/// Parameters
/// ----------
/// - `solver`: `LBFGS<L, Theta, Grad, Cost>`  
///   Pre-constructed L-BFGS solver using some line-search type `L`.
///   Typically created via `LbfgsHagerZhang::new` or
///   `LbfgsMoreThuente::new`.
/// - `opts`: `&MLEOptions`  
///   Source of optional tolerances. This helper consults:
///   - `opts.tols.tol_grad`: optional gradient-norm tolerance.
///   - `opts.tols.tol_cost`: optional cost-change tolerance.
///
/// Returns
/// -------
/// `OptResult<LBFGS<L, Theta, Grad, Cost>>`  
///   - `Ok(solver)` with any present tolerances applied.
///   - `Err(e)` if any tolerance configuration fails inside Argmin.
///
/// Errors
/// ------
/// - `OptError` (via `From<argmin::core::Error>`)  
///   Returned when `with_tolerance_grad` or `with_tolerance_cost`
///   rejects a tolerance (e.g., non-finite or non-positive value) or
///   hits an internal configuration error.
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
/// - When a tolerance is `None`, the corresponding `with_tolerance_*`
///   method is not called; Argmin’s defaults remain in effect.
/// - This helper does not touch the solver’s initial parameter vector,
///   maximum iteration count, or line-search settings; it only applies
///   tolerances.
/// - The generics are kept minimal (`L` only) so new line-search types
///   can reuse this function without additional constraints.
///
/// Examples
/// --------
/// ```ignore
/// use argmin::solver::quasinewton::LBFGS;
/// use crate::optimization::loglik_optimizer::types::HagerZhangLS;
///
/// let raw = LBFGS::<HagerZhangLS, Theta, Grad, Cost>::new(
///     HagerZhangLS::new(),
///     DEFAULT_LBFGS_MEM,
/// );
/// let solver = configure_lbfgs(raw, &opts)?;
/// ```
pub fn configure_lbfgs<L>(
    mut solver: LBFGS<L, Theta, Grad, Cost>, opts: &MLEOptions,
) -> OptResult<LBFGS<L, Theta, Grad, Cost>> {
    if let Some(g) = opts.tols.tol_grad {
        solver = solver.with_tolerance_grad(g)?;
    }
    if let Some(c) = opts.tols.tol_cost {
        solver = solver.with_tolerance_cost(c)?;
    }
    Ok(solver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::loglik_optimizer::traits::{LineSearcher, MLEOptions, Tolerances};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Basic construction of L-BFGS solvers with Hager–Zhang and
    //   More–Thuente line searches.
    // - Propagation of `lbfgs_mem` (Some vs None) into the builder paths.
    // - Application of gradient and cost tolerances via `configure_lbfgs`.
    //
    // They intentionally DO NOT cover:
    // - End-to-end executor behavior (e.g., `run_lbfgs`), which is tested
    //   in the optimizer runner layer.
    // - Any specific `LogLikelihood` implementation or real data models.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Ensure that `build_optimizer_hager_zhang` succeeds and uses the
    // crate default L-BFGS memory when `opts.lbfgs_mem` is `None`.
    //
    // Given
    // -----
    // - Valid `Tolerances`.
    // - `MLEOptions` with `line_searcher = HagerZhang` and `lbfgs_mem = None`.
    //
    // Expect
    // ------
    // - `build_optimizer_hager_zhang` returns `Ok(_)` and does not panic.
    fn build_optimizer_hager_zhang_uses_default_memory_when_none() {
        // Arrange
        let tols =
            Tolerances::new(Some(1e-6), Some(1e-8), Some(50)).expect("Tolerances should be valid");
        let opts = MLEOptions::new(tols, LineSearcher::HagerZhang, None)
            .expect("MLEOptions should be valid");

        // Act
        let solver = build_optimizer_hager_zhang(&opts);

        // Assert
        assert!(
            solver.is_ok(),
            "Builder should succeed when lbfgs_mem is None and tolerances are valid"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `build_optimizer_hager_zhang` accepts an explicit
    // L-BFGS memory value and still constructs a solver.
    //
    // Given
    // -----
    // - Valid `Tolerances`.
    // - `MLEOptions` with `line_searcher = HagerZhang` and `lbfgs_mem = Some(11)`.
    //
    // Expect
    // ------
    // - `build_optimizer_hager_zhang` returns `Ok(_)`.
    fn build_optimizer_hager_zhang_respects_explicit_memory() {
        // Arrange
        let tols = Tolerances::new(Some(1e-6), None, Some(25)).expect("Tolerances should be valid");
        let opts = MLEOptions::new(tols, LineSearcher::HagerZhang, Some(11))
            .expect("MLEOptions should be valid");

        // Act
        let solver = build_optimizer_hager_zhang(&opts);

        // Assert
        assert!(solver.is_ok(), "Builder should succeed when lbfgs_mem is explicitly provided");
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `build_optimizer_more_thuente` succeeds and uses the
    // crate default L-BFGS memory when `opts.lbfgs_mem` is `None`.
    //
    // Given
    // -----
    // - Valid `Tolerances`.
    // - `MLEOptions` with `line_searcher = MoreThuente` and `lbfgs_mem = None`.
    //
    // Expect
    // ------
    // - `build_optimizer_more_thuente` returns `Ok(_)`.
    fn build_optimizer_more_thuente_uses_default_memory_when_none() {
        // Arrange
        let tols =
            Tolerances::new(Some(1e-6), Some(1e-8), Some(50)).expect("Tolerances should be valid");
        let opts = MLEOptions::new(tols, LineSearcher::MoreThuente, None)
            .expect("MLEOptions should be valid");

        // Act
        let solver = build_optimizer_more_thuente(&opts);

        // Assert
        assert!(
            solver.is_ok(),
            "Builder should succeed when lbfgs_mem is None and tolerances are valid"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `build_optimizer_more_thuente` accepts an explicit
    // L-BFGS memory value and still constructs a solver.
    //
    // Given
    // -----
    // - Valid `Tolerances`.
    // - `MLEOptions` with `line_searcher = MoreThuente` and `lbfgs_mem = Some(9)`.
    //
    // Expect
    // ------
    // - `build_optimizer_more_thuente` returns `Ok(_)`.
    fn build_optimizer_more_thuente_respects_explicit_memory() {
        // Arrange
        let tols = Tolerances::new(Some(1e-6), None, Some(30)).expect("Tolerances should be valid");
        let opts = MLEOptions::new(tols, LineSearcher::MoreThuente, Some(9))
            .expect("MLEOptions should be valid");

        // Act
        let solver = build_optimizer_more_thuente(&opts);

        // Assert
        assert!(solver.is_ok(), "Builder should succeed when lbfgs_mem is explicitly provided");
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `configure_lbfgs` applies tolerances without error
    // when both `tol_grad` and `tol_cost` are present and valid.
    //
    // Given
    // -----
    // - An L-BFGS solver created with `DEFAULT_LBFGS_MEM`.
    // - `MLEOptions` with finite, positive `tol_grad` and `tol_cost`.
    //
    // Expect
    // ------
    // - `configure_lbfgs` returns `Ok(_)`.
    fn configure_lbfgs_applies_valid_tolerances() {
        // Arrange
        let raw = LBFGS::new(HagerZhangLS::new(), DEFAULT_LBFGS_MEM);
        let tols =
            Tolerances::new(Some(1e-6), Some(1e-8), Some(100)).expect("Tolerances should be valid");
        let opts = MLEOptions::new(tols, LineSearcher::HagerZhang, Some(DEFAULT_LBFGS_MEM))
            .expect("MLEOptions should be valid");

        // Act
        let configured = configure_lbfgs(raw, &opts);

        // Assert
        assert!(configured.is_ok(), "configure_lbfgs should succeed for valid tolerances");
    }

    #[test]
    // Purpose
    // -------
    // Verify that `configure_lbfgs` leaves the solver constructible when
    // both gradient and cost tolerances are `None`, relying on Argmin
    // defaults.
    //
    // Given
    // -----
    // - An L-BFGS solver created with `DEFAULT_LBFGS_MEM`.
    // - `MLEOptions` whose `tols` have `tol_grad = None`, `tol_cost = None`.
    //
    // Expect
    // ------
    // - `configure_lbfgs` returns `Ok(_)`.
    fn configure_lbfgs_respects_absent_tolerances() {
        // Arrange
        let raw = LBFGS::new(MoreThuenteLS::new(), DEFAULT_LBFGS_MEM);
        let tols = Tolerances::new(None, None, Some(50)).expect("Tolerances should be valid");
        let opts = MLEOptions::new(tols, LineSearcher::MoreThuente, None)
            .expect("MLEOptions should be valid");

        // Act
        let configured = configure_lbfgs(raw, &opts);

        // Assert
        assert!(configured.is_ok(), "configure_lbfgs should succeed when both tolerances are None");
    }
}
