//! ACD parameter workspace — zero-copy buffer for ω, α, β, and slack.
//!
//! Purpose
//! -------
//! Provide a mutable, zero-copy workspace for ACD(p, q) parameters during
//! optimization. The workspace borrows caller-owned α/β buffers and exposes
//! an allocation-free mapping from unconstrained optimizer parameters `θ` to
//! model-space parameters `(ω, α, β, slack)`.
//!
//! Key behaviors
//! -------------
//! - Construct a workspace over caller-owned α/β buffers with lengths tied to
//!   an [`ACDShape`] descriptor.
//! - Map `θ` into strictly positive `ω` via a numerically stable softplus
//!   transform and into α/β/slack via a max-shift softmax scaled by
//!   [`STATIONARITY_MARGIN`].
//!
//! Invariants & assumptions
//! ------------------------
//! - ACD orders follow the Engle–Russell convention: `q` = number of α terms
//!   (lags on durations τ), `p` = number of β terms (lags on conditional
//!   means ψ).
//! - After a successful call to [`WorkSpace::update`]:
//!   - `alpha.len() == q` and `beta.len() == p`,
//!   - all α/β coordinates are finite and non-negative,
//!   - `ω` is finite and strictly positive, and
//!   - `sum(α) + sum(β) + slack ≈ 1 − STATIONARITY_MARGIN` within a small
//!     numerical tolerance.
//! - The `theta` input must have length `1 + q + p`; mismatches are surfaced
//!   as [`ParamError::ThetaLengthMismatch`].
//!
//! Conventions
//! -----------
//! - Indexing is 0-based and uses standard `ndarray` semantics for views and
//!   slicing.
//! - All validation of ω, α, β, and the stationarity margin is delegated to
//!   `duration::core::validation` helpers, which return [`ParamResult`];
//!   this module does not panic on invalid *inputs*.
//! - `WorkSpace` is purely numeric; it performs no I/O and emits no logging.
//!
//! Downstream usage
//! ----------------
//! - Construct a `WorkSpace` over preallocated α/β buffers before launching
//!   an optimizer.
//! - On each optimizer step, call [`WorkSpace::update`] with the
//!   current `θ` to refresh `omega`, `alpha`, `beta`, `slack` in place and
//!   then evaluate the likelihood elsewhere.
//! - After convergence, read out `omega`, `alpha`, `beta`, `slack`, and
//!   [`WorkSpace::uncond_mean`] to build an owned parameter object (e.g.,
//!   `ACDParams`) for storage or downstream use.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module cover:
//!   - length validation in [`WorkSpace::new`] and
//!     [`WorkSpace::update`],
//!   - mapping from simple `θ` inputs into α/β/ω/slack that respect the
//!     documented invariants, and
//!   - behavior of [`WorkSpace::uncond_mean`] under well-formed parameters.
//! - Higher-level behavior (likelihood evaluation, optimizer convergence,
//!   Python bindings) is exercised in integration tests and Python-level
//!   tests, not here.
use crate::{
    duration::{
        core::{
            shape::ACDShape,
            validation::{
                validate_alpha, validate_alpha_beta_lengths, validate_beta, validate_omega,
                validate_stationarity_and_slack,
            },
        },
        errors::{ParamError, ParamResult},
    },
    optimization::numerical_stability::transformations::{safe_softmax, safe_softplus},
};
use ndarray::{ArrayView1, ArrayViewMut1, s};

/// WorkSpace — zero-copy buffer for ACD(p, q) parameters during optimization.
///
/// Purpose
/// -------
/// Represent a mutable, zero-copy workspace for ACD(p, q) parameters. A
/// `WorkSpace` borrows caller-owned α/β buffers and stores scalar `omega` and
/// `slack`, allowing optimization routines to update parameters in place
/// without heap allocation.
///
/// Key behaviors
/// -------------
/// - Holds `alpha` and `beta` as mutable `ndarray` views into external
///   storage, rather than owning their memory.
/// - Overwrites `alpha`, `beta`, `omega`, and `slack` on each call to
///   [`WorkSpace::update`] based on the current optimizer parameter
///   vector `θ`.
/// - Exposes [`WorkSpace::uncond_mean`] as a convenience method for the
///   implied unconditional duration mean under unit-mean innovations.
///
/// Parameters
/// ----------
/// Constructed via [`WorkSpace::new`]:
/// - `alpha`: `ArrayViewMut1<'a, f64>`
///   Borrowed buffer for α coefficients of length `q`.
/// - `beta`: `ArrayViewMut1<'a, f64>`
///   Borrowed buffer for β coefficients of length `p`.
/// - `shape`: `&ACDShape`
///   Shape descriptor carrying ACD orders `(p, q)` used for length checks.
///
/// Fields
/// ------
/// - `alpha`: `ArrayViewMut1<'a, f64>`
///   Mutable view over the α coefficients. Updated in place by
///   [`WorkSpace::update`].
/// - `beta`: `ArrayViewMut1<'a, f64>`
///   Mutable view over the β coefficients. Updated in place by
///   [`WorkSpace::update`].
/// - `omega`: `f64`
///   Baseline ACD parameter ω, set by [`WorkSpace::update`] via a
///   numerically stable softplus transform.
/// - `slack`: `f64`
///   Non-negative slack term chosen so that `sum(α) + sum(β) + slack` lies
///   close to `1 − STATIONARITY_MARGIN`.
///
/// Invariants
/// ----------
/// - After a successful call to [`WorkSpace::update`]:
///   - all entries of `alpha` and `beta` are finite and ≥ 0,
///   - `omega` is finite and strictly > 0, and
///   - `sum(α) + sum(β) + slack` satisfies the stationarity margin constraint
///     within tolerance.
/// - The lengths of `alpha` and `beta` remain equal to `shape.q` and `shape.p`
///   respectively; these are checked by [`WorkSpace::new`].
///
/// Performance
/// -----------
/// - Constructing a `WorkSpace` does not allocate; it only stores views and
///   performs length validation.
/// - [`WorkSpace::update`] operates entirely on borrowed buffers and
///   uses a max-shift softmax to remain numerically stable without heap
///   allocation.
///
/// Notes
/// -----
/// - The lifetime `'a` ties `WorkSpace` to the caller-owned α/β buffers; the
///   workspace must not outlive those buffers.
/// - This type is intended as an internal optimization artifact; public
///   high-level APIs are expected to expose owned parameter types (such as
///   `ACDParams`) rather than `WorkSpace` directly.
#[derive(Debug)]
pub struct WorkSpace<'a> {
    pub alpha: ArrayViewMut1<'a, f64>,
    pub beta: ArrayViewMut1<'a, f64>,
    pub omega: f64,
    pub slack: f64,
}

impl<'a> WorkSpace<'a> {
    /// Construct a `WorkSpace` over caller-owned α/β buffers.
    ///
    /// Parameters
    /// ----------
    /// - `alpha`: `ArrayViewMut1<'a, f64>`
    ///   Mutable view into the caller's α buffer. Must have length `shape.q`.
    /// - `beta`: `ArrayViewMut1<'a, f64>`
    ///   Mutable view into the caller's β buffer. Must have length `shape.p`.
    /// - `shape`: `&ACDShape`
    ///   Shape descriptor carrying the ACD orders `(p, q)` used for length checks.
    ///
    /// Returns
    /// -------
    /// `ParamResult<WorkSpace<'a>>`
    ///   - `Ok(WorkSpace)` if `alpha.len() == shape.q` and `beta.len() == shape.p`.
    ///   - `Err(ParamError)` if a length mismatch is detected.
    ///
    /// Errors
    /// ------
    /// - `ParamError::AlphaLengthMismatch`
    ///   Returned when `alpha.len() != shape.q`.
    /// - `ParamError::BetaLengthMismatch`
    ///   Returned when `beta.len() != shape.p`.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Notes
    /// -----
    /// - This constructor validates **only lengths**; it does not inspect the
    ///   numeric contents of `alpha` or `beta`, and it initializes `omega` and
    ///   `slack` to `0.0`.
    /// - Parameter semantics (positivity, stationarity margin) are enforced later
    ///   by [`WorkSpace::update`].
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::shape::ACDShape;
    /// # use rust_timeseries::duration::core::workspace::WorkSpace;
    /// # use rust_timeseries::duration::errors::ParamResult;
    /// # use ndarray::Array1;
    /// # fn example(shape: &ACDShape) -> ParamResult<()> {
    /// #   let mut alpha_buf = Array1::<f64>::zeros(shape.q);
    /// #   let mut beta_buf = Array1::<f64>::zeros(shape.p);
    /// let ws = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), shape)?;
    /// #   let _ = ws;
    /// #   Ok(())
    /// # }
    /// ```
    pub fn new(
        alpha: ArrayViewMut1<'a, f64>, beta: ArrayViewMut1<'a, f64>, shape: &ACDShape,
    ) -> ParamResult<Self> {
        validate_alpha_beta_lengths(alpha.view(), beta.view(), shape.q, shape.p)?;
        Ok(WorkSpace { alpha, beta, omega: 0.0, slack: 0.0 })
    }

    /// Overwrite the workspace from an unconstrained optimizer parameter vector `θ`.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `ArrayView1<'_, f64>`
    ///   Unconstrained optimizer parameter vector of length `1 + q + p`, where:
    ///   - `theta[0]` is the log-parameter mapped to `omega` via softplus, and
    ///   - `theta[1..]` are logits for the α/β weights.
    ///
    /// Returns
    /// -------
    /// `ParamResult<()>`
    ///   - `Ok(())` if `theta` has the expected length and the mapped
    ///     `(omega, alpha, beta, slack)` satisfy all documented invariants.
    ///   - `Err(ParamError)` if length checks, positivity, or stationarity
    ///     constraints fail.
    ///
    /// Errors
    /// ------
    /// - `ParamError::ThetaLengthMismatch`
    ///   Returned if `theta.len() != 1 + shape.q + shape.p`.
    /// - `ParamError::InvalidOmega`
    ///   Returned if the softplus-mapped `omega` is not finite or ≤ 0.
    /// - `ParamError::InvalidAlpha` / `ParamError::InvalidBeta`
    ///   Returned if any α/β coordinate is negative or non-finite after the
    ///   softmax mapping.
    /// - `ParamError::InvalidSlack` / `ParamError::StationarityViolated`
    ///   Returned if the `slack` is negative/non-finite or if
    ///   `sum(α) + sum(β) + slack` deviates from `1 − STATIONARITY_MARGIN`
    ///   beyond the internal tolerance.
    ///
    /// Panics
    /// ------
    /// - Never panics on invalid user inputs; all such conditions are surfaced as
    ///   [`ParamError`] variants.
    ///
    /// Notes
    /// -----
    /// - The mapping uses:
    ///   - `omega = safe_softplus(theta[0])`, ensuring `omega > 0` without
    ///     overflow/underflow.
    ///   - a three-pass max-shift softmax over `theta[1..]` to produce α/β/slack weights
    ///     scaled by `1 − STATIONARITY_MARGIN`, written directly into the borrowed
    ///     `alpha` and `beta` buffers and slack is returned.
    /// - After safe_softmax, `slack` is checked by
    ///   [`validate_stationarity_and_slack`].
    /// - This function performs **no heap allocation**; all operations are carried
    ///   out on the borrowed views and scalars held in the workspace.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::workspace::WorkSpace;
    /// # use ndarray::array;
    /// # fn example() {
    /// #   let mut alpha_buf = array![0.0_f64];
    /// #   let mut beta_buf = array![0.0_f64];
    /// #   let mut ws = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut())).unwrap();
    /// let theta = array![0.0_f64, 0.1_f64, -0.2_f64];
    /// ws.update(theta.view(), &shape).unwrap();
    /// #   let _ = (ws.omega, ws.slack);
    /// # }
    /// ```
    pub fn update<'b>(&mut self, theta: ArrayView1<'b, f64>) -> ParamResult<()> {
        let q = self.alpha.len();
        let p = self.beta.len();
        let n = theta.len();
        if theta.len() != p + q + 1 {
            return Err(ParamError::ThetaLengthMismatch { expected: p + q + 1, actual: n });
        }
        let omega = safe_softplus(theta[0]);
        validate_omega(omega)?;
        self.omega = omega;
        self.slack =
            safe_softmax(self.alpha.view_mut(), self.beta.view_mut(), &theta.slice(s![1..]));
        validate_alpha(self.alpha.view(), q)?;
        validate_beta(self.beta.view(), p)?;
        validate_stationarity_and_slack(self.alpha.view(), self.beta.view(), self.slack)?;
        Ok(())
    }

    /// Implied unconditional mean of durations under the current parameters.
    ///
    /// Parameters
    /// ----------
    /// - `&self`: `&WorkSpace<'_>`
    ///   Workspace containing a valid set of parameters `(omega, alpha, beta)`
    ///   produced by a prior call to [`WorkSpace::update`].
    ///
    /// Returns
    /// -------
    /// `f64`
    ///   The long-run mean duration
    ///   `μ = omega / (1 − sum(α) − sum(β))`, assuming unit-mean innovations.
    ///
    /// Errors
    /// ------
    /// - This method does not return a `Result`; it assumes that the workspace
    ///   parameters satisfy the stationarity margin enforced by
    ///   [`WorkSpace::update`]. If the invariants are manually violated
    ///   (e.g., by mutating fields directly), the denominator may become
    ///   non-positive and the returned value will be undefined or nonsensical.
    ///
    /// Panics
    /// ------
    /// - Does not explicitly panic, but dividing by a non-positive denominator
    ///   can lead to infinities or NaNs if the workspace invariants are not
    ///   respected.
    ///
    /// Notes
    /// -----
    /// - Under the stationarity margin constraint,
    ///   `1 − sum(α) − sum(β) = STATIONARITY_MARGIN + slack`, so the denominator
    ///   is strictly positive after a successful [`WorkSpace::update`].
    /// - This quantity is often useful for diagnostics and for checking that
    ///   fitted parameters imply reasonable long-run durations in the units of
    ///   the input data.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::shape::ACDShape;
    /// # use rust_timeseries::duration::core::workspace::WorkSpace;
    /// # use ndarray::array;
    /// # fn example() {
    /// #   let shape = ACDShape { p: 1, q: 1 };
    /// #   let mut alpha_buf = array![0.0_f64];
    /// #   let mut beta_buf = array![0.0_f64];
    /// #   let mut ws = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape).unwrap();
    /// #   let theta = array![0.0_f64, 0.1_f64, -0.2_f64];
    /// #   ws.update(theta.view(), &shape).unwrap();
    /// let mu = ws.uncond_mean();
    /// #   let _ = mu;
    /// # }
    /// ```
    pub fn uncond_mean(&self) -> f64 {
        let sum_alpha = self.alpha.sum();
        let sum_beta = self.beta.sum();
        self.omega / (1.0 - sum_alpha - sum_beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::numerical_stability::transformations::STATIONARITY_MARGIN;
    use ndarray::{Array1, array};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Length validation in `WorkSpace::new` and `WorkSpace::update`.
    // - Mapping from simple `θ` inputs into (ω, α, β, slack) that respect the
    //   documented invariants (positivity, simplex mass).
    // - Behavior of `WorkSpace::uncond_mean` under well-formed parameters.
    //
    // They intentionally DO NOT cover:
    // - Edge-case numerics for `safe_softplus` / `safe_softmax` (tested elsewhere).
    // - Higher-level ψ–recursion, likelihood, or gradient behavior.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `WorkSpace::new` accepts caller-owned α/β buffers whose
    // lengths match the `(p, q)` orders encoded in `ACDShape`.
    //
    // Given
    // -----
    // - A shape with p = 2, q = 1.
    // - α-buffer of length q and β-buffer of length p.
    //
    // Expect
    // ------
    // - `WorkSpace::new` returns `Ok(WorkSpace)`.
    // - The workspace views report the same lengths.
    // - `omega` and `slack` are initialized to 0.0.
    fn workspace_new_accepts_buffers_with_lengths_matching_shape() {
        // Arrange
        let shape = ACDShape { p: 2, q: 1 };
        let mut alpha_buf = Array1::<f64>::zeros(shape.q);
        let mut beta_buf = Array1::<f64>::zeros(shape.p);

        // Act
        let ws_result = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape);

        // Assert
        assert!(
            ws_result.is_ok(),
            "expected WorkSpace::new to succeed for matching buffer lengths"
        );
        let ws = ws_result.unwrap();
        assert_eq!(ws.alpha.len(), shape.q);
        assert_eq!(ws.beta.len(), shape.p);
        assert_eq!(ws.omega, 0.0);
        assert_eq!(ws.slack, 0.0);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `WorkSpace::new` rejects an α-buffer whose length does not
    // match the expected `q` from `ACDShape`.
    //
    // Given
    // -----
    // - A shape with p = 1, q = 2.
    // - α-buffer of length q + 1 (3) and β-buffer of length p (1).
    //
    // Expect
    // ------
    // - `WorkSpace::new` returns an error.
    // - The debug representation of the error mentions `AlphaLengthMismatch`.
    fn workspace_new_rejects_alpha_length_mismatch() {
        // Arrange
        let shape = ACDShape { p: 1, q: 2 };
        let mut alpha_buf = Array1::<f64>::zeros(shape.q + 1); // too long
        let mut beta_buf = Array1::<f64>::zeros(shape.p);

        // Act
        let err = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape)
            .expect_err("expected WorkSpace::new to fail on α length mismatch");

        // Assert
        let debug = format!("{err:?}");
        assert!(
            debug.contains("AlphaLengthMismatch"),
            "expected AlphaLengthMismatch in error, got {debug}"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `WorkSpace::new` rejects a β-buffer whose length does not
    // match the expected `p` from `ACDShape`.
    //
    // Given
    // -----
    // - A shape with p = 2, q = 1.
    // - α-buffer of length q (1) and β-buffer of length p + 1 (3).
    //
    // Expect
    // ------
    // - `WorkSpace::new` returns an error.
    // - The debug representation of the error mentions `BetaLengthMismatch`.
    fn workspace_new_rejects_beta_length_mismatch() {
        // Arrange
        let shape = ACDShape { p: 2, q: 1 };
        let mut alpha_buf = Array1::<f64>::zeros(shape.q);
        let mut beta_buf = Array1::<f64>::zeros(shape.p + 1); // too long

        // Act
        let err = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape)
            .expect_err("expected WorkSpace::new to fail on β length mismatch");

        // Assert
        let debug = format!("{err:?}");
        assert!(
            debug.contains("BetaLengthMismatch"),
            "expected BetaLengthMismatch in error, got {debug}"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `WorkSpace::update` rejects a θ vector whose length does not
    // equal `1 + q + p`, and that it leaves the existing workspace state
    // unchanged on this error path.
    //
    // Given
    // -----
    // - A shape with p = 2, q = 1 (so expected θ length = 4).
    // - A workspace with sentinel values in ω, slack, α, and β.
    // - A θ vector of length 2.
    //
    // Expect
    // ------
    // - `update` returns `Err(ParamError::ThetaLengthMismatch)` with the
    //   expected/actual lengths.
    // - `omega`, `slack`, `alpha`, and `beta` are unchanged.
    fn workspace_update_rejects_theta_length_mismatch_and_preserves_state() {
        // Arrange
        let shape = ACDShape { p: 2, q: 1 };
        let mut alpha_buf = Array1::<f64>::zeros(shape.q);
        let mut beta_buf = Array1::<f64>::zeros(shape.p);
        let mut ws = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape).unwrap();

        // Sentinel values
        ws.omega = 123.0;
        ws.slack = 0.5;
        ws.alpha.fill(1.0);
        ws.beta.fill(2.0);

        let theta_too_short = array![0.0_f64, 0.1_f64]; // len = 2, expected = 4

        // Act
        let result = ws.update(theta_too_short.view());

        // Assert
        match result {
            Err(ParamError::ThetaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 1 + shape.p + shape.q);
                assert_eq!(actual, theta_too_short.len());
            }
            Ok(()) => panic!("expected ThetaLengthMismatch error for short θ, got Ok(())"),
            Err(other) => {
                panic!("expected ThetaLengthMismatch error, got {other:?}");
            }
        }

        // State should be unchanged
        assert_eq!(ws.omega, 123.0);
        assert_eq!(ws.slack, 0.5);
        assert!(ws.alpha.iter().all(|&a| a == 1.0));
        assert!(ws.beta.iter().all(|&b| b == 2.0));
    }

    #[test]
    // Purpose
    // -------
    // Verify that `WorkSpace::update` maps a simple θ vector into:
    // - strictly positive, finite ω,
    // - non-negative, finite α/β entries,
    // - non-negative, finite slack,
    // and that (α, β, slack) lie on a scaled simplex with total mass
    // approximately equal to `1 − STATIONARITY_MARGIN`.
    //
    // Given
    // -----
    // - A shape with p = 1, q = 2 (expected θ length = 4).
    // - Zero-initialized α/β buffers.
    // - A small θ vector with mixed signs.
    //
    // Expect
    // ------
    // - `update` succeeds.
    // - ω > 0 and finite.
    // - αᵢ, βⱼ, slack ≥ 0 and finite.
    // - sum(α) + sum(β) + slack ≈ 1 − STATIONARITY_MARGIN.
    fn workspace_update_maps_theta_into_positive_omega_and_valid_simplex() {
        // Arrange
        let shape = ACDShape { p: 1, q: 2 };
        let mut alpha_buf = Array1::<f64>::zeros(shape.q);
        let mut beta_buf = Array1::<f64>::zeros(shape.p);
        let mut ws = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape).unwrap();

        // Non-trivial θ vector of expected length 1 + q + p = 4.
        let theta = array![0.5_f64, -1.0_f64, 0.0_f64, 1.0_f64];

        // Act
        ws.update(theta.view()).expect("expected WorkSpace::update to succeed for simple θ");

        // Assert: ω
        assert!(ws.omega.is_finite(), "omega must be finite");
        assert!(ws.omega > 0.0, "omega must be strictly positive");

        // Assert: α, β, slack non-negative and finite
        for &a in ws.alpha.iter() {
            assert!(a.is_finite(), "alpha entries must be finite");
            assert!(a >= 0.0, "alpha entries must be non-negative");
        }
        for &b in ws.beta.iter() {
            assert!(b.is_finite(), "beta entries must be finite");
            assert!(b >= 0.0, "beta entries must be non-negative");
        }
        assert!(ws.slack.is_finite(), "slack must be finite");
        assert!(ws.slack >= 0.0, "slack must be non-negative");

        // Assert: simplex mass ≈ 1 − STATIONARITY_MARGIN
        let alpha_sum = ws.alpha.sum();
        let beta_sum = ws.beta.sum();
        let total_mass = alpha_sum + beta_sum + ws.slack;
        let target_mass = 1.0 - STATIONARITY_MARGIN;
        let diff = (total_mass - target_mass).abs();

        assert!(
            diff <= 1e-12,
            "alpha + beta + slack = {total_mass}, expected ≈ {target_mass}, diff = {diff}"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `WorkSpace::uncond_mean` returns a finite, positive
    // unconditional mean and respects the identity:
    //   1 − ∑α − ∑β ≈ STATIONARITY_MARGIN + slack
    // implied by the stationarity-margin construction.
    //
    // Given
    // -----
    // - A shape with p = 1, q = 1 (θ length = 3).
    // - A simple θ vector that yields a valid parameter configuration.
    //
    // Expect
    // ------
    // - `uncond_mean()` is finite and strictly positive.
    // - The denominator `1 − ∑α − ∑β` is > 0.
    // - `1 − ∑α − ∑β` ≈ `STATIONARITY_MARGIN + slack`.
    // - `uncond_mean()` ≈ `omega / (1 − ∑α − ∑β)`.
    fn workspace_uncond_mean_respects_stationarity_identity_and_is_positive() {
        // Arrange
        let shape = ACDShape { p: 1, q: 1 };
        let mut alpha_buf = Array1::<f64>::zeros(shape.q);
        let mut beta_buf = Array1::<f64>::zeros(shape.p);
        let mut ws = WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), &shape).unwrap();

        // θ of length 1 + q + p = 3.
        let theta = array![0.1_f64, -0.3_f64, 0.7_f64];
        ws.update(theta.view()).expect("expected WorkSpace::update to succeed for simple θ");

        // Act
        let mu = ws.uncond_mean();

        // Assert: basic sanity
        assert!(mu.is_finite(), "uncond_mean must be finite");
        assert!(mu > 0.0, "uncond_mean must be strictly positive");

        // Stationarity identity
        let sum_alpha = ws.alpha.sum();
        let sum_beta = ws.beta.sum();
        let denom1 = 1.0 - sum_alpha - sum_beta;
        let denom2 = STATIONARITY_MARGIN + ws.slack;

        assert!(
            denom1 > 0.0,
            "denominator 1 - sum(alpha) - sum(beta) must be positive under stationarity"
        );

        let denom_diff = (denom1 - denom2).abs();
        assert!(
            denom_diff <= 1e-12,
            "expected 1 - sum(alpha) - sum(beta) ≈ STATIONARITY_MARGIN + slack; diff = {denom_diff}"
        );

        let expected_mu = ws.omega / denom1;
        let mu_diff = (mu - expected_mu).abs();
        assert!(
            mu_diff <= 1e-12,
            "uncond_mean() = {mu}, expected omega / denom = {expected_mu}, diff = {mu_diff}"
        );
    }
}
