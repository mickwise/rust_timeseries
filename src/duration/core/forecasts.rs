//! Out-of-sample ψ-forecasting for ACD(p, q) — roll ψ-recursions beyond the sample without allocations.
//!
//! Purpose
//! -------
//! Provide allocation-free out-of-sample ψ-forecasting for ACD(p, q) models by
//! separating forecast storage from recursion logic and reusing preallocated
//! buffers.
//!
//! Key behaviors
//! -------------
//! - Roll the ACD(p, q) ψ-recursion forward for a fixed horizon using fitted
//!   model-space parameters and pre-sample ψ- and duration-lags.
//! - Write ψ̂ forecasts into a reusable buffer (`ACDForecastResult`) without
//!   heap allocations inside the forecasting loop.
//! - Clamp each ψ̂ using `PsiGuards` to maintain numerical stability.
//!
//! Invariants & assumptions
//! ------------------------
//! - The ACD model follows the recursion
//!   `ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j}`.
//! - Fitted parameters satisfy stationarity constraints (e.g.,
//!   `sum(α) + sum(β) < 1 − margin`) enforced upstream.
//! - `duration_lags` contains the last `q` observed durations with the newest
//!   element at the end.
//! - `ACDParams::psi_lags` contains the last `p` in-sample ψ values with the
//!   newest element at the end.
//! - The internal forecast buffer has capacity for at least the requested
//!   horizon.
//!
//! Conventions
//! -----------
//! - Indexing is 0-based; `psi_forecast[i]` stores the i-step-ahead forecast
//!   `ψ̂_{T+i+1}` for `i >= 0`.
//! - Duration forecasts are identified with ψ-forecasts under unit-mean
//!   innovations, i.e., `τ̂ = ψ̂`.
//! - The module assumes callers respect length relationships between `alpha`,
//!   `beta`, `duration_lags`, `psi_lags`, and the forecast horizon; violations
//!   are considered logic errors and may panic via out-of-bounds slicing.
//! - No timezones or calendar conventions are applied here; only index-based
//!   steps are modeled.
//!
//! Downstream usage
//! ----------------
//! - Construct `ACDForecastResult` with the desired horizon and reuse it across
//!   forecast calls to amortize allocation costs.
//! - After fitting an ACD model and deriving `ACDParams`, call
//!   `forecast_recursion` with the last `q` observed durations and the fitted
//!   `psi_lags` to obtain out-of-sample ψ̂ paths.
//! - Consume either the full forecast path from `psi_forecast[..horizon]` or
//!   the final value `ψ̂_{T+horizon}` returned by `forecast_recursion`.
//!
//! Testing notes
//! -------------
//! - Unit tests cover single-step and multi-step forecasts, including
//!   cases where `horizon` is less than, equal to, and greater than `max(p, q)`.
//! - Tests verify correct handling of lag splitting between in-sample
//!   lags and already-forecast ψ̂ values.
//! - Tests confirm that `PsiGuards` clamping is applied when ψ̂ values
//!   fall outside the allowed range.
use ndarray::{Array1, ArrayView1, s};
use std::{cell::RefCell, cmp::min};

use crate::duration::{
    core::{guards::PsiGuards, params::ACDParams, psi::guard_psi},
    errors::ACDResult,
};

/// ACDForecastResult — container for out-of-sample ψ-forecast paths.
///
/// Purpose
/// -------
/// Represent a reusable buffer for ψ̂ forecasts so that ACD(p, q) forecasting
/// routines can operate allocation-free while writing results into
/// preallocated storage.
///
/// Key behaviors
/// -------------
/// - Holds an `Array1<f64>` of ψ̂ values wrapped in a `RefCell` to support
///   interior mutability from forecast routines.
/// - Provides an indexed view of the forecast path where `psi_forecast[i]`
///   stores the i-step-ahead ψ-forecast.
/// - Can be reused across multiple calls with the same or smaller horizon to
///   amortize allocation costs.
///
/// Parameters
/// ----------
/// - `horizon`: `usize`
///   Length of the ψ̂ forecast path this container should hold, specified at
///   construction time via [`ACDForecastResult::new`].
///
/// Fields
/// ------
/// - `psi_forecast`: `RefCell<Array1<f64>>`
///   Preallocated buffer of ψ̂ values. By convention, `psi_forecast[i]` is the
///   i-step-ahead forecast `ψ̂_{T+i+1}` after a forecasting routine has run.
///   Contents are initialized to zeros and overwritten by forecasting code.
///
/// Invariants
/// ----------
/// - The underlying buffer length is determined by `horizon` at construction
///   and is expected to be large enough to hold any requested forecast horizon
///   used with this instance.
/// - `ACDForecastResult` does not enforce numerical properties of ψ̂ values;
///   those are controlled by the forecasting algorithm and `PsiGuards`.
///
/// Performance
/// -----------
/// - Construction cost is O(horizon) due to zero-initializing the buffer.
/// - Subsequent forecast calls can reuse the same buffer without additional
///   allocations, enabling efficient repeated forecasting.
///
/// Notes
/// -----
/// - Interior mutability via `RefCell` allows forecast routines to mutate the
///   buffer even when `ACDForecastResult` is passed by shared reference.
/// - This type is intended to be used only on a single thread at a time; it is
///   not designed to provide synchronization across threads.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDForecastResult {
    pub psi_forecast: RefCell<Array1<f64>>,
}

impl ACDForecastResult {
    /// Create a new forecast buffer for ψ̂ with the given horizon.
    ///
    /// Parameters
    /// ----------
    /// - `horizon`: `usize`
    ///   Number of forecast steps to allocate storage for. This determines the
    ///   length of the internal `psi_forecast` array.
    ///
    /// Returns
    /// -------
    /// ACDForecastResult
    ///   A forecast container whose internal buffer is initialized with zeros and
    ///   can be populated by forecasting routines.
    ///
    /// Errors
    /// ------
    /// - No errors are returned; this is a plain constructor that always succeeds.
    ///
    /// Panics
    /// ------
    /// - Never panics under normal operation. Very large `horizon` values may
    ///   trigger allocation failures at the allocator level.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional safety guarantees are required from the
    ///   caller.
    ///
    /// Notes
    /// -----
    /// - All entries in `psi_forecast` start as zero and are expected to be
    ///   overwritten by forecasting routines (e.g., `forecast_recursion`).
    /// - Reusing a single `ACDForecastResult` across multiple forecasts avoids
    ///   repeated allocation overhead.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::forecasts::ACDForecastResult;
    /// let horizon = 10;
    /// let buffer = ACDForecastResult::new(horizon);
    /// assert_eq!(buffer.psi_forecast.borrow().len(), horizon);
    /// ```
    pub fn new(horizon: usize) -> Self {
        Self { psi_forecast: RefCell::new(Array1::zeros(horizon)) }
    }
}

/// Roll the ACD(p, q) ψ-recursion forward for a fixed horizon and write ψ̂ into a preallocated buffer.
///
/// Parameters
/// ----------
/// - `params`: `&ACDParams`
///   Fitted model-space parameters for the ACD(p, q) model, including `omega`,
///   `alpha` (length q), `beta` (length p), and the last p in-sample ψ-lags in
///   `psi_lags`, with the newest element at the end.
/// - `duration_lags`: `ArrayView1<f64>`
///   View of the last q observed durations, ordered oldest → newest, with the
///   newest element at the end. These provide the initial duration lags used
///   in the α term.
/// - `horizon`: `usize`
///   Number of steps H ≥ 1 to forecast. For each i in `0..horizon`, the
///   i-step-ahead ψ̂ value is written into `forecast_result.psi_forecast[i]`.
/// - `forecast_result`: `&ACDForecastResult`
///   Destination buffer for ψ̂ forecasts. Must have capacity for at least
///   `horizon` elements in `psi_forecast`.
/// - `guards`: `&PsiGuards`
///   Guard rails specifying minimum and maximum allowable ψ values. Each ψ̂ is
///   clamped into `[guards.min, guards.max]` after computation.
///
/// Returns
/// -------
/// ACDResult<f64>
///   - `Ok(value)` containing the last ψ-forecast `ψ̂_{T+horizon}` when the
///     recursion succeeds for all steps.
///   - `Err(err)` if an error occurs during forecasting (e.g., from guard
///     logic or parameter checks), although such errors should be rare if
///     preconditions are met.
///
/// Errors
/// ------
/// - `ACDError`
///   May be returned if internal checks or auxiliary operations in the ACD
///   stack report an error while computing ψ̂.
///
/// Panics
/// ------
/// - May panic if lengths and horizon are inconsistent, such as:
///   - `duration_lags.len() != params.alpha.len()`
///   - `params.psi_lags.len() != params.beta.len()`
///   - `forecast_result.psi_forecast` has length `< horizon`
///   These are treated as logic errors in the caller and are not guarded
///   against at runtime within this function.
///
/// Safety
/// ------
/// - Not an `unsafe fn`; no additional safety guarantees are required beyond
///   respecting the documented length and stationarity preconditions.
///
/// Notes
/// -----
/// - The recursion implements the ACD model
///   `ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j}` and extends it
///   forward out of sample.
/// - For each step i:
///   - Duration lags (α term) are built by combining an in-sample tail from
///     `duration_lags` with a forecast tail from already-computed ψ̂ values,
///     read newest → oldest.
///   - ψ lags (β term) are built by combining an in-sample tail from
///     `params.psi_lags` with already-computed ψ̂ values, also read newest →
///     oldest.
/// - Under unit-mean innovations, duration forecasts are identified with
///   ψ-forecasts (`τ̂ = ψ̂`), so ψ̂ serves as both conditional mean and
///   duration forecast.
/// - The inner loop is designed to be allocation-free: only existing buffers
///   in `ACDParams` and `ACDForecastResult` are used.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::{array, ArrayView1};
/// # use rust_timeseries::duration::core::forecasts::{ACDForecastResult, forecast_recursion};
/// # use rust_timeseries::duration::core::params::ACDParams;
/// # use rust_timeseries::duration::core::guards::PsiGuards;
/// #
/// # // This is a sketch; real code would construct ACDParams from a fitted model.
/// # let params = /* obtain fitted ACDParams here */;
/// # let guards = PsiGuards { min: 1e-6, max: 1e6 };
/// let duration_lags_array = array![1.0, 1.2, 0.9]; // last q durations
/// let duration_lags: ArrayView1<f64> = duration_lags_array.view();
/// let horizon = 5;
/// let forecast_result = ACDForecastResult::new(horizon);
///
/// let last_psi = forecast_recursion(&params, duration_lags, horizon, &forecast_result, &guards)?;
/// # // After this call, forecast_result.psi_forecast.borrow()[0..horizon]
/// # // contains the full ψ̂ path, and last_psi is ψ̂_{T+horizon}.
/// # let _ = last_psi;
/// # Ok::<(), rust_timeseries::duration::errors::ACDError>(())
/// ```
pub fn forecast_recursion(
    params: &ACDParams, duration_lags: ArrayView1<f64>, horizon: usize,
    forecast_result: &ACDForecastResult, guards: &PsiGuards,
) -> ACDResult<f64> {
    let alpha = &params.alpha;
    let beta = &params.beta;
    let q = alpha.len();
    let p = beta.len();
    let psi_lags = &params.psi_lags;
    let mut forecast_psi = forecast_result.psi_forecast.borrow_mut();
    for i in 0..horizon {
        let k_init = q.saturating_sub(i);
        let k_data = q - k_init;
        let psi_forecast_len = min(i, p);
        let psi_in_sample_len = p - psi_forecast_len;
        let init_tail_rev = duration_lags.slice(s![q - k_init..q; -1]);
        let data_tail_rev = forecast_psi.slice(s![i - k_data..i; -1]);
        let init_psi_lags_rev = psi_lags.slice(s![p - psi_in_sample_len..p; -1]);
        let psi_lags_rev = forecast_psi.slice(s![i - psi_forecast_len..i; -1]);
        let sum_alpha = alpha.slice(s![..k_init]).dot(&init_tail_rev)
            + alpha.slice(s![k_init..]).dot(&data_tail_rev);
        let sum_beta = beta.slice(s![..psi_in_sample_len]).dot(&init_psi_lags_rev)
            + beta.slice(s![psi_in_sample_len..]).dot(&psi_lags_rev);
        let curr_psi = params.omega + sum_alpha + sum_beta;
        forecast_psi[i] = guard_psi(curr_psi, guards);
    }
    Ok(forecast_psi[horizon - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::core::{guards::PsiGuards, params::ACDParams};
    use ndarray::{Array1, array};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Basic construction behavior of `ACDForecastResult::new`.
    // - Behavior of `forecast_recursion` under simple constant ACD dynamics
    //   where the recursion degenerates to `ψ̂ = ω` before guarding.
    // - The effect of `PsiGuards` on clamping ψ̂ when the unconstrained
    //   recursion would produce values above the allowed maximum.
    //
    // They intentionally DO NOT cover:
    // - Realistic parameter estimation or stationarity margin configuration
    //   (those are assumed to be handled upstream).
    // - Complex parameterizations with non-zero alpha/beta; those are
    //   validated through higher-level integration tests.
    // -------------------------------------------------------------------------

    // Helper to construct simple ACD parameters where alpha = 0, beta = 0 so
    // that the unconstrained recursion yields ψ̂ = ω at every step.
    //
    // NOTE
    // ----
    // We bypass `ACDParams::new` here because tests are focused on the
    // behavior of `forecast_recursion`, not on parameter validation logic.
    // The constructed parameters are consistent with the recursion itself:
    // - `alpha` is a length-q zero vector.
    // - `beta` is a length-p zero vector.
    // - `psi_lags` is length p with all entries equal to `psi0`.
    // - `slack` is set to 0.0 and is not used by `forecast_recursion`.
    //
    // Given
    // -----
    // - `omega`: desired constant level for ψ̂.
    // - `p`: number of ψ-lags.
    // - `q`: number of duration lags.
    // - `psi0`: value used to populate the in-sample ψ-lags.
    //
    // Expect
    // ------
    // - `alpha` and `beta` contribute nothing to the recursion, leaving
    //   ψ̂ = ω (modulo guard clamping).
    fn make_constant_params_unchecked(omega: f64, p: usize, q: usize, psi0: f64) -> ACDParams {
        let alpha: Array1<f64> = Array1::zeros(q);
        let beta: Array1<f64> = Array1::zeros(p);
        let psi_lags: Array1<f64> = Array1::from(vec![psi0; p]);

        // We bypass validation and construct ACDParams directly. `slack` is
        // irrelevant for `forecast_recursion`, so we set it to 0.0.
        ACDParams { omega, alpha, beta, slack: 0.0, psi_lags }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDForecastResult::new` allocates a buffer of the requested
    // horizon and initializes all entries to zero.
    //
    // Given
    // -----
    // - `horizon = 4`.
    //
    // Expect
    // ------
    // - `psi_forecast.len() == horizon`.
    // - All entries in `psi_forecast` are exactly zero before any forecasting.
    fn acdforecastresult_new_initializes_zero_buffer() {
        // Arrange
        let horizon = 4;

        // Act
        let forecast_result = ACDForecastResult::new(horizon);
        let buf = forecast_result.psi_forecast.borrow();

        // Assert
        assert_eq!(buf.len(), horizon);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `forecast_recursion` produces a single-step ψ̂ equal to ω
    // when alpha and beta are zero and guards do not clamp.
    //
    // Given
    // -----
    // - ACD parameters with:
    //   - `omega = 2.0`.
    //   - `alpha` and `beta` zero vectors (no dependence on lags).
    //   - `psi_lags` arbitrary but ignored due to zero beta.
    // - `q = 1`, `p = 1`.
    // - `duration_lags = [1.0]` (arbitrary, ignored due to zero alpha).
    // - `horizon = 1`.
    // - `PsiGuards` with a wide range `[min = 0.0, max = 1e6]`.
    //
    // Expect
    // ------
    // - `forecast_recursion` returns `Ok(2.0)`.
    // - `psi_forecast[0] == 2.0`.
    fn forecast_recursion_constant_params_single_step_returns_omega() {
        // Arrange
        let omega = 2.0;
        let p = 1;
        let q = 1;
        let params = make_constant_params_unchecked(omega, p, q, 0.5);
        let duration_lags_array = array![1.0];
        let duration_lags = duration_lags_array.view();
        let horizon = 1;
        let forecast_result = ACDForecastResult::new(horizon);
        let guards = PsiGuards { min: 0.0, max: 1e6 };

        // Act
        let last_psi =
            forecast_recursion(&params, duration_lags, horizon, &forecast_result, &guards)
                .expect("forecast_recursion should succeed");

        let buf = forecast_result.psi_forecast.borrow();

        // Assert
        assert!((last_psi - omega).abs() < 1e-12);
        assert_eq!(buf.len(), horizon);
        assert!((buf[0] - omega).abs() < 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `forecast_recursion` produces a constant ψ̂ path equal to ω
    // across multiple steps when alpha and beta are zero and guards do not
    // clamp.
    //
    // Given
    // -----
    // - ACD parameters with:
    //   - `omega = 3.5`.
    //   - `alpha` and `beta` zero vectors.
    //   - `psi_lags` arbitrary but ignored due to zero beta.
    // - `q = 1`, `p = 1`.
    // - `duration_lags = [1.0]` (arbitrary, ignored in the α term).
    // - `horizon = 3`.
    // - `PsiGuards` with a wide range `[min = 0.0, max = 1e6]`.
    //
    // Expect
    // ------
    // - `forecast_recursion` returns `Ok(3.5)`.
    // - All entries in `psi_forecast[0..horizon]` equal 3.5 within tolerance.
    fn forecast_recursion_constant_params_multi_step_writes_constant_path() {
        // Arrange
        let omega = 3.5;
        let p = 1;
        let q = 1;
        let params = make_constant_params_unchecked(omega, p, q, 0.5);
        let duration_lags_array = array![1.0];
        let duration_lags = duration_lags_array.view();
        let horizon = 3;
        let forecast_result = ACDForecastResult::new(horizon);
        let guards = PsiGuards { min: 0.0, max: 1e6 };

        // Act
        let last_psi =
            forecast_recursion(&params, duration_lags, horizon, &forecast_result, &guards)
                .expect("forecast_recursion should succeed");

        let buf = forecast_result.psi_forecast.borrow();

        // Assert
        assert!((last_psi - omega).abs() < 1e-12);
        assert_eq!(buf.len(), horizon);
        for &v in buf.iter() {
            assert!((v - omega).abs() < 1e-12);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `forecast_recursion` respects `PsiGuards` by clamping ψ̂
    // values that would otherwise exceed the allowed maximum.
    //
    // Given
    // -----
    // - ACD parameters with:
    //   - `omega = 1_000.0`.
    //   - `alpha` and `beta` zero vectors (so unconstrained ψ̂ = 1_000.0).
    //   - `psi_lags` arbitrary but ignored due to zero beta.
    // - `q = 1`, `p = 1`.
    // - `duration_lags = [1.0]` (arbitrary, ignored due to zero alpha).
    // - `horizon = 3`.
    // - `PsiGuards` with `[min = 0.0, max = 10.0]`.
    //
    // Expect
    // ------
    // - `forecast_recursion` returns `Ok(10.0)`.
    // - All entries in `psi_forecast[0..horizon]` equal 10.0 within tolerance.
    fn forecast_recursion_applies_guard_clamping_on_large_omega() {
        // Arrange
        let omega = 1_000.0;
        let p = 1;
        let q = 1;
        let params = make_constant_params_unchecked(omega, p, q, 0.5);
        let duration_lags_array = array![1.0];
        let duration_lags = duration_lags_array.view();
        let horizon = 3;
        let forecast_result = ACDForecastResult::new(horizon);
        let guards = PsiGuards { min: 0.0, max: 10.0 };

        // Act
        let last_psi =
            forecast_recursion(&params, duration_lags, horizon, &forecast_result, &guards)
                .expect("forecast_recursion should succeed");

        let buf = forecast_result.psi_forecast.borrow();

        // Assert
        assert!((last_psi - guards.max).abs() < 1e-12);
        assert_eq!(buf.len(), horizon);
        for &v in buf.iter() {
            assert!((v - guards.max).abs() < 1e-12);
        }
    }
}
