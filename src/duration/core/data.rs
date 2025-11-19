//! Duration data containers for ACD family models.
//!
//! Purpose
//! -------
//! Provide small, validated containers for duration series and their metadata
//! used by ACD(p, q) models. This module centralizes input validation for raw
//! duration data and standardizes how time units and scaling are represented.
//!
//! Key behaviors
//! -------------
//! - [`ACDData`] enforces basic data invariants (non-empty, finite, strictly
//!   positive durations, and in-bounds `t0` when present).
//! - [`ACDMeta`] describes how to interpret the series (units, optional
//!   scaling, and diurnal adjustment flag) without mutating raw values.
//!
//! Invariants & assumptions
//! ------------------------
//! - Durations must be **strictly positive** and **finite**.
//! - The series must be non-empty at construction time.
//! - If `t0` is `Some(i)`, then `i < data.len()` must hold.
//! - `ACDMeta::scale` is an internal numeric stabilizer only; it never changes
//!   the raw values exposed by `ACDData::data`.
//!
//! Conventions
//! -----------
//! - Indexing is 0-based and `t0` is an optional index into the series for
//!   likelihood / estimation windows.
//! - Units are described via [`ACDUnit`] (e.g., milliseconds, seconds).
//! - This module does **not** apply any diurnal or calendar adjustments; it
//!   only records whether such adjustments were performed upstream.
//!
//! Downstream usage
//! ----------------
//! - Construct [`ACDData`] at the Rust boundary where raw durations enter the
//!   ACD modeling stack.
//! - Use [`ACDMeta`] to communicate units and pre-processing flags to
//!   downstream modules (e.g., recursion, inference, forecasting).
//! - Consumers may safely rely on `ACDData` invariants when implementing
//!   recursions and likelihoods.
//!
//! Testing notes
//! -------------
//! - Unit tests cover construction behavior for `ACDData::new` (happy path,
//!   empty series, non-finite values, non-positive values, and out-of-range
//!   `t0`).
//! - `ACDMeta::new` is a plain constructor with no additional validation and
//!   is tested implicitly via `ACDData` and higher-level modules.
use crate::duration::{
    core::units::ACDUnit,
    errors::{ACDError, ACDResult},
};
use ndarray::Array1;

/// `ACDData` — validated duration series plus optional origin and metadata.
///
/// Purpose
/// -------
/// Represent a single, validated time series of durations for ACD family
/// models, together with an optional estimation origin `t0` and interpretation
/// metadata. This type centralizes basic input checks so downstream code can
/// assume clean, strictly positive data.
///
/// Key behaviors
/// -------------
/// - Stores raw duration observations as an `ndarray::Array1<f64>`.
/// - Enforces non-emptiness, finiteness, and strict positivity at
///   construction time via [`ACDData::new`].
/// - Optionally records an origin index `t0` for likelihood / estimation
///   windows (without modifying the data).
///
/// Fields
/// ------
/// - `data`: `Array1<f64>`
///   Observed durations; must be finite and strictly greater than zero.
/// - `t0`: `Option<usize>`
///   Optional index of the first observation in the estimation window. When
///   `Some(i)`, it must satisfy `i < data.len()`.
/// - `meta`: [`ACDMeta`]
///   Interpretation details such as time units, internal scaling, and whether
///   diurnal adjustment has been applied.
///
/// Invariants
/// ----------
/// - `data.len() > 0`.
/// - All entries in `data` are finite and strictly positive.
/// - If `t0.is_some()`, then `t0.unwrap() < data.len()`.
///
/// Performance
/// -----------
/// - Validation is O(n) in the number of observations due to a single scan
///   over `data`.
/// - After construction, this type is a lightweight container with no hidden
///   allocations.
///
/// Notes
/// -----
/// - This type does not perform any rescaling or transformation of the input
///   durations; it only validates them.
/// - Higher-level modules (e.g., ACD models, forecasting) may rely on these
///   invariants and avoid re-validating basic properties.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDData {
    /// Observed durations (must be finite and > 0).
    pub data: Array1<f64>,
    /// Optional index of the first observation in the estimation window.
    pub t0: Option<usize>,
    /// Interpretation details (time units, scaling flags).
    pub meta: ACDMeta,
}

impl ACDData {
    /// Construct a validated [`ACDData`] instance from raw durations.
    ///
    /// Parameters
    /// ----------
    /// - `data`: `Array1<f64>`
    ///   Raw duration series. Must be non-empty, finite, and strictly positive.
    /// - `t0`: `Option<usize>`
    ///   Optional index representing the start of the estimation window relative
    ///   to `data`. If `Some(i)`, it must satisfy `i < data.len()`.
    /// - `meta`: [`ACDMeta`]
    ///   Metadata describing how to interpret the series (units, internal scaling,
    ///   and diurnal adjustment flag).
    ///
    /// Returns
    /// -------
    /// `ACDResult<ACDData>`
    ///   - `Ok(ACDData)` if all invariants are satisfied.
    ///   - `Err(ACDError)` if validation fails.
    ///
    /// Errors
    /// ------
    /// - `ACDError::EmptySeries`  
    ///   Returned when `data.len() == 0`.
    /// - `ACDError::NonFiniteData { index, value }`  
    ///   Returned when any element of `data` is NaN or ±∞; `index` points to the
    ///   first offending element.
    /// - `ACDError::NonPositiveData { index, value }`  
    ///   Returned when any element of `data` is ≤ 0; `index` points to the first
    ///   offending element.
    /// - `ACDError::T0OutOfRange { t0, len }`  
    ///   Returned when `t0` is `Some(i)` and `i >= data.len()`.
    ///
    /// Panics
    /// ------
    /// - Never panics. All invalid inputs are reported via `ACDError`.
    ///
    /// Notes
    /// -----
    /// - Validation is performed in a single pass over `data`, stopping at the
    ///   first invalid element.
    /// - `meta` is not modified or validated beyond its own constructor; it is
    ///   stored as-is.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use ndarray::array;
    /// # use rust_timeseries::duration::core::data::{ACDData, ACDMeta};
    /// # use rust_timeseries::duration::core::units::ACDUnit;
    /// #
    /// let data = array![1.0, 2.0, 3.0];
    /// let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
    /// let acd = ACDData::new(data, Some(0), meta).unwrap();
    /// assert_eq!(acd.t0, Some(0));
    /// ```
    pub fn new(data: Array1<f64>, t0: Option<usize>, meta: ACDMeta) -> ACDResult<Self> {
        if data.is_empty() {
            return Err(ACDError::EmptySeries);
        }

        for (index, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(ACDError::NonFiniteData { index, value });
            }
            if value <= 0.0 {
                return Err(ACDError::NonPositiveData { index, value });
            }
        }

        if let Some(t0_val) = t0 {
            if t0_val >= data.len() {
                return Err(ACDError::T0OutOfRange { t0: t0_val, len: data.len() });
            }
        }

        Ok(ACDData { data, t0, meta })
    }
}

/// `ACDMeta` — interpretation metadata for duration series.
///
/// Purpose
/// -------
/// Describe how a duration series should be interpreted without altering the
/// raw numeric values. This type captures units, optional internal scaling
/// used for numerical stability, and whether diurnal adjustment has been
/// applied upstream.
///
/// Key behaviors
/// -------------
/// - Encapsulates time units via [`ACDUnit`] (e.g., milliseconds, seconds).
/// - Stores an optional internal scaling factor used by downstream numerics.
/// - Records whether durations have been pre-adjusted for diurnal patterns.
///
/// Fields
/// ------
/// - `unit`: [`ACDUnit`]
///   Time unit of the durations (e.g., milliseconds, seconds).
/// - `scale`: `Option<f64>`
///   Optional internal scaling factor for numerical stability. This is **not**
///   user-facing and does not change the raw values in [`ACDData::data`].
/// - `diurnal_adjusted`: `bool`
///   Indicates whether the input series has already been adjusted for diurnal
///   effects by upstream preprocessing.
///
/// Invariants
/// ----------
/// - This type does not enforce numeric invariants itself; it assumes the
///   provided values are consistent with the calling context.
/// - The meaning of `scale` is internal to the modeling stack; downstream
///   code must ensure it is used consistently.
///
/// Notes
/// -----
/// - `ACDMeta` is intentionally lightweight and does not validate its fields
///   beyond the type system; validation policies belong to higher-level
///   components.
/// - The presence of `scale` allows internal rescaling strategies without
///   losing the original units or raw data.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDMeta {
    /// Time unit of the durations (e.g., milliseconds, seconds).
    pub unit: ACDUnit,
    /// Optional internal scaling factor used for numerical stability; not
    /// user-specified and does not alter raw input values.
    pub scale: Option<f64>,
    /// Whether the input has been pre-adjusted for diurnal effects.
    pub diurnal_adjusted: bool,
}

impl ACDMeta {
    /// Construct a new [`ACDMeta`] instance.
    ///
    /// Parameters
    /// ----------
    /// - `unit`: [`ACDUnit`]
    ///   Time unit of the duration series (e.g., milliseconds, seconds).
    /// - `scale`: `Option<f64>`
    ///   Optional internal scaling factor used for numerical stability. This does
    ///   not modify raw input values; it is purely an internal hint.
    /// - `diurnal_adjusted`: `bool`
    ///   Flag indicating whether the input series has been adjusted for diurnal
    ///   patterns upstream.
    ///
    /// Returns
    /// -------
    /// `ACDMeta`
    ///   A metadata value that can be attached to [`ACDData`] or used by
    ///   downstream modeling components.
    ///
    /// Errors
    /// ------
    /// - Never returns an error; this is a plain constructor.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Notes
    /// -----
    /// - This constructor assumes the caller has already chosen consistent values
    ///   for `unit`, `scale`, and `diurnal_adjusted`.
    /// - Any additional validation of `scale` or interpretation conventions
    ///   belong to higher-level components in the ACD stack.
    pub fn new(unit: ACDUnit, scale: Option<f64>, diurnal_adjusted: bool) -> ACDMeta {
        ACDMeta { unit, scale, diurnal_adjusted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::core::units::ACDUnit;
    use crate::duration::errors::ACDError;
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Construction behavior of `ACDData::new`.
    // - Enforcement of invariants:
    //   * non-empty series,
    //   * finite values,
    //   * strictly positive durations,
    //   * in-bounds `t0` when provided.
    //
    // These tests intentionally DO NOT cover:
    // - Semantics or validation of `ACDMeta` beyond being constructible.
    // -------------------------------------------------------------------------

    // Purpose
    // -------
    // Provide a minimal, consistent `ACDMeta` instance for use in tests.
    //
    // Given
    // -----
    // - Concrete `ACDUnit::Seconds`.
    // - No internal scaling (`scale = None`).
    // - `diurnal_adjusted = false`.
    //
    // Expect
    // ------
    // - Returns an `ACDMeta` that can be safely reused across tests without
    //   affecting the invariants of `ACDData::new`.
    fn make_meta_stub() -> ACDMeta {
        ACDMeta::new(ACDUnit::Seconds, None, false)
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDData::new` succeeds on a valid, non-empty, strictly
    // positive and finite series with an in-bounds `t0`.
    //
    // Given
    // -----
    // - `data = [1.0, 2.0, 3.0]`.
    // - `t0 = Some(1)` (strictly less than `data.len()`).
    // - A valid `ACDMeta` from `make_meta_stub()`.
    //
    // Expect
    // ------
    // - `ACDData::new` returns `Ok(..)`.
    // - The returned `ACDData` preserves `data`, `t0` and `meta` exactly.
    fn acddata_new_returns_ok_for_valid_input() {
        let data = array![1.0, 2.0, 3.0];
        let t0 = Some(1);
        let meta = make_meta_stub();

        let result = ACDData::new(data.clone(), t0, meta.clone());

        assert!(result.is_ok());
        let acddata = result.unwrap();
        assert_eq!(acddata.data, data);
        assert_eq!(acddata.t0, t0);
        assert_eq!(acddata.meta, meta);
    }

    #[test]
    // Purpose
    // -------
    // Ensure `ACDData::new` rejects an empty series.
    //
    // Given
    // -----
    // - `data = []` (length zero).
    // - `t0 = None`.
    // - A valid `ACDMeta` from `make_meta_stub()`.
    //
    // Expect
    // ------
    // - `ACDData::new` returns `Err(ACDError::EmptySeries)`.
    fn acddata_new_returns_error_for_empty_series() {
        let data = array![];
        let meta = make_meta_stub();

        let result = ACDData::new(data, None, meta);

        assert_eq!(result.unwrap_err(), ACDError::EmptySeries);
    }

    #[test]
    // Purpose
    // -------
    // Ensure `ACDData::new` rejects non-finite values (NaN / ±∞) and reports
    // the index and offending value.
    //
    // Given
    // -----
    // - `data = [1.0, +∞, 3.0]`.
    // - `t0 = None`.
    // - A valid `ACDMeta` from `make_meta_stub()`.
    //
    // Expect
    // ------
    // - `ACDData::new` returns `Err(ACDError::NonFiniteData { index: 1, value })`
    //   where `value` is the non-finite element.
    fn acddata_new_returns_error_for_non_finite_data() {
        let data = array![1.0, f64::INFINITY, 3.0];
        let meta = make_meta_stub();

        let result = ACDData::new(data.clone(), None, meta);

        assert_eq!(result.unwrap_err(), ACDError::NonFiniteData { index: 1, value: data[1] });
    }

    #[test]
    // Purpose
    // -------
    // Ensure `ACDData::new` rejects non-positive durations (<= 0.0) and reports
    // the first offending index and value.
    //
    // Given
    // -----
    // - `data = [1.0, 0.0, -1.0]` so the first non-positive value is at index 1.
    // - `t0 = None`.
    // - A valid `ACDMeta` from `make_meta_stub()`.
    //
    // Expect
    // ------
    // - `ACDData::new` returns
    //   `Err(ACDError::NonPositiveData { index: 1, value: 0.0 })`.
    fn acddata_new_returns_error_for_non_positive_data() {
        let data = array![1.0, 0.0, -1.0];
        let meta = make_meta_stub();

        let result = ACDData::new(data.clone(), None, meta);

        // First non-positive is at index 1 with value 0.0
        assert_eq!(result.unwrap_err(), ACDError::NonPositiveData { index: 1, value: data[1] });
    }

    #[test]
    // Purpose
    // -------
    // Ensure `ACDData::new` rejects an out-of-range `t0` and reports both the
    // invalid `t0` and the series length.
    //
    // Given
    // -----
    // - `data = [1.0, 2.0, 3.0]` so `data.len() == 3`.
    // - `t0 = Some(3)` which is exactly `len` and therefore out of range.
    // - A valid `ACDMeta` from `make_meta_stub()`.
    //
    // Expect
    // ------
    // - `ACDData::new` returns
    //   `Err(ACDError::T0OutOfRange { t0: 3, len: 3 })`.
    fn acddata_new_returns_error_for_out_of_range_t0() {
        let data = array![1.0, 2.0, 3.0];
        let meta = make_meta_stub();
        let t0 = Some(3); // == len, so out of range

        let result = ACDData::new(data.clone(), t0, meta);

        assert_eq!(result.unwrap_err(), ACDError::T0OutOfRange { t0: 3, len: data.len() });
    }
}
