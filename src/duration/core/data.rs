//! Data containers for duration models (ACD family).
//!
//! This module defines two small types:
//! - [`ACDData`]: the observed durations plus an optional reference origin `t0` and metadata.
//! - [`ACDMeta`]: how to interpret the data (time unit, scale and optional flags).
//!
//! Invariants enforced by constructors:
//! - Durations must be strictly positive and finite.
//! - The series must be non-empty.
//! - If provided, `t0` is an in-bounds index into the series.
//!
//! Notes
//! -----
//! - `t0` is an optional index of the first observation in the estimation window
//!   relative to some external offset.
//! - `scale` in [`ACDMeta`] is an internal numeric stabilizer; it does not change
//!   the raw input values exposed by [`ACDData::data`].
use crate::duration::{
    core::units::ACDUnit,
    errors::{ACDError, ACDResult},
};
use ndarray::Array1;

/// Container for input duration data and associated metadata.
///
/// Guarantees after construction:
/// - the series is non-empty;
/// - all durations are finite and strictly positive;
/// - if `t0` is `Some(i)`, then `i < data.len()`.
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
    /// Construct a validated [`ACDData`] instance.
    ///
    /// # Errors
    /// Returns an error if:
    /// - the series is empty;
    /// - any element is non-finite or non-positive;
    /// - `t0` is specified but out of range.
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

/// Metadata describing how the duration data should be interpreted.
///
/// Includes the time unit, optional numeric scaling,
/// and whether durations were adjusted for diurnal patterns.
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
    /// Assumes the provided arguments are already consistent; no additional
    /// validation is performed here.
    pub fn new(unit: ACDUnit, scale: Option<f64>, diurnal_adjusted: bool) -> ACDMeta {
        ACDMeta { unit, scale, diurnal_adjusted }
    }
}
