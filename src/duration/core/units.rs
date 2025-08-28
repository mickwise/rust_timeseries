//! Units and zero-handling policies for duration data.
//!
//! - [`ACDUnit`] declares the time granularity (micro/milli/seconds).
//!
//! Notes
//! -----
//! - `ACDUnit` is metadata only; it does not rescale values by itself.

/// Units of measurement for durations in an ACD model.
///
/// This sets the assumed time scale for the data and for any
/// reporting/interpretation downstream. It does **not** rescale values
/// automatically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ACDUnit {
    /// Microseconds (1e-6 s).
    Microseconds,
    /// Milliseconds (1e-3 s).
    Milliseconds,
    /// Seconds.
    Seconds,
}
