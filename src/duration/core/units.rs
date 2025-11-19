//! Duration units — metadata for the time scale of ACD durations.
//!
//! Purpose
//! -------
//! Provide a small enum describing the time units used for duration data in
//! ACD models. This metadata is carried alongside the series so downstream
//! code can format, report, or convert durations consistently without having
//! to guess the intended time scale.
//!
//! Key behaviors
//! -------------
//! - Declare the time granularity used for durations via [`ACDUnit`].
//! - Act as **metadata only**: choosing a unit does *not* rescale underlying
//!   numeric values by itself.
//! - Allow downstream components (formatters, I/O, Python wrappers) to decide
//!   how to present or convert durations based on a single, shared enum.
//!
//! Invariants & assumptions
//! ------------------------
//! - The numeric duration values stored elsewhere (e.g., in [`ACDData`]) are
//!   assumed to already be expressed in the unit declared by [`ACDUnit`].
//! - This module does **not** enforce positivity, finiteness, or any other
//!   constraints on the data; it only describes the unit.
//! - Callers that change the physical time scale of the data (e.g., from
//!   seconds to milliseconds) are responsible for updating both the numeric
//!   values and the associated [`ACDUnit`] consistently.
//!
//! Conventions
//! -----------
//! - Units follow SI-style naming and are interpreted as:
//!   - `Microseconds` = 1e-6 seconds,
//!   - `Milliseconds` = 1e-3 seconds,
//!   - `Seconds`      = 1 second.
//! - Indexing, [start, end) semantics, and zero-handling policies are defined
//!   in data/recursion modules (e.g., `data`, `psi`), not here.
//! - This module never panics; it defines a plain enum with no runtime logic.
//!
//! Downstream usage
//! ----------------
//! - Store an [`ACDUnit`] inside duration metadata structs (e.g., `ACDMeta`)
//!   to describe the scale of the series.
//! - Use the unit when formatting durations for logs, Python reprs, or plots.
//! - Use it as a dispatch key if you later add helper functions that convert
//!   between units at the edges of the system (I/O, adapters).
//!
//! Testing notes
//! -------------
//! - There is no runtime behavior to test here; [`ACDUnit`] is a small,
//!   exhaustively-typed enum with no methods.
//! - Any logic that interprets or converts units should be tested in the
//!   modules where that behavior is implemented.

/// ACDUnit — units of measurement for durations in an ACD model.
///
/// Purpose
/// -------
/// Represent the time scale used for duration data (microseconds,
/// milliseconds, or seconds). This is carried as metadata so that
/// downstream components can format, report, or convert durations
/// consistently without modifying the underlying values.
///
/// Variants
/// --------
/// - `Microseconds`  
///   Durations are expressed in microseconds (1e-6 seconds).
/// - `Milliseconds`  
///   Durations are expressed in milliseconds (1e-3 seconds).
/// - `Seconds`  
///   Durations are expressed in seconds (1 second).
///
/// Invariants
/// ----------
/// - This enum is **pure metadata**: it does not rescale or validate the
///   numeric duration values.
/// - Callers are responsible for ensuring that whatever array/series they
///   attach this unit to is actually in the declared time scale.
/// - All variants are value-like and may be freely copied; no additional
///   state is attached.
///
/// Notes
/// -----
/// - Because [`ACDUnit`] is used as metadata, it is included in higher-level meta
///   structs (e.g., `ACDMeta`) and passed across FFI boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ACDUnit {
    /// Microseconds (1e-6 s).
    Microseconds,
    /// Milliseconds (1e-3 s).
    Milliseconds,
    /// Seconds.
    Seconds,
}
