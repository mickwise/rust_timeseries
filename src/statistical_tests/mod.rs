//! statistical_tests — robust time-series diagnostics and helpers.
//!
//! Purpose
//! -------
//! Collect statistical-test routines and their shared infrastructure for
//! time-series diagnostics. This subtree currently implements the
//! Escanciano–Lobato heteroskedasticity–robust portmanteau test together
//! with common input validation and error handling, including Python
//! bridges for PyO3-based bindings.
//!
//! Key behaviors
//! -------------
//! - Expose a heteroskedasticity–robust automatic portmanteau test via
//!   [`ELOutcome`] and its constructor
//!   [`ELOutcome::escanciano_lobato`](escanciano_lobato::ELOutcome::escanciano_lobato).
//! - Centralize test input guards in [`validate_input`], ensuring series
//!   length, finiteness, and tuning parameters are checked once in a
//!   consistent way across test modules.
//! - Provide a dedicated error type [`ELError`] and result alias
//!   [`ELResult`] for statistical tests, plus a conversion layer to
//!   Python exceptions when the `python-bindings` feature is enabled.
//!
//! Invariants & assumptions
//! ------------------------
//! - Time-series inputs for test routines are expected to be finite,
//!   real-valued residuals or demeaned observations; modules call
//!   [`validate_input`] before performing any lag-based computations.
//! - Statistical tests in this subtree report failures via [`ELResult`]
//!   and never panic on user-facing invalid inputs; panics indicate
//!   programming errors (e.g., out-of-range indexing not caught by
//!   validation).
//! - [`ELError`] variants are small and cloneable so they can be used
//!   comfortably in both unit tests and higher-level orchestration code.
//! - At the Python boundary, all [`ELError`] values are mapped into a
//!   single exception class (`PyValueError`) with the Rust `Display`
//!   message preserved verbatim.
//!
//! Conventions
//! -----------
//! - This subtree is focused on *statistical tests*; model-specific error
//!   types (e.g., duration or optimization errors) live in their own
//!   `errors` modules under the relevant subtrees.
//! - Error messages are phrased in terms of domain constraints such as
//!   “q must be positive” or “1 ≤ d < n” rather than low-level details.
//! - Public entry points for tests (e.g.,
//!   [`ELOutcome::escanciano_lobato`](escanciano_lobato::ELOutcome::escanciano_lobato))
//!   are thin wrappers that delegate shape checks to [`validate_input`]
//!   and propagate [`ELError`] via [`ELResult`].
//!
//! Downstream usage
//! ----------------
//! - Typical Rust code imports the main surface as:
//!
//!   ```rust
//!   use rust_timeseries::statistical_tests::{ELOutcome, ELResult};
//!
//!   let outcome: ELOutcome = ELOutcome::escanciano_lobato(&residuals, q, d)?;
//!   ```
//!
//!   and only refers to `statistical_tests::errors` or
//!   `statistical_tests::validation` directly when matching on
//!   [`ELError`] or reusing [`validate_input`].
//! - Model-level diagnostics (e.g., for duration models) are expected to
//!   call [`ELOutcome::escanciano_lobato`](escanciano_lobato::ELOutcome::escanciano_lobato)
//!   on residual series and then report the selected lag, robust
//!   Box–Pierce statistic, and p-value from [`ELOutcome`].
//! - Python bindings expose thin wrappers around the same Rust entry
//!   points; they rely on `From<ELError> for PyErr` to raise `ValueError`
//!   instances instead of returning [`ELResult`] explicitly.
//!
//! Testing notes
//! -------------
//! - Unit tests in [`errors`] verify `Display` messages, payload
//!   embedding for [`ELError`] variants, and the PyO3 conversion path.
//! - Unit tests in [`validation`] exercise all branches of
//!   [`validate_input`], including insufficient data, non-finite values,
//!   invalid `q`, and invalid `d`.
//! - Unit tests in [`escanciano_lobato`] cover low-level helper
//!   correctness (γ̂ⱼ, τ̂ⱼ, ρ̃ⱼ², Qₚ*, p̃), monotonicity of Qₚ*, and
//!   end-to-end behavior of
//!   [`ELOutcome::escanciano_lobato`](escanciano_lobato::ELOutcome::escanciano_lobato)
//!   on small synthetic series.

pub mod errors;
pub mod escanciano_lobato;
pub mod validation;

// ---- Re-exports (primary public surface) ----------------------------------

pub use self::errors::{ELError, ELResult};
pub use self::escanciano_lobato::ELOutcome;
pub use self::validation::validate_input;

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::statistical_tests::prelude::*;
//
// to import the main statistical-testing surface in a single line.

pub mod prelude {
    pub use super::errors::{ELError, ELResult};
    pub use super::escanciano_lobato::ELOutcome;
}
