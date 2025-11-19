# Module doc:
//! <Module name> — <short phrase about what this module does.>
//! 
//! Purpose
//! -------
//! <What this module does and why it exists. One–three sentences.>
//! 
//! Key behaviors
//! -------------
//! - <Main responsibility / operation #1.>
//! - <Main responsibility / operation #2.>
//! - <Important side effects (e.g., logging, I/O, FFI).>
//! 
//! Invariants & assumptions
//! ------------------------
//! - <Invariants the module enforces (e.g., “durations > 0”).>
//! - <Assumptions about caller behavior (e.g., “must be called with sorted input”).>
//! - <Any domain constraints (e.g., stationarity, positivity).>
//! 
//! Conventions
//! -----------
//! - <Indexing conventions (0-based vs 1-based, [start, end) windows, etc.).>
//! - <Units, timezones, scaling schemes.>
//! - <Panics vs error returns policy.>
//! 
//! Downstream usage
//! ----------------
//! - <How other modules are expected to use this module.>
//! - <What is considered public surface vs internal helpers.>
//! - <Example: “construct ACDData here, pass into duration::models::acd”.>
//! 
//! Testing notes
//! -------------
//! - <What is covered by unit tests in this module.>
//! - <What is covered elsewhere by integration tests / Python tests.>

# Function doc:
/// <One-sentence summary of what this function does.>
///
/// Parameters
/// ----------
/// - `param1`: <type>
///   <Description: meaning, units, constraints.>
/// - `param2`: <type>
///   <Description: meaning, units, constraints.>
///
/// Returns
/// -------
/// <ReturnType>
///   <What is returned and under what conditions.>
///
/// Errors
/// ------
/// - `<ErrorType>`
///   <When this error is returned and why.>
/// - `<OtherErrorType>`
///   <Optional additional cases.>
///
/// Panics
/// ------
/// - <When this function may panic, or “Never panics.”>
///
/// Safety
/// ------
/// - <Only for `unsafe fn`: what the caller must guarantee.>
///
/// Notes
/// -----
/// - <Important implementation details, numerical tricks, guards.>
/// - <Performance characteristics, allocation behavior, etc.>
///
/// Examples
/// --------
/// ```rust
/// # use <crate path>::<fn_name>;
/// # // adjust imports as needed
/// let result = <fn_name>(/* args */);
/// # assert!(/* invariants about result */);
/// ```

# Type doc:
/// <Type name> — <short phrase about what this type represents.>
///
/// Purpose
/// -------
/// <What this type represents and why it exists. One–two sentences.>
///
/// Key behaviors
/// -------------
/// - <Behavior #1: “Holds ACD duration data and enforces positivity.”>
/// - <Behavior #2: “Provides accessors but does not mutate internal invariants.”>
///
/// Parameters
/// ----------
/// <If constructed via `new(...)`, list the constructor parameters here.>
/// - `param1`: <type>
///   <Meaning, units, constraints.>
/// - `param2`: <type>
///   <Meaning, units, constraints.>
///
/// Fields
/// ------
/// - `field1`: <type>
///   <What it stores, units, constraints.>
/// - `field2`: <type>
///   <What it stores, when it changes, whether it’s cached or derived.>
///
/// Invariants
/// ----------
/// - <List invariants that must hold (e.g., “length > 0”, “values > 0”).>
/// - <Call out relationships between fields (e.g., sums, bounds).>
///
/// Performance
/// -----------
/// - <Any relevant notes (e.g., no heap allocations after construction, O(n) copy).>
///
/// Notes
/// -----
/// - <Thread-safety / Send/Sync expectations.>
/// - <How this type is expected to be used by other modules.>


# Enum doc:
/// <Enum name> — <short phrase about the variants.>
///
/// Purpose
/// -------
/// <What this enum models: e.g., innovation family, kernel type.>
///
/// Variants
/// --------
/// - `<Variant1>`
///   <What this variant means; any parameter constraints if tuple struct.>
/// - `<Variant2(param: f64)>`
///   <Interpretation, units, constraints.>
///
/// Invariants
/// ----------
/// - <Any global constraints across variants.>
///
/// Notes
/// -----
/// - <How downstream code is expected to pattern-match on this enum.>


# Trait doc:
/// <Trait name> — <short phrase about the behavior.>
///
/// Purpose
/// -------
/// <What capability this trait abstracts (e.g., a log-likelihood, kernel, etc.).>
///
/// Required methods
/// ----------------
/// - `method1(&self, ...) -> ...`
///   <Contract and guarantees.>
/// - `method2(&self, ...) -> ...`
///   <Contract and guarantees.>
///
/// Blanket impls / auto traits
/// ---------------------------
/// - <Mention if it’s typically used as a trait object, generic bound, etc.>
///
/// Notes
/// -----
/// - <Anything about object safety, lifetimes, or common implementations.>


# Test mod doc:
#[cfg(test)]
mod tests {
    use super::*;
    // <Other imports as needed.>

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - <What invariants / behaviors this module enforces.>
    // - <Edge cases we care about (e.g., empty series, NaNs, out-of-range t0).>
    //
    // They intentionally DO NOT cover:
    // - <Things tested in integration tests or Python-level tests.>
    // -------------------------------------------------------------------------

    // <test functions go here>
}

# Single test doc:
#[test]
// Purpose
// -------
// <One-liner: what behavior this test is asserting.>
//
// Given
// -----
// - <Initial state / inputs.>
//
// Expect
// ------
// - <Expected outcome (Ok/Err, value, invariant, etc.).>
fn <unit_under_test>_<condition>_<expected_behavior>() {
    // Arrange
    // <set up inputs>

    // Act
    let result = /* call function under test */;

    // Assert
    // <assertions about the result>
    // e.g., assert!(result.is_ok());
    //       assert_eq!(result.unwrap_err(), ACDError::EmptySeries);
}


# Integration test doc:
//! Integration tests for <feature or module>.
//!
//! Purpose
//! -------
//! - <What scenario / pipeline we are validating end-to-end.>
//!
//! Coverage
//! --------
//! - <Which public APIs are exercised.>
//! - <Which invariants we rely on from multiple modules.>
//!
//! Exclusions
//! ----------
//! - <What we intentionally leave to unit tests or Python-level tests.>

use rust_timeseries::/* whatever you expose */;

#[test]
// Purpose
// -------
// Ensure <high-level behavior> works across the Rust–Python boundary / full stack.
fn <scenario_name>_behaves_as_expected() {
    // ...
}

