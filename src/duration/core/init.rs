//! Initialization policies for pre-sample lags in ACD(p, q) — control how ψ and
//! duration lags are seeded.
//!
//! Purpose
//! -------
//! Provide a small set of initialization policies for the pre-sample ψ and
//! duration lags required by ACD(p, q) recursions. This centralizes how
//! pre-sample state is seeded so that estimation and forecasting code can
//! choose between unconditional-mean, sample-mean, fixed-scalar, or fully
//! specified vector initializations in a uniform way.
//!
//! Key behaviors
//! -------------
//! - Represent pre-sample lag initialization as an explicit policy via
//!   [`Init`], including unconditional mean, sample mean, fixed scalar, and
//!   explicit vector options.
//! - Validate fixed-scalar and fixed-vector policies to ensure that all lag
//!   values are finite and strictly positive, and that vector lengths agree
//!   with the requested ACD(p, q) order.
//! - Surface invalid configurations as typed errors (`ACDError`) instead of
//!   panicking, allowing callers to handle bad inputs gracefully.
//!
//! Invariants & assumptions
//! ------------------------
//! - All pre-sample ψ and duration lags must be finite and strictly positive.
//! - For vector-based initialization, ψ-lag and duration-lag vectors must have
//!   lengths matching `p` and `q` respectively.
//! - The caller is responsible for passing consistent `(p, q)` values that
//!   match the model configuration and any downstream recursion logic.
//! - This module does not compute the unconditional mean μ or sample mean x̄;
//!   it only encodes *how* those quantities will be used once available.
//!
//! Conventions
//! -----------
//! - ψ denotes the conditional mean duration process of an ACD model; duration
//!   lags correspond to observed inter-arrival times τ.
//! - Initialization policies are modeled as an enum [`Init`], which is carried
//!   into model construction or recursion code to determine how pre-sample
//!   lags are filled.
//! - Invalid initialization inputs return `ACDError` variants rather than
//!   panicking; callers are expected to propagate or handle these errors.
//!
//! Downstream usage
//! ----------------
//! - Choose an [`Init`] variant at model setup time (e.g., from configuration
//!   or API arguments) and pass it into ACD parameter / recursion builders.
//! - Use `Init::uncond_mean()` or `Init::sample_mean()` when pre-sample lags
//!   should be derived from unconditional or sample moments of the data.
//! - Use `Init::fixed(...)` or `Init::fixed_vector(...)` when you want fully
//!   explicit control over pre-sample state (e.g., warm-starting from a prior
//!   run).
//!
//! Testing notes
//! -------------
//! - Unit tests in this module verify that:
//!   - `Init::fixed` accepts finite, strictly positive scalars and rejects
//!     non-finite or non-positive values with `ACDError::InvalidInitFixed`.
//!   - `Init::fixed_vector` enforces length constraints for ψ and duration
//!     lags and propagates element-level validation errors from
//!     `validate_psi_lags` / `validate_duration_lags`.
//! - End-to-end behavior of these policies in ACD recursions is covered by
//!   higher-level estimation / forecasting tests rather than here.
use crate::duration::{
    core::validation::{validate_duration_lags, validate_psi_lags},
    errors::{ACDError, ACDResult},
};
use ndarray::Array1;

/// Init — policies for seeding pre-sample ψ and duration lags in ACD(p, q).
///
/// Purpose
/// -------
/// Encode how the pre-sample ψ and duration lags required by an ACD(p, q)
/// recursion should be initialized, ranging from moment-based schemes
/// (unconditional mean, sample mean) to explicit fixed values and fully
/// specified vectors.
///
/// Key behaviors
/// -------------
/// - Represents a discrete set of initialization strategies (`UncondMean`,
///   `SampleMean`, `Fixed`, `FixedVector`) that can be chosen at model setup
///   time.
/// - Allows downstream code to branch on initialization policy without
///   re-encoding configuration details or ad-hoc flags.
/// - Enables both simple “plug-and-play” initializations and advanced
///   workflows where pre-sample state is carried over from previous runs.
///
/// Parameters
/// ----------
/// Constructed via:
/// - `Init::uncond_mean()`
///   Use the model’s unconditional mean μ for all ψ and duration lags.
/// - `Init::sample_mean()`
///   Use the sample mean x̄ of the observed durations for all lags.
/// - `Init::fixed(value: f64)`
///   Use a strictly positive scalar for all ψ and duration lags (validated).
/// - `Init::fixed_vector(psi_lags: Array1<f64>, duration_lags: Array1<f64>, p: usize, q: usize)`
///   Use explicit pre-sample vectors for ψ and duration lags (validated for
///   length and positivity).
///
/// Variants
/// --------
/// - `UncondMean`
///   Pre-sample ψ and τ lags are filled with the unconditional mean μ of the
///   model’s duration process.
/// - `SampleMean`
///   Pre-sample ψ and τ lags are filled with the sample mean x̄ of the
///   observed durations.
/// - `Fixed(f64)`
///   A single strictly positive scalar used for all ψ and τ lags. The stored
///   value is validated by `Init::fixed`.
/// - `FixedVector { psi_lags: Array1<f64>, duration_lags: Array1<f64> }`
///   Fully explicit pre-sample ψ and τ lag vectors, whose lengths and
///   element-wise properties are validated by `Init::fixed_vector`.
///
/// Invariants
/// ----------
/// - For `Fixed`, the inner scalar must be finite and strictly positive.
/// - For `FixedVector`, `psi_lags` and `duration_lags` must:
///   - have lengths matching the model orders `p` and `q`, and
///   - contain only finite, strictly positive entries.
/// - `UncondMean` and `SampleMean` are abstract policies; the actual mean
///   values are computed and applied by downstream code.
///
/// Notes
/// -----
/// - This enum is designed to be part of the public configuration surface for
///   ACD models.
/// - Pattern-matching on `Init` in estimation / forecasting code should be
///   exhaustive, so new policies added in the future will trigger compiler
///   warnings where they need to be handled.
#[derive(Debug, Clone, PartialEq)]
pub enum Init {
    /// Use the unconditional mean μ for all ψ and duration lags.
    UncondMean,
    /// Use the sample mean of the observed durations x̄ for all lags.
    SampleMean,
    /// Use a strictly positive fixed scalar for all lags.
    Fixed(f64),
    /// Use explicitly provided pre-sample vectors.
    ///
    /// - `psi_lags`: length p, supplies ψ_{-1}..ψ_{-p}
    /// - `duration_lags`: length q, supplies τ_{-1}..τ_{-q}
    FixedVector { psi_lags: Array1<f64>, duration_lags: Array1<f64> },
}

impl Init {
    /// Initialize all pre-sample ψ and duration lags with the unconditional mean μ.
    ///
    /// Parameters
    /// ----------
    /// - *(none)*
    ///   This constructor represents a policy choice; the actual unconditional mean
    ///   μ is computed elsewhere (e.g., from model parameters) and applied by
    ///   downstream code.
    ///
    /// Returns
    /// -------
    /// Init
    ///   An `Init::UncondMean` value indicating that all pre-sample ψ and τ lags
    ///   should be filled with the model’s unconditional mean μ.
    ///
    /// Errors
    /// ------
    /// - Never returns an error; this is a plain const constructor.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional safety requirements are imposed.
    ///
    /// Notes
    /// -----
    /// - This function does not compute μ itself; it only selects the policy.
    ///   The actual numeric value is injected later when the recursion is
    ///   initialized.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::init::Init;
    /// let init_policy = Init::uncond_mean();
    /// // later: use `init_policy` when building ACD pre-sample lags
    /// ```
    pub const fn uncond_mean() -> Self {
        Init::UncondMean
    }

    /// Initialize all pre-sample ψ and duration lags with the sample mean x̄.
    ///
    /// Parameters
    /// ----------
    /// - *(none)*
    ///   This constructor captures the policy choice; the actual sample mean x̄ is
    ///   computed from observed durations by downstream code.
    ///
    /// Returns
    /// -------
    /// Init
    ///   An `Init::SampleMean` value indicating that all pre-sample ψ and τ lags
    ///   should be filled with the sample mean x̄ of the observed durations.
    ///
    /// Errors
    /// ------
    /// - Never returns an error; this is a plain const constructor.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional safety requirements are imposed.
    ///
    /// Notes
    /// -----
    /// - This function does not compute x̄ itself; it only encodes that the
    ///   initialization should be based on the sample mean of the input data.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::init::Init;
    /// let init_policy = Init::sample_mean();
    /// // later: use `init_policy` when constructing ACD pre-sample lags
    /// ```
    pub const fn sample_mean() -> Self {
        Init::SampleMean
    }

    /// Initialize all pre-sample ψ and duration lags with a fixed positive scalar.
    ///
    /// Parameters
    /// ----------
    /// - `value`: `f64`
    ///   Scalar used to fill all pre-sample ψ and τ lags. Must be finite and
    ///   strictly positive (`value > 0.0`) to be accepted.
    ///
    /// Returns
    /// -------
    /// ACDResult<Init>
    ///   - `Ok(Init::Fixed(value))` when `value` is finite and strictly positive.
    ///   - `Err(ACDError::InvalidInitFixed { value })` when `value` is non-finite
    ///     or non-positive.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidInitFixed`
    ///   Returned when `value` is not finite (`!value.is_finite()`) or `value <= 0.0`,
    ///   as such a scalar would violate the positivity requirement for pre-sample
    ///   lags.
    ///
    /// Panics
    /// ------
    /// - Never panics; invalid scalars are reported via `ACDError::InvalidInitFixed`.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional safety requirements are imposed beyond
    ///   choosing a sensible positive scalar.
    ///
    /// Notes
    /// -----
    /// - This constructor is useful when you want deterministic, user-controlled
    ///   pre-sample lags (e.g., warm-starts or debugging scenarios).
    /// - The same scalar is applied to both ψ and τ lags; for more granular
    ///   control, use `Init::fixed_vector`.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::init::Init;
    /// # use rust_timeseries::duration::errors::ACDError;
    /// // Valid fixed scalar
    /// let init = Init::fixed(1.0).unwrap();
    ///
    /// // Invalid fixed scalar (non-positive)
    /// let err = Init::fixed(0.0).unwrap_err();
    /// if let ACDError::InvalidInitFixed { value } = err {
    ///     assert_eq!(value, 0.0);
    /// }
    /// ```
    pub fn fixed(value: f64) -> ACDResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(ACDError::InvalidInitFixed { value });
        }
        Ok(Init::Fixed(value))
    }

    /// Initialize pre-sample ψ and duration lags with explicit vectors.
    ///
    /// Parameters
    /// ----------
    /// - `psi_lags`: `Array1<f64>`
    ///   Vector of pre-sample ψ lags with expected length `p`. All entries must be
    ///   finite and strictly positive. Conceptually supplies ψ_{-1}, …, ψ_{-p}.
    /// - `duration_lags`: `Array1<f64>`
    ///   Vector of pre-sample duration lags with expected length `q`. All entries
    ///   must be finite and strictly positive. Conceptually supplies τ_{-1}, …,
    ///   τ_{-q}.
    /// - `p`: `usize`
    ///   Expected number of ψ-lags for the ACD(p, q) model; used to validate
    ///   `psi_lags.len()`.
    /// - `q`: `usize`
    ///   Expected number of duration lags for the ACD(p, q) model; used to validate
    ///   `duration_lags.len()`.
    ///
    /// Returns
    /// -------
    /// ACDResult<Init>
    ///   - `Ok(Init::FixedVector { psi_lags, duration_lags })` when lengths and
    ///     element-wise properties satisfy all lag validation rules.
    ///   - `Err(ACDError)` if length or element-wise validation fails.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidPsiLength`
    ///   Returned when `psi_lags.len()` does not match the expected ψ order `p`.
    /// - `ACDError::InvalidDurationLength`
    ///   Returned when `duration_lags.len()` does not match the expected duration
    ///   order `q`.
    /// - Other `ACDError` variants from `validate_psi_lags` / `validate_duration_lags`
    ///   Returned when any lag value is non-finite or non-positive (e.g., NaN,
    ///   ±∞, or `<= 0.0`).
    ///
    /// Panics
    /// ------
    /// - Never panics; invalid length or element-wise properties are reported via
    ///   `ACDError`.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only ensure that `(p, q)` and the vector
    ///   contents are consistent with their intended model configuration.
    ///
    /// Notes
    /// -----
    /// - This constructor delegates detailed length and element-wise checks to
    ///   `validate_psi_lags` and `validate_duration_lags`, centralizing validation
    ///   logic for pre-sample lags.
    /// - Use this when you need full control over pre-sample state, such as
    ///   continuing from a previous fit or injecting domain-specific initial
    ///   conditions.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use ndarray::array;
    /// # use rust_timeseries::duration::core::init::Init;
    /// # use rust_timeseries::duration::errors::ACDError;
    /// let psi_lags = array![1.0, 1.1];
    /// let duration_lags = array![0.9, 1.0];
    /// let p = 2;
    /// let q = 2;
    ///
    /// let init = Init::fixed_vector(psi_lags.clone(), duration_lags.clone(), p, q).unwrap();
    /// // `init` now encodes explicit pre-sample ψ and τ lags
    /// ```
    pub fn fixed_vector(
        psi_lags: Array1<f64>, duration_lags: Array1<f64>, p: usize, q: usize,
    ) -> ACDResult<Self> {
        validate_psi_lags(&psi_lags, p)?;
        validate_duration_lags(&duration_lags, q)?;

        Ok(Init::FixedVector { psi_lags, duration_lags })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Construction behavior of the `Init` policies (`uncond_mean`, `sample_mean`,
    //   `fixed`, and `fixed_vector`).
    // - Validation invariants for fixed-scalar and fixed-vector initialization:
    //   positivity, finiteness, and length constraints (p, q).
    //
    // They intentionally DO NOT cover:
    // - How these policies are applied inside ACD ψ-recursions or log-likelihoods
    //   (those behaviors are validated in higher-level estimation/forecast tests).
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Ensure that `Init::uncond_mean` constructs the `UncondMean` policy variant.
    //
    // Given
    // -----
    // - No inputs; `Init::uncond_mean` is a const constructor that encodes a
    //   policy choice (use unconditional mean μ).
    //
    // Expect
    // ------
    // - The returned value is exactly `Init::UncondMean`.
    fn init_uncond_mean_constructs_uncond_mean_variant() {
        // Arrange
        // (no setup needed)

        // Act
        let init_policy = Init::uncond_mean();

        // Assert
        assert_eq!(init_policy, Init::UncondMean);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `Init::sample_mean` constructs the `SampleMean` policy variant.
    //
    // Given
    // -----
    // - No inputs; `Init::sample_mean` is a const constructor that encodes a
    //   policy choice (use sample mean x̄).
    //
    // Expect
    // ------
    // - The returned value is exactly `Init::SampleMean`.
    fn init_sample_mean_constructs_sample_mean_variant() {
        // Arrange
        // (no setup needed)

        // Act
        let init_policy = Init::sample_mean();

        // Assert
        assert_eq!(init_policy, Init::SampleMean);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed` accepts a finite, strictly positive scalar.
    //
    // Given
    // -----
    // - `value = 1.0`, which is finite and > 0.0.
    //
    // Expect
    // ------
    // - `Init::fixed(value)` returns `Ok(Init::Fixed(1.0))`.
    fn init_fixed_accepts_positive_finite_scalar() {
        // Arrange
        let value = 1.0_f64;

        // Act
        let result = Init::fixed(value);

        // Assert
        assert_eq!(result, Ok(Init::Fixed(value)));
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed` rejects non-positive scalars.
    //
    // Given
    // -----
    // - `value = 0.0` and `value = -1.0`, which violate the strict positivity
    //   constraint for pre-sample lags.
    //
    // Expect
    // ------
    // - `Init::fixed(0.0)` and `Init::fixed(-1.0)` both return
    //   `Err(ACDError::InvalidInitFixed { value })`.
    fn init_fixed_rejects_non_positive_scalars() {
        // Arrange
        let zero_value = 0.0_f64;
        let negative_value = -1.0_f64;

        // Act
        let zero_result = Init::fixed(zero_value);
        let negative_result = Init::fixed(negative_value);

        // Assert
        assert_eq!(zero_result.unwrap_err(), ACDError::InvalidInitFixed { value: zero_value });
        assert_eq!(
            negative_result.unwrap_err(),
            ACDError::InvalidInitFixed { value: negative_value }
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed` rejects non-finite scalars.
    //
    // Given
    // -----
    // - `value = f64::INFINITY`, which is not finite.
    //
    // Expect
    // ------
    // - `Init::fixed(value)` returns `Err(ACDError::InvalidInitFixed { value })`.
    fn init_fixed_rejects_non_finite_scalars() {
        // Arrange
        let inf_value = f64::INFINITY;
        let nan_value = f64::NAN;

        // Act
        let inf_result = Init::fixed(inf_value);
        let nan_result = Init::fixed(nan_value);

        // Assert
        assert_eq!(inf_result.unwrap_err(), ACDError::InvalidInitFixed { value: inf_value });

        // For NaN we cannot use direct equality on the inner value.
        if let ACDError::InvalidInitFixed { value } = nan_result.unwrap_err() {
            assert!(value.is_nan());
        } else {
            panic!("expected InvalidInitFixed error for NaN input");
        }
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `Init::fixed_vector` accepts valid ψ and duration lag vectors.
    //
    // Given
    // -----
    // - `psi_lags` of length `p = 2` with finite, strictly positive entries.
    // - `duration_lags` of length `q = 3` with finite, strictly positive entries.
    //
    // Expect
    // ------
    // - `Init::fixed_vector` returns `Ok(Init::FixedVector { .. })` with the
    //   provided lag vectors.
    fn init_fixed_vector_accepts_valid_lag_vectors() {
        // Arrange
        let psi_lags = array![1.0, 1.2];
        let duration_lags = array![0.9, 1.0, 1.1];
        let p = 2_usize;
        let q = 3_usize;

        // Act
        let result = Init::fixed_vector(psi_lags.clone(), duration_lags.clone(), p, q);

        // Assert
        let init = result.expect("fixed_vector should accept valid lengths and values");
        match init {
            Init::FixedVector { psi_lags: psi_out, duration_lags: duration_out } => {
                assert_eq!(psi_out, psi_lags);
                assert_eq!(duration_out, duration_lags);
            }
            _ => panic!("expected Init::FixedVector variant"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed_vector` rejects mismatched ψ-lag length.
    //
    // Given
    // -----
    // - `psi_lags` of length 1 but `p = 2`.
    // - `duration_lags` of correct length `q = 2`.
    //
    // Expect
    // ------
    // - `Init::fixed_vector` returns `Err(ACDError::InvalidPsiLength { expected: 2, actual: 1 })`.
    fn init_fixed_vector_rejects_mismatched_psi_length() {
        // Arrange
        let psi_lags = array![1.0]; // len = 1
        let duration_lags = array![1.0, 1.1]; // len = 2
        let p = 2_usize;
        let q = 2_usize;

        // Act
        let result = Init::fixed_vector(psi_lags, duration_lags, p, q);

        // Assert
        assert_eq!(result.unwrap_err(), ACDError::InvalidPsiLength { expected: p, actual: 1 });
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed_vector` rejects mismatched duration-lag length.
    //
    // Given
    // -----
    // - `psi_lags` of correct length `p = 2`.
    // - `duration_lags` of length 1 but `q = 2`.
    //
    // Expect
    // ------
    // - `Init::fixed_vector` returns
    //   `Err(ACDError::InvalidDurationLength { expected: 2, actual: 1 })`.
    fn init_fixed_vector_rejects_mismatched_duration_length() {
        // Arrange
        let psi_lags = array![1.0, 1.1]; // len = 2
        let duration_lags = array![1.0]; // len = 1
        let p = 2_usize;
        let q = 2_usize;

        // Act
        let result = Init::fixed_vector(psi_lags, duration_lags, p, q);

        // Assert
        assert_eq!(result.unwrap_err(), ACDError::InvalidDurationLength { expected: q, actual: 1 });
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed_vector` rejects non-positive ψ-lag entries.
    //
    // Given
    // -----
    // - `psi_lags` of length `p = 2` where the second entry is 0.0.
    // - `duration_lags` of correct length and strictly positive.
    //
    // Expect
    // ------
    // - `Init::fixed_vector` returns
    //   `Err(ACDError::InvalidPsiLags { index: 1, value: 0.0 })`.
    fn init_fixed_vector_rejects_non_positive_psi_lags() {
        // Arrange
        let psi_lags = array![1.0, 0.0];
        let duration_lags = array![1.0, 1.1];
        let p = 2_usize;
        let q = 2_usize;

        // Act
        let result = Init::fixed_vector(psi_lags, duration_lags, p, q);

        // Assert
        assert_eq!(result.unwrap_err(), ACDError::InvalidPsiLags { index: 1, value: 0.0 });
    }

    #[test]
    // Purpose
    // -------
    // Verify that `Init::fixed_vector` rejects non-positive duration-lag entries.
    //
    // Given
    // -----
    // - `psi_lags` of correct length and strictly positive.
    // - `duration_lags` of length `q = 2` where the first entry is -1.0.
    //
    // Expect
    // ------
    // - `Init::fixed_vector` returns
    //   `Err(ACDError::InvalidDurationLags { index: 0, value: -1.0 })`.
    fn init_fixed_vector_rejects_non_positive_duration_lags() {
        // Arrange
        let psi_lags = array![1.0, 1.1];
        let duration_lags = array![-1.0, 1.0];
        let p = 2_usize;
        let q = 2_usize;

        // Act
        let result = Init::fixed_vector(psi_lags, duration_lags, p, q);

        // Assert
        assert_eq!(result.unwrap_err(), ACDError::InvalidDurationLags { index: 0, value: -1.0 });
    }
}
