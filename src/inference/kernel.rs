//! inference::kernel — HAC tapers and plug-in bandwidth selection.
//!
//! Purpose
//! -------
//! Provide kernel tapers and data-driven bandwidth selection for
//! heteroskedasticity and autocorrelation consistent (HAC) covariance
//! estimation. This module encapsulates the choice of HAC kernel family
//! and implements Andrews-style plug-in bandwidth rules based on
//! column-wise AR(1) fits.
//!
//! Key behaviors
//! -------------
//! - Enumerate common HAC kernel families via [`KernelType`] and provide
//!   per-lag kernel weights `w(x)` for `x = k/(L+1)`.
//! - Select a plug-in bandwidth `L` using AR(1)-based Andrews formulas,
//!   with kernel-specific exponents and constants.
//! - Fall back to rule-of-thumb bandwidths `⌊n^{1/4}⌉` when the plug-in
//!   calculation is unstable (e.g., near-unit root or numerically tiny
//!   denominators).
//!
//! Invariants & assumptions
//! ------------------------
//! - Input series matrices are `n×p` with `n ≥ 1` and `p ≥ 1`. For the
//!   plug-in bandwidth to be meaningful, each column used in AR(1) fits
//!   should have at least two observations (`n ≥ 2`).
//! - The same series matrix passed into [`KernelType::optimal_bandwidth`]
//!   should be the one used for HAC aggregation (e.g., centered if
//!   upstream options request centering).
//! - AR(1) coefficients `φ` must satisfy a strict stationarity margin
//!   `|φ| < 1 − STATIONARITY_MARGIN`; violations are treated as plug-in
//!   failures and surfaced as [`InferenceError::StationarityViolated`].
//! - Plug-in parameter `ord` is restricted to `1` (Bartlett) or `2`
//!   (Parzen / QuadraticSpectral); other values are rejected as
//!   [`InferenceError::OrderNotSupported`].
//!
//! Conventions
//! -----------
//! - Rows index time; columns index series/components (scores, residuals,
//!   or other per-observation quantities).
//! - Kernel weights are evaluated at `x = k/(L+1)` so that `x ∈ [0, 1)`
//!   even at the maximum lag `k = L`, avoiding division by zero in
//!   compact-support kernels.
//! - Plug-in bandwidths are returned as non-negative integers `L` and
//!   are expected to be truncated by callers to `L ≤ n−1`.
//! - Failures in the plug-in calculation (stationarity, unsupported
//!   order, numerically tiny denominators, or AR(1) estimation errors)
//!   are reported as [`InferenceError`] and typically trigger a fallback
//!   rule-of-thumb bandwidth in callers.
//!
//! Downstream usage
//! ----------------
//! - [`KernelType`] is embedded in HAC configuration objects (e.g.,
//!   [`crate::inference::hac::HACOptions`]) to select both tapers and
//!   plug-in regime.
//! - [`KernelType::weight`] is used by HAC aggregation routines to
//!   compute per-lag weights `w_k` for lags `k = 0,…,L`.
//! - [`KernelType::optimal_bandwidth`] is invoked on the score matrix
//!   (centered if applicable) to choose `L` before building HAC
//!   covariance estimates.
//! - The helper [`calc_opt_bandwidth_param`] is internal to this module
//!   and not expected to be called directly by library users.
//!
//! Testing notes
//! -------------
//! - Unit tests cover:
//!   - Basic shape and symmetry of kernel weights for each `KernelType`
//!     (e.g., taper to zero at |x| > 1 for compact-support kernels).
//!   - IID semantics (`KernelType::IID` gives weight 1 at `x = 0` and 0
//!     elsewhere).
//!   - Plug-in bandwidth behavior on simple synthetic AR(1) series and
//!     fallback to `n^{1/4}` when `calc_opt_bandwidth_param` fails.
//! - Integration tests verify that:
//!   - Bandwidths scale with sample size in the expected `n^{γ}` regime
//!     for each kernel.
//!   - Near-unit-root behavior in the series triggers conservative
//!     fallbacks rather than unstable bandwidths.
use crate::{
    inference::errors::{InferenceError, InferenceResult},
    optimization::numerical_stability::transformations::{GENERAL_TOL, STATIONARITY_MARGIN},
};
use arima::estimate;
use ndarray::Array2;

/// KernelType — HAC taper family for score covariance estimation.
///
/// Purpose
/// -------
/// Represent the choice of HAC kernel used to down-weight higher lags in
/// score covariance estimation. The kernel family determines both the
/// per-lag weights `w_k` and the exponent/constant used in the
/// Andrews-style plug-in bandwidth rule.
///
/// Variants
/// --------
/// - `IID`
///   No serial correlation: only lag `k = 0` contributes. In the HAC
///   aggregator, this corresponds to the outer-product-of-gradients
///   estimator with bandwidth `L = 0`.
/// - `Bartlett`
///   Triangular (Newey–West) compact-support kernel with weights
///   `w(x) = 1 − |x|` for `|x| ≤ 1` and `0` otherwise. Plug-in bandwidth
///   uses order `q = 1`.
/// - `Parzen`
///   Smoother compact-support kernel. For `|x| ≤ 0.5`, `w(x)` is a cubic
///   polynomial; for `0.5 < |x| ≤ 1`, `w(x)` tapers as a cubic in
///   `(1 − |x|)`. Plug-in bandwidth uses order `q = 2`.
/// - `QuadraticSpectral`
///   Infinite-support kernel with high asymptotic efficiency, defined by
///   the standard quadratic spectral formula. Plug-in bandwidth uses
///   order `q = 2`.
///
/// Invariants
/// ----------
/// - The taper argument is interpreted as `x = k/(L+1)` with `k ≥ 0` and
///   integer bandwidth `L ≥ 0`. In HAC usage, `x ∈ [0, 1)` for
///   compact-support kernels.
/// - For `IID`, callers are expected to use bandwidth `L = 0` so that
///   only the lag `k = 0` contributes with weight `1.0`.
///
/// Notes
/// -----
/// - Downstream HAC routines pattern-match on `KernelType` to
///   choose both a plug-in order `q ∈ {1, 2}` and the `w(x)` formula.
/// - The infinite support of `QuadraticSpectral` is truncated at lag `L`
///   by the bandwidth choice; only lags `k = 0,…,L` are ever evaluated.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    IID,
    Bartlett,
    Parzen,
    QuadraticSpectral,
}

impl KernelType {
    /// Evaluate the kernel weight at the given taper argument.
    ///
    /// Parameters
    /// ----------
    /// - `input`: `f64`
    ///   Taper argument `x`, typically `x = k/(L+1)` where `k` is the lag
    ///   and `L` the bandwidth. In the HAC context, callers usually pass
    ///   non-negative `x` with `x ∈ [0, 1)` for compact-support kernels.
    ///
    /// Returns
    /// -------
    /// `f64`
    ///   Kernel value `w(x)`. For `KernelType::IID`, this is `1.0` when
    ///   `x == 0.0` and `0.0` otherwise; for other kernels, the value
    ///   follows the standard Bartlett, Parzen, or quadratic spectral
    ///   formulae.
    ///
    /// Errors
    /// ------
    /// - None  
    ///   This function always returns a finite `f64` for finite `input`.
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
    /// - For `IID` and `QuadraticSpectral`, this method treats `input == 0.0`
    ///   as a special case. In the HAC call path where `input = k/(L+1)`,
    ///   the lag-zero case is passed as exactly `0.0`, so an equality check
    ///   against `0.0` is safe.
    /// - For `Parzen` and `Bartlett`, inputs with `|x| > 1` are mapped to
    ///   `0.0`, as in the standard compact-support definitions used by
    ///   Newey–West and Andrews (1991).
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::inference::kernel::KernelType;
    /// let k = 1_usize;
    /// let l = 4_usize;
    /// let x = k as f64 / (l as f64 + 1.0);
    ///
    /// let w_bartlett = KernelType::Bartlett.weight(x);
    /// assert!(w_bartlett > 0.0 && w_bartlett < 1.0);
    ///
    /// let w_iid = KernelType::IID.weight(0.0);
    /// assert_eq!(w_iid, 1.0);
    /// ```
    pub fn weight(&self, input: f64) -> f64 {
        let abs_input = input.abs();
        match self {
            KernelType::IID => {
                if input == 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            KernelType::Bartlett => {
                if abs_input <= 1.0 {
                    1.0 - abs_input
                } else {
                    0.0
                }
            }
            KernelType::Parzen => {
                if abs_input <= 0.5 {
                    let abs_input_squared = abs_input * abs_input;
                    1.0 - 6.0 * abs_input_squared + 6.0 * abs_input * abs_input_squared
                } else if abs_input <= 1.0 {
                    2.0 * (1.0 - abs_input).powi(3)
                } else {
                    0.0
                }
            }
            KernelType::QuadraticSpectral => {
                if input == 0.0 {
                    1.0
                } else {
                    let pi_x = std::f64::consts::PI * input;
                    let trig_input = 6.0 * pi_x / 5.0;
                    (25.0 / (12.0 * (pi_x.powi(2))))
                        * ((trig_input).sin() / trig_input - (trig_input).cos())
                }
            }
        }
    }

    /// optimal_bandwidth — Andrews-style plug-in bandwidth for HAC.
    ///
    /// Purpose
    /// -------
    /// Select a non-negative integer bandwidth `L` for HAC estimation
    /// using Andrews-type plug-in rules specialized to the chosen kernel.
    /// For non-IID kernels, this routine fits AR(1) models column-wise
    /// and aggregates an `α(q)` parameter; if the plug-in fails, it
    /// falls back to a simple rule-of-thumb bandwidth `⌊n^{1/4}⌉`.
    ///
    /// Parameters
    /// ----------
    /// - `series_mat`: `&Array2<f64>`
    ///   `n×p` series matrix (rows=time, columns=components), typically
    ///   containing per-observation scores or residuals. Callers should
    ///   pass the same series that will be used for HAC aggregation
    ///   (e.g., centered if upstream options request centering).
    ///
    /// Returns
    /// -------
    /// `usize`
    ///   Recommended bandwidth `L ≥ 0`. For:
    ///   - `KernelType::IID`: always returns `0`.
    ///   - `KernelType::Bartlett`: uses `L ≈ 1.1447 · (n·α(1))^{1/3}`.
    ///   - `KernelType::Parzen`: uses `L ≈ 2.6614 · (n·α(2))^{1/5}`.
    ///   - `KernelType::QuadraticSpectral`: uses
    ///     `L ≈ 1.3221 · (n·α(2))^{1/5}`.
    /// - If `calc_opt_bandwidth_param` fails, returns
    ///   `⌊n^{1/4}⌉` as a conservative fallback.
    ///
    /// Errors
    /// ------
    /// - None  
    ///   This method does not propagate [`InferenceError`] directly; any
    ///   plug-in failures inside [`calc_opt_bandwidth_param`] are
    ///   absorbed and mapped to a fallback `⌊n^{1/4}⌉` bandwidth.
    ///
    /// Panics
    /// ------
    /// - Never panics under the documented invariants. A panic would
    ///   indicate a violation of assumptions in the AR(1) estimation
    ///   layer (e.g., invalid `series_mat` sizes).
    ///
    /// Safety
    /// ------
    /// - No `unsafe` code is used. Callers should ensure that `n` is
    ///   large enough for AR(1) estimation to be meaningful (`n ≥ 2`).
    ///
    /// Notes
    /// -----
    /// - The constants `1.1447`, `2.6614`, and `1.3221` are the Andrews
    ///   (1991, *Econometrica* 59(3):817–858) plug-in constants that minimize
    ///   the asymptotic mean squared error of the long-run variance estimator
    ///   for the Bartlett, Parzen, and quadratic spectral kernels,
    ///   respectively.
    /// - The returned bandwidth is **not** truncated to `n−1`; callers
    ///   are expected to apply `min(L, n−1)` when building HAC covariance
    ///   estimates.
    /// - The plug-in parameter `α(q)` is computed by
    ///   [`calc_opt_bandwidth_param`], which aggregates AR(1)-based
    ///   contributions from each column of `series_mat` and enforces both
    ///   stationarity and a minimum denominator size to avoid numerical
    ///   degeneracy.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use ndarray::array;
    /// # use rust_timeseries::inference::kernel::KernelType;
    /// let series = array![
    ///     [1.0_f64, 0.5],
    ///     [0.9,      0.4],
    ///     [1.1,      0.6],
    ///     [1.0,      0.5],
    /// ];
    /// let kernel = KernelType::Bartlett;
    /// let l = kernel.optimal_bandwidth(&series);
    /// assert!(l >= 0);
    /// ```
    pub fn optimal_bandwidth(&self, series_mat: &Array2<f64>) -> usize {
        let n = series_mat.nrows() as f64;
        match self {
            KernelType::IID => 0,
            KernelType::Bartlett => {
                let ord = 1;
                let alpha = match calc_opt_bandwidth_param(series_mat, ord) {
                    Ok(alpha) => alpha,
                    Err(_) => return (n.powf(1.0 / 4.0)).round() as usize,
                };
                (1.1447 * (n * alpha).powf(1.0 / 3.0)).round() as usize
            }
            KernelType::Parzen => {
                let ord = 2;
                let alpha = match calc_opt_bandwidth_param(series_mat, ord) {
                    Ok(alpha) => alpha,
                    Err(_) => return (n.powf(1.0 / 4.0)).round() as usize,
                };
                (2.6614 * (n * alpha).powf(1.0 / 5.0)).round() as usize
            }
            KernelType::QuadraticSpectral => {
                let ord = 2;
                let alpha = match calc_opt_bandwidth_param(series_mat, ord) {
                    Ok(alpha) => alpha,
                    Err(_) => return (n.powf(1.0 / 4.0)).round() as usize,
                };
                (1.3221 * (n * alpha).powf(1.0 / 5.0)).round() as usize
            }
        }
    }
}

/// calc_opt_bandwidth_param — Andrews plug-in α(q) from AR(1) fits.
///
/// Purpose
/// -------
/// Compute the kernel-specific plug-in parameter `α(q)` by aggregating
/// AR(1)-based contributions across the columns of a series matrix.
/// This parameter enters the Andrews bandwidth formulas used by
/// [`KernelType::optimal_bandwidth`].
///
/// Parameters
/// ----------
/// - `series_mat`: `&Array2<f64>`
///   `n×p` series matrix (rows=time, columns=components). Each column is
///   treated as a univariate time series to which an AR(1) model is
///   fitted via `arima::estimate::fit`.
/// - `ord`: `usize`
///   Plug-in order `q`. Currently:
///   - `1` corresponds to the Bartlett kernel (`KernelType::Bartlett`).
///   - `2` corresponds to Parzen / quadratic spectral kernels
///     (`KernelType::Parzen`, `KernelType::QuadraticSpectral`).
///
/// Returns
/// -------
/// `InferenceResult<f64>`
///   On success, a positive scalar `α(q)` used by the kernel-specific
///   bandwidth formulas. On failure, an [`InferenceError`] describing
///   the reason plug-in bandwidth selection is not reliable and callers
///   should fall back to a rule-of-thumb.
///
/// Errors
/// ------
/// - [`InferenceError::StationarityViolated`] { phi }
///   Returned if any column’s AR(1) estimate satisfies
///   `|φ| ≥ 1 − STATIONARITY_MARGIN`, indicating a near-unit-root or
///   nonstationary process for which the plug-in formulas are unstable.
/// - [`InferenceError::OrderNotSupported`] { ord }
///   Returned immediately if `ord` is not one of the supported values
///   (`1` for Bartlett, `2` for Parzen / quadratic spectral).
/// - [`InferenceError::DenominatorTooSmall`] { denominator }
///   Returned if the aggregated denominator in the `α(q)` formula is
///   numerically tiny (e.g., when there are fewer than two observations
///   per series or when all series contribute negligible variance),
///   which would make the plug-in unstable.
/// - `<other InferenceError variants>`
///   Propagated from ARIMA estimation calls (e.g., failures in
///   `estimate::fit` or `estimate::residuals`), wrapped as
///   [`InferenceError::Anyhow`] by the calling layer.
///
/// Panics
/// ------
/// - May panic if the residual length for any column is less than 2 and
///   the caller does not guard `n ≥ 2`, due to division by `n−1` in the
///   variance estimate. Callers should ensure `series_mat.nrows() ≥ 2`
///   before invoking this helper, or this function should explicitly
///   treat such cases as `DenominatorTooSmall`.
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that `series_mat`
///   contains finite values and is suitable for AR(1) fitting.
///
/// Notes
/// -----
/// - This helper implements the AR(1)-based plug-in parameter `α(q)`
///   used in Andrews (1991) for HAC bandwidth selection. For each
///   column, it:
///   1. Fits AR(1) coefficients `(intercept, φ)` to the series.
///   2. Enforces a strict stationarity margin via `STATIONARITY_MARGIN`.
///   3. Computes residuals and their sample variance `σ²` (with `n−1`
///      in the denominator) and then `σ⁴`.
///   4. Accumulates numerator and denominator terms involving `σ⁴` and
///      powers of `(1 − φ)` and `(1 + φ)` in the way prescribed for
///      `q = 1` (Bartlett) or `q = 2` (Parzen / quadratic spectral).
/// - If all columns are rejected (e.g., by stationarity) or contribute
///   only negligible variance, the aggregate denominator will fall
///   below [`GENERAL_TOL`], at which point the function returns
///   `DenominatorTooSmall` to signal that the plug-in bandwidth is not
///   reliable and callers should fall back to a rule-of-thumb.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::array;
/// # use rust_timeseries::inference::kernel::calc_opt_bandwidth_param;
/// # use rust_timeseries::inference::errors::InferenceResult;
/// let series = array![
///     [1.0_f64],
///     [0.9],
///     [1.1],
///     [1.0],
/// ];
/// let alpha: InferenceResult<f64> = calc_opt_bandwidth_param(&series, 1);
/// assert!(alpha.is_ok());
/// assert!(alpha.unwrap() > 0.0);
/// ```
fn calc_opt_bandwidth_param(series_mat: &Array2<f64>, ord: usize) -> InferenceResult<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let n = series_mat.nrows();

    if series_mat.nrows() < 2 {
        return Err(InferenceError::DenominatorTooSmall { denominator });
    }

    if ord != 1 && ord != 2 {
        return Err(InferenceError::OrderNotSupported { ord });
    }

    let mut scratch_col = Vec::with_capacity(n);

    for col in series_mat.columns() {
        scratch_col.clear();
        scratch_col.extend(col.iter().copied());

        let coeff = estimate::fit(&scratch_col, 1, 0, 0)?;
        let intercept = coeff[0];
        let phi = coeff[1];
        if phi.abs() >= 1.0 - STATIONARITY_MARGIN {
            return Err(InferenceError::StationarityViolated { phi });
        }
        let phi_squared = phi * phi;
        let residuals = estimate::residuals(&scratch_col, intercept, Some(&[phi]), None)?;
        let sigma2 = residuals.iter().map(|&e| e * e).sum::<f64>() / ((residuals.len() - 1) as f64);
        let sigma4 = sigma2 * sigma2;
        denominator += sigma4 / (1.0 - phi_squared).powi(4);
        let numerator_numerator = 4.0 * phi_squared * sigma4;
        numerator += match ord {
            1 => numerator_numerator / ((1.0 - phi).powi(6) * (1.0 + phi) * (1.0 + phi)),
            2 => numerator_numerator / (1.0 - phi).powi(8),
            _ => return Err(InferenceError::OrderNotSupported { ord }),
        };
    }
    if denominator < GENERAL_TOL {
        return Err(InferenceError::DenominatorTooSmall { denominator });
    }
    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Basic shape and semantics of kernel weights for each `KernelType`
    //   (IID mass at lag zero, compact support for Bartlett/Parzen, symmetry).
    // - Minimal sanity properties of the Andrews plug-in helper
    //   `calc_opt_bandwidth_param` (guard behavior, positive α on simple data).
    // - Fallback and kernel-specific behavior of `KernelType::optimal_bandwidth`
    //   (IID → L = 0, rule-of-thumb fallback when plug-in fails).
    //
    // They intentionally DO NOT cover:
    // - Asymptotic optimality of the bandwidth formulas.
    // - End-to-end HAC covariance estimation or score-construction flows.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `KernelType::IID` assigns unit mass at x = 0.0 and zero
    // weight at any non-zero taper argument.
    //
    // Given
    // -----
    // - An IID kernel instance.
    //
    // Expect
    // ------
    // - `weight(0.0)` returns 1.0.
    // - `weight(x)` returns 0.0 for small non-zero x.
    fn kerneltype_weight_iid_has_unit_mass_at_zero_only() {
        // Arrange
        let kernel = KernelType::IID;

        // Act
        let w0 = kernel.weight(0.0);
        let w_small = kernel.weight(1e-6);
        let w_neg = kernel.weight(-1e-3);

        // Assert
        assert_eq!(w0, 1.0);
        assert_eq!(w_small, 0.0);
        assert_eq!(w_neg, 0.0);
    }

    #[test]
    // Purpose
    // -------
    // Check that the Bartlett kernel is linear on [-1, 1], symmetric, and
    // compactly supported (zero outside |x| <= 1).
    //
    // Given
    // -----
    // - A Bartlett kernel instance.
    //
    // Expect
    // ------
    // - `w(0) = 1`, `w(0.5) = 0.5`, `w(1) = 0`, `w(1.5) = 0`.
    // - Symmetry: `w(x) = w(-x)` for representative x.
    fn kerneltype_weight_bartlett_linear_and_compact_support() {
        // Arrange
        let kernel = KernelType::Bartlett;

        // Act
        let w0 = kernel.weight(0.0);
        let w_half = kernel.weight(0.5);
        let w_one = kernel.weight(1.0);
        let w_outside = kernel.weight(1.5);
        let w_sym_pos = kernel.weight(0.3);
        let w_sym_neg = kernel.weight(-0.3);

        // Assert
        assert_eq!(w0, 1.0);
        assert_eq!(w_half, 0.5);
        assert_eq!(w_one, 0.0);
        assert_eq!(w_outside, 0.0);
        assert!((w_sym_pos - w_sym_neg).abs() < 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Check that the Parzen kernel uses the correct polynomial pieces on
    // |x| <= 0.5 and 0.5 < |x| <= 1, is symmetric, and vanishes outside
    // |x| <= 1.
    //
    // Given
    // -----
    // - A Parzen kernel instance.
    //
    // Expect
    // ------
    // - `w(0.25)` matches the first polynomial piece.
    // - `w(0.75)` matches the tapering cubic.
    // - `w(1.25) = 0`.
    // - Symmetry in x.
    fn kerneltype_weight_parzen_polynomial_pieces_and_compact_support() {
        // Arrange
        let kernel = KernelType::Parzen;

        // Act
        let x1 = 0.25_f64;
        let w_x1 = kernel.weight(x1);
        let x2 = 0.75_f64;
        let w_x2 = kernel.weight(x2);
        let w_outside = kernel.weight(1.25);
        let w_sym_pos = kernel.weight(0.4);
        let w_sym_neg = kernel.weight(-0.4);

        // First piece: 1 - 6x^2 + 6x^3 at x = 0.25.
        let expected_x1 = 0.71875_f64; // 23/32
        // Second piece: 2 (1 - x)^3 at x = 0.75.
        let expected_x2 = 0.03125_f64; // 1/32

        // Assert
        assert!((w_x1 - expected_x1).abs() < 1e-12);
        assert!((w_x2 - expected_x2).abs() < 1e-12);
        assert_eq!(w_outside, 0.0);
        assert!((w_sym_pos - w_sym_neg).abs() < 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Verify basic properties of the quadratic spectral kernel: unit mass at
    // x = 0, symmetry, and weights strictly between 0 and 1 for small
    // non-zero |x|.
    //
    // Given
    // -----
    // - A QuadraticSpectral kernel instance.
    //
    // Expect
    // ------
    // - `w(0) = 1`.
    // - `0 < w(0.1) < 1`.
    // - `w(0.1) = w(-0.1)` within numerical tolerance.
    fn kerneltype_weight_quadraticspectral_symmetric_and_finite() {
        // Arrange
        let kernel = KernelType::QuadraticSpectral;

        // Act
        let w0 = kernel.weight(0.0);
        let w_pos = kernel.weight(0.1);
        let w_neg = kernel.weight(-0.1);

        // Assert
        assert_eq!(w0, 1.0);
        assert!(w_pos > 0.0 && w_pos < 1.0);
        assert!((w_pos - w_neg).abs() < 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `calc_opt_bandwidth_param` rejects series with fewer than
    // two time points as numerically unstable, returning `DenominatorTooSmall`.
    //
    // Given
    // -----
    // - A 1×1 series matrix.
    //
    // Expect
    // ------
    // - `calc_opt_bandwidth_param` returns
    //   `Err(InferenceError::DenominatorTooSmall { denominator: 0.0 })`.
    fn calc_opt_bandwidth_param_n_less_than_two_returns_denominator_toosmall() {
        // Arrange
        let series = array![[0.0_f64]];

        // Act
        let result = calc_opt_bandwidth_param(&series, 1);

        // Assert
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InferenceError::DenominatorTooSmall { denominator: 0.0 });
    }

    #[test]
    // Purpose
    // -------
    // Verify that `calc_opt_bandwidth_param` rejects unsupported plug-in
    // orders with `OrderNotSupported` before touching ARIMA estimation.
    //
    // Given
    // -----
    // - A simple 2×1 series matrix.
    // - An unsupported order `ord = 3`.
    //
    // Expect
    // ------
    // - `calc_opt_bandwidth_param` returns
    //   `Err(InferenceError::OrderNotSupported { ord: 3 })`.
    fn calc_opt_bandwidth_param_unsupported_order_returns_ordernotsupported() {
        // Arrange
        let series = array![[1.0_f64], [0.9_f64],];

        // Act
        let result = calc_opt_bandwidth_param(&series, 3);

        // Assert
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), InferenceError::OrderNotSupported { ord: 3 });
    }

    #[test]
    // Purpose
    // -------
    // Check that `calc_opt_bandwidth_param` returns a strictly positive α(q)
    // on a small, well-behaved series when the plug-in order is supported.
    //
    // Given
    // -----
    // - A 4×1 series with mild variation.
    // - Plug-in order `ord = 1` (Bartlett).
    //
    // Expect
    // ------
    // - The result is `Ok(alpha)` with `alpha > 0.0`.
    fn calc_opt_bandwidth_param_simple_series_returns_positive_alpha() {
        // Arrange
        let series = array![[1.0_f64], [0.9_f64], [1.1_f64], [1.0_f64],];

        // Act
        let result = calc_opt_bandwidth_param(&series, 1);

        // Assert
        assert!(result.is_ok());
        let alpha = result.unwrap();
        assert!(alpha > 0.0);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `KernelType::IID::optimal_bandwidth` always returns
    // bandwidth L = 0, regardless of the series used.
    //
    // Given
    // -----
    // - A small 4×2 series matrix.
    // - `KernelType::IID`.
    //
    // Expect
    // ------
    // - `optimal_bandwidth` returns 0.
    fn kerneltype_optimal_bandwidth_iid_always_zero() {
        // Arrange
        let series = array![[1.0_f64, 0.5], [0.9_f64, 0.4], [1.1_f64, 0.6], [1.0_f64, 0.5],];
        let kernel = KernelType::IID;

        // Act
        let l = kernel.optimal_bandwidth(&series);

        // Assert
        assert_eq!(l, 0);
    }

    #[test]
    // Purpose
    // -------
    // Verify that non-IID kernels fall back to the rule-of-thumb bandwidth
    // `round(n^{1/4})` when the plug-in helper fails (e.g., n < 2).
    //
    // Given
    // -----
    // - A 1×1 series matrix (insufficient for AR(1) estimation).
    // - `KernelType::Bartlett`.
    //
    // Expect
    // ------
    // - `optimal_bandwidth` returns `round(1^{1/4}) = 1`.
    fn kerneltype_optimal_bandwidth_fallback_uses_rule_of_thumb_for_small_n() {
        // Arrange
        let series = array![[0.0_f64]];
        let kernel = KernelType::Bartlett;

        // Act
        let l = kernel.optimal_bandwidth(&series);

        // Assert
        assert_eq!(l, 1);
    }

    #[test]
    // Purpose
    // -------
    // Sanity-check that `optimal_bandwidth` for a non-IID kernel on a small,
    // well-behaved series returns a non-negative bandwidth without panicking.
    //
    // Given
    // -----
    // - A 4×1 series with mild variation.
    // - `KernelType::Parzen`.
    //
    // Expect
    // ------
    // - `optimal_bandwidth` returns an integer `L >= 0`.
    fn kerneltype_optimal_bandwidth_parzen_returns_non_negative_bandwidth() {
        // Arrange
        let series = array![[1.0_f64], [0.9_f64], [1.1_f64], [1.0_f64],];
        let kernel = KernelType::Parzen;

        // Act
        let _ = kernel.optimal_bandwidth(&series);
    }
}
