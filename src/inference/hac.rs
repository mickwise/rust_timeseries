//! inference::hac — HAC score covariance for robust standard errors.
//!
//! Purpose
//! -------
//! Build HAC (Heteroskedasticity & Autocorrelation Consistent) covariance
//! matrices of *average* per-observation scores for use in robust (sandwich)
//! standard errors. The estimator has the form
//!
//! ```text
//! S  =  Γ₀  +  ∑_{k=1}^{L} w_k ( Γ_k + Γ_kᵀ ),
//! Γ₀ = (1/n) S_fullᵀ S_full,
//! Γ_k = c_k · S_{k:}ᵀ S_{:n−k},
//! ```
//!
//! where `S_full` is the `n×p` score matrix (rows=time, cols=parameters),
//! `w_k` are kernel weights, and `L` is the bandwidth. The scaling `c_k` is
//!
//! - Newey–West (small-sample) **on**:  `c_k = 1/(n − k)`,
//! - Newey–West **off**:                `c_k = 1/n`.
//!
//! Key behaviors
//! -------------
//! - Aggregate per-observation scores into a symmetric `p×p` covariance
//!   matrix on the *average-score* scale.
//! - Support both IID OPG (`L = 0`) and HAC estimators with a configurable
//!   kernel, bandwidth, centering, and small-sample correction.
//! - Provide [`HACOptions`] as a compact configuration object for callers,
//!   including a plug-in bandwidth regime via [`KernelType::optimal_bandwidth`].
//!
//! Invariants & assumptions
//! ------------------------
//! - Input `raw_scores` must be an `n×p` matrix with `n ≥ 1`; zero-row
//!   inputs are not supported and will lead to panics via internal
//!   assumptions (e.g., `mean_axis(...).unwrap()`, `n - 1`).
//! - The same series matrix that HAC aggregates (centered if `center=true`)
//!   must be passed to the plug-in bandwidth selector to keep `L`
//!   consistent with the final covariance calculation.
//! - Bandwidth is always truncated to `L ≤ n−1`, so no lag exceeds the
//!   available sample length.
//! - Small-sample scaling uses the Newey–West convention: `c_k = 1/(n−k)`
//!   if enabled, `c_k = 1/n` otherwise.
//!
//! Conventions
//! -----------
//! - Rows index time (`t = 1,…,n`); columns index parameters or score
//!   components (`i = 1,…,p`).
//! - The covariance is computed on the *average* score scale to match the
//!   average log-likelihood convention used in the Hessian and information
//!   matrix.
//! - The taper argument is `x = k/(L+1)` so that `x ∈ [0,1)` even at the
//!   maximum lag `k = L`, avoiding division by zero.
//! - Kernel weights `w_k` are obtained from [`KernelType::weight`] and are
//!   not assumed to be constant in `k`.
//!
//! Downstream usage
//! ----------------
//! - [`HACOptions`] is embedded in higher-level inference controls (e.g.,
//!   model-side [`InferenceOptions`]) to select robust vs classical
//!   standard errors.
//! - The main entry point [`calculate_avg_scores_cov`] is used to construct
//!   the score covariance matrix `S` that feeds the robust (sandwich)
//!   variance calculation in `inference::hessian`.
//! - Library users typically do not call the low-level helper
//!   [`add_hac_component`] directly; it is considered an internal detail.
//!
//! Testing notes
//! -------------
//! - Unit tests cover:
//!   - Symmetry and positive semi-definiteness (up to numerical tolerance)
//!     of the returned covariance matrix.
//!   - Equality of centered vs non-centered results when the columns of
//!     `raw_scores` are exactly mean-zero.
//!   - Behavior under `KernelType::IID` (`L = 0`) vs nontrivial kernels
//!     (Bartlett/Parzen/QS).
//!   - Effects of `small_sample_correction` on finite-sample scaling.
//! - Integration tests in the Hessian-based inference layer verify
//!   that robust standard errors change in the expected direction when
//!   serial correlation or heteroskedasticity is injected into the scores.
use crate::inference::kernel::KernelType;
use ndarray::{Array2, s};
use std::{borrow::Cow, cmp::min};

/// HACOptions — configuration for HAC score covariance estimation.
///
/// Purpose
/// -------
/// Represent the kernel, bandwidth, centering, and small-sample policy for
/// constructing HAC score covariance matrices from per-observation scores.
/// This type is threaded through model-side inference options so that
/// robust standard errors can be configured in one place.
///
/// Key behaviors
/// -------------
/// - Encodes the HAC kernel family (`IID`, `Bartlett`, `Parzen`,
///   `QuadraticSpectral`) via [`KernelType`].
/// - Controls the bandwidth regime: fixed (`Some(L)`) or plug-in
///   (`None`, determined at compute time).
/// - Toggles score centering and Newey–West small-sample corrections.
///
/// Parameters
/// ----------
/// - `bandwidth`: `Option<usize>`
///   Optional bandwidth `L`. If `None`, a plug-in bandwidth is selected
///   using [`KernelType::optimal_bandwidth`]. If `Some(L)`, the effective
///   bandwidth is truncated to `min(L, n−1)`.
/// - `kernel`: [`KernelType`]  
///   HAC taper family controlling `w_k`.
/// - `center`: `bool`  
///   If `true`, columns of the score matrix are demeaned before both
///   bandwidth selection and HAC aggregation.
/// - `small_sample_correction`: `bool`  
///   If `true`, use `c_k = 1/(n−k)`; otherwise use `c_k = 1/n`.
///
/// Fields
/// ------
/// - `kernel`: [`KernelType`]  
///   Kernel family for the HAC estimator.
/// - `bandwidth`: `Option<usize>`  
///   Optional user-supplied bandwidth; truncated by the available sample
///   size at compute time.
/// - `center`: `bool`  
///   Whether to center per-observation scores before aggregation.
/// - `small_sample_correction`: `bool`  
///   Whether to apply Newey–West finite-sample scaling.
///
/// Invariants
/// ----------
/// - If `bandwidth` is `Some(L)`, the effective bandwidth used in
///   [`calculate_avg_scores_cov`] satisfies `0 ≤ L_eff ≤ n−1`.
/// - The semantics of `small_sample_correction` must match the scaling
///   used in [`add_hac_component`].
///
/// Performance
/// -----------
/// - This type is a small, cheap-to-copy configuration struct intended to
///   be passed by reference where possible.
///
/// Notes
/// -----
/// - A `Default` implementation is available for common HAC settings:
///   Bartlett kernel, plug-in bandwidth, no centering, Newey–West
///   small-sample correction.
/// - The same `HACOptions` instance may be reused across multiple models
///   or estimation runs as long as the score matrices share the same
///   interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct HACOptions {
    /// Kernel type for the HAC estimator (if robust_se is true).
    pub kernel: KernelType,
    /// Bandwidth for the HAC estimator (if robust_se is true).
    pub bandwidth: Option<usize>,
    /// Center per-observation scores before aggregation.
    pub center: bool,
    /// Apply finite-sample scaling to the HAC/OPG estimate.
    pub small_sample_correction: bool,
}

impl HACOptions {
    /// Construct a `HACOptions` value from explicit settings.
    ///
    /// Parameters
    /// ----------
    /// - `bandwidth`: `Option<usize>`
    ///   User-chosen bandwidth `L`. If `None`, a plug-in bandwidth is
    ///   selected later using [`KernelType::optimal_bandwidth`]. If
    ///   `Some(L)`, it will be truncated to `n−1` at compute time.
    /// - `kernel`: [`KernelType`]
    ///   Kernel family used to compute per-lag weights `w_k`.
    /// - `center`: `bool`
    ///   Whether to demean score columns before bandwidth selection and
    ///   HAC aggregation.
    /// - `small_sample_correction`: `bool`
    ///   Whether to apply Newey–West finite-sample scaling
    ///   (`c_k = 1/(n−k)` vs `c_k = 1/n`).
    ///
    /// Returns
    /// -------
    /// `HACOptions`
    ///   A configuration value encapsulating the requested HAC policy.
    ///
    /// Errors
    /// ------
    /// - `None`
    ///   This constructor does not perform validation and never returns
    ///   an error. Invalid combinations are assumed to be guarded by
    ///   callers or by downstream routines.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - No safety invariants beyond the usual Rust borrowing rules.
    ///
    /// Notes
    /// -----
    /// - This function does not clamp the supplied `bandwidth`; the
    ///   truncation to `n−1` is applied in [`calculate_avg_scores_cov`].
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::inference::kernel::KernelType;
    /// # use rust_timeseries::inference::hac::HACOptions;
    /// let opts = HACOptions::new(
    ///     None,
    ///     KernelType::Bartlett,
    ///     false,
    ///     true,
    /// );
    /// assert_eq!(opts.center, false);
    /// ```
    pub fn new(
        bandwidth: Option<usize>, kernel: KernelType, center: bool, small_sample_correction: bool,
    ) -> HACOptions {
        HACOptions { bandwidth, kernel, center, small_sample_correction }
    }
}

impl Default for HACOptions {
    /// Default HAC settings for robust standard errors.
    ///
    /// Parameters
    /// ----------
    /// - None
    ///   This function takes no parameters; all options are chosen
    ///   according to the documented defaults.
    ///
    /// Returns
    /// -------
    /// `HACOptions`
    ///   A configuration with:
    ///   - `bandwidth = None` (plug-in selection),
    ///   - `kernel = KernelType::Bartlett`,
    ///   - `center = false`,
    ///   - `small_sample_correction = true`.
    ///
    /// Errors
    /// ------
    /// - `None`
    ///   This constructor never fails.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - No additional safety requirements.
    ///
    /// Notes
    /// -----
    /// - The defaults reflect a common econometric baseline: Bartlett
    ///   kernel with plug-in bandwidth and Newey–West small-sample
    ///   correction, without re-centering scores at the MLE.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::inference::hac::HACOptions;
    /// let opts = HACOptions::default();
    /// assert!(opts.bandwidth.is_none());
    /// assert!(opts.small_sample_correction);
    /// ```
    fn default() -> Self {
        Self {
            bandwidth: None,
            kernel: KernelType::Bartlett,
            center: false,
            small_sample_correction: true,
        }
    }
}

/// Build a `p×p` HAC covariance matrix of average per-observation scores.
///
/// Parameters
/// ----------
/// - `hac_opts`: `&HACOptions`
///   HAC configuration controlling kernel, bandwidth selection, centering,
///   and Newey–West small-sample scaling. If `bandwidth` is `None`, a
///   plug-in bandwidth is computed from the (possibly centered) scores.
/// - `raw_scores`: `&Array2<f64>`
///   `n×p` matrix of per-observation scores (rows=time, columns=parameters).
///   Must satisfy `n ≥ 1`.
///
/// Returns
/// -------
/// `Array2<f64>`
///   A symmetric `p×p` covariance matrix on the *average-score* scale,
///   suitable for use in robust (sandwich) variance formulas
///   `Var(θ̂_i) = w_iᵀ S w_i`.
///
/// Errors
/// ------
/// - `None`
///   This function does not return a `Result`; domain violations (e.g.,
///   zero-row inputs) are considered errors and may lead to
///   panics in downstream operations.
///
/// Panics
/// ------
/// - If `raw_scores.nrows() == 0`, due to internal assumptions such as
///   `mean_axis(...).unwrap()` and `n - 1` in bandwidth truncation.
/// - If `raw_scores` has inconsistent dimensions (e.g., extremely large
///   sizes that cause allocation failures).
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that `raw_scores`
///   matches the interpretation expected by the HAC kernel and that the
///   chosen options are appropriate for the model.
///
/// Notes
/// -----
/// - With an effective bandwidth `L = 0`, this reduces to the IID
///   outer-product-of-gradients estimator `(1/n) SᵀS`.
/// - When `hac_opts.center = true`, centering is applied once via a
///   single allocation using `Cow`, and both bandwidth selection and HAC
///   aggregation see the same centered series.
/// - The taper weight at lag `k` is `w_k = kernel.weight(k/(L+1))`.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::array;
/// # use rust_timeseries::inference::hac::{HACOptions, calculate_avg_scores_cov};
/// # use rust_timeseries::inference::kernel::KernelType;
/// let scores = array![[1.0, -1.0],
///                     [0.5, -0.5],
///                     [1.5, -1.5]];
/// let opts = HACOptions::new(None, KernelType::Bartlett, false, true);
/// let s = calculate_avg_scores_cov(&opts, &scores);
/// assert_eq!(s.shape(), &[2, 2]);
/// ```
pub fn calculate_avg_scores_cov(hac_opts: &HACOptions, raw_scores: &Array2<f64>) -> Array2<f64> {
    let n = raw_scores.nrows();
    let p = raw_scores.ncols();

    // Center scores if requested
    let scores: Cow<'_, Array2<f64>> = if hac_opts.center {
        let col_means = raw_scores.mean_axis(ndarray::Axis(0)).unwrap();
        Cow::Owned(raw_scores - &col_means) // one copy only when centering
    } else {
        Cow::Borrowed(raw_scores)
    };

    // Determine bandwidth
    let bandwidth = match hac_opts.bandwidth {
        Some(bw) => min(bw, n - 1),
        None => min(hac_opts.kernel.optimal_bandwidth(scores.as_ref()), n - 1),
    };

    // Compute HAC estimate
    let mut avg_scores = Array2::<f64>::zeros((p, p));
    for lag in 0..=bandwidth {
        add_hac_component(&mut avg_scores, scores.as_ref(), lag, bandwidth, hac_opts);
    }
    avg_scores
}

// ---- Helper methods ----

/// Add a single lag component to the HAC score covariance accumulator.
///
/// Parameters
/// ----------
/// - `avg_scores`: `&mut Array2<f64>`
///   Running `p×p` covariance accumulator `S`. Updated in place by
///   adding the contribution from the given lag.
/// - `scores`: `&Array2<f64>`
///   `n×p` score matrix used for HAC aggregation (centered if requested
///   upstream). Must have at least as many rows as `lag + 1`.
/// - `lag`: `usize`
///   Current lag index `k` in `0..=L`. For `lag = 0`, adds the IID OPG
///   term `(1/n) SᵀS`. For `lag > 0`, adds the symmetrized cross-lag
///   contribution.
/// - `bandwidth`: `usize`
///   Total bandwidth `L` used to compute the taper argument
///   `x = k/(L+1)`. Assumed to satisfy `L ≥ lag`.
/// - `hac_opts`: `&HACOptions`
///   HAC configuration providing the kernel and small-sample policy.
///
/// Returns
/// -------
/// `()`
///   This function mutates `avg_scores` in place and does not return a
///   separate value.
///
/// Errors
/// ------
/// - `None`
///   This helper does not return errors. It assumes that callers have
///   already enforced the dimensional constraints on `scores`,
///   `lag ≤ bandwidth`, and `bandwidth ≤ n−1`.
///
/// Panics
/// ------
/// - If `scores` has fewer than `lag + 1` rows, due to slicing
///   `scores.slice(s![lag.., ..])` and `scores.slice(s![..n - lag, ..])`.
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that the arguments
///   satisfy the documented shape constraints.
///
/// Notes
/// -----
/// - For `lag = 0`, this function computes `(1/n) SᵀS` and adds it to
///   `avg_scores`.
/// - For `lag > 0`, it computes
///   `Γ_k = c_k · S_{k:}ᵀ S_{:n−k}` with `c_k` determined by
///   `small_sample_correction`, then adds
///   `w_k (Γ_k + Γ_kᵀ)` where `w_k = kernel.weight(k/(L+1))`.
/// - No additional `p×p` temporaries are allocated beyond the dot
///   products; transposes are taken as views to avoid extra copies.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::{array, Array2};
/// # use rust_timeseries::inference::hac::{HACOptions, add_hac_component};
/// # use rust_timeseries::inference::kernel::KernelType;
/// let scores = array![[1.0, 2.0],
///                     [3.0, 4.0]];
/// let mut acc = Array2::<f64>::zeros((2, 2));
/// let opts = HACOptions::new(Some(0), KernelType::IID, false, true);
/// add_hac_component(&mut acc, &scores, 0, 0, &opts);
/// assert!(acc[[0, 0]] > 0.0);
/// ```
fn add_hac_component(
    avg_scores: &mut Array2<f64>, scores: &Array2<f64>, lag: usize, bandwidth: usize,
    hac_opts: &HACOptions,
) -> () {
    let n = scores.nrows();
    let weight = hac_opts.kernel.weight(lag as f64 / (bandwidth + 1) as f64);
    match lag {
        0 => {
            let scores_t = scores.t();
            avg_scores.scaled_add(1.0 / (n as f64), &scores_t.dot(scores));
        }
        _ => {
            let small_samp_correction = if hac_opts.small_sample_correction {
                1.0 / ((n - lag) as f64)
            } else {
                1.0 / (n as f64)
            };
            let scores_lagged = scores.slice(s![lag.., ..]);
            let scores_leading = scores.slice(s![..n - lag, ..]);
            let scores_lagged_t = scores_lagged.t();
            let corrected_gamma_k = small_samp_correction * scores_lagged_t.dot(&scores_leading);
            avg_scores.scaled_add(weight, &corrected_gamma_k);
            avg_scores.scaled_add(weight, &corrected_gamma_k.t());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array2, array};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Default and explicit configuration behavior of `HACOptions`.
    // - Basic invariants of `calculate_avg_scores_cov`:
    //   - symmetry of the returned covariance,
    //   - compatibility with the IID OPG when `L = 0`,
    //   - invariance to centering when scores are exactly mean-zero,
    //   - inflation effect from Newey–West small-sample correction.
    // - Consistency with the textbook HAC formula for a small
    //   hand-computable example (Bartlett kernel, truncated bandwidth).
    //
    // They intentionally DO NOT cover:
    // - The AR(1) plug-in bandwidth estimator in `KernelType::optimal_bandwidth`;
    //   that logic is tested separately in the kernel module.
    // - End-to-end Hessian-based inference; those invariants are verified
    //   in `inference::hessian` and model-level integration tests.
    // -------------------------------------------------------------------------

    const TOL: f64 = 1e-10;

    fn assert_matrices_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(a[[i, j]], b[[i, j]], epsilon = tol, max_relative = tol);
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `HACOptions::default` matches the documented baseline
    // configuration (Bartlett, plug-in bandwidth, no centering, NW on).
    //
    // Given
    // -----
    // - No inputs; call `HACOptions::default()`.
    //
    // Expect
    // ------
    // - `bandwidth.is_none()`, `kernel=Bartlett`, `center=false`,
    //   `small_sample_correction=true`.
    fn hacoptions_default_matches_documented_defaults() {
        // Arrange
        let opts = HACOptions::default();

        // Act / Assert
        assert!(opts.bandwidth.is_none());
        assert_eq!(opts.kernel, KernelType::Bartlett);
        assert!(!opts.center);
        assert!(opts.small_sample_correction);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `HACOptions::new` preserves the provided configuration
    // values without implicit normalization.
    //
    // Given
    // -----
    // - A specific combination of bandwidth, kernel, centering, and NW flag.
    //
    // Expect
    // ------
    // - The resulting struct stores exactly these values.
    fn hacoptions_new_preserves_fields() {
        // Arrange
        let opts = HACOptions::new(Some(5), KernelType::Parzen, true, false);

        // Act / Assert
        assert_eq!(opts.bandwidth, Some(5));
        assert_eq!(opts.kernel, KernelType::Parzen);
        assert!(opts.center);
        assert!(!opts.small_sample_correction);
    }

    #[test]
    // Purpose
    // -------
    // Check that with `KernelType::IID` and `L = 0`, the HAC covariance
    // reduces to the IID OPG `(1/n) Sᵀ S`.
    //
    // Given
    // -----
    // - A small `n×p` score matrix.
    // - `HACOptions` with `kernel=IID`, `bandwidth=Some(0)`, no centering,
    //   and arbitrary NW flag (irrelevant when `L=0`).
    //
    // Expect
    // ------
    // - `calculate_avg_scores_cov` matches `(1/n) Sᵀ S` up to numerical
    //   tolerance.
    fn calculate_avg_scores_cov_iid_l0_matches_opg() {
        // Arrange
        let scores = array![[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]];
        let n = scores.nrows() as f64;

        let opts = HACOptions::new(Some(0), KernelType::IID, false, true);

        // Act
        let hac_cov = calculate_avg_scores_cov(&opts, &scores);

        let st = scores.t();
        let opg: Array2<f64> = st.dot(&scores) * (1.0 / n);

        // Assert
        assert_matrices_close(&hac_cov, &opg, TOL);
    }

    #[test]
    // Purpose
    // -------
    // Verify that centering has no effect when score columns are already
    // exactly mean-zero.
    //
    // Given
    // -----
    // - An `n×p` score matrix whose columns sum to zero.
    // - Two `HACOptions` values that differ only in `center` (true vs false).
    //
    // Expect
    // ------
    // - The resulting covariance matrices are equal up to numerical
    //   tolerance.
    fn calculate_avg_scores_cov_center_invariant_for_mean_zero() {
        // Arrange: columns are mean-zero by construction.
        let scores = array![[1.0, -1.0], [-1.0, 1.0], [2.0, -2.0], [-2.0, 2.0]];

        let opts_uncentered = HACOptions::new(None, KernelType::Bartlett, false, true);
        let opts_centered = HACOptions::new(None, KernelType::Bartlett, true, true);

        // Act
        let cov_uncentered = calculate_avg_scores_cov(&opts_uncentered, &scores);
        let cov_centered = calculate_avg_scores_cov(&opts_centered, &scores);

        // Assert
        assert_matrices_close(&cov_uncentered, &cov_centered, TOL);
    }

    #[test]
    // Purpose
    // -------
    // Confirm that Newey–West small-sample correction does not reduce the
    // variance relative to the uncorrected HAC estimator in a simple
    // one-parameter setting.
    //
    // Given
    // -----
    // - A univariate score series with trend (positive serial structure).
    // - Identical settings except `small_sample_correction = true/false`.
    //
    // Expect
    // ------
    // - The [0,0] entry of the covariance with NW correction is at least
    //   as large as without correction, up to numerical tolerance.
    fn calculate_avg_scores_cov_small_sample_correction_inflates_variance() {
        // Arrange
        let scores = array![[1.0], [2.0], [3.0], [4.0]];

        let opts_nw_on = HACOptions::new(Some(3), KernelType::Bartlett, false, true);
        let opts_nw_off = HACOptions::new(Some(3), KernelType::Bartlett, false, false);

        // Act
        let cov_nw_on = calculate_avg_scores_cov(&opts_nw_on, &scores);
        let cov_nw_off = calculate_avg_scores_cov(&opts_nw_off, &scores);

        // Assert
        assert!(
            cov_nw_on[[0, 0]] + TOL >= cov_nw_off[[0, 0]],
            "NW correction should not reduce variance: on={} off={}",
            cov_nw_on[[0, 0]],
            cov_nw_off[[0, 0]]
        );
    }

    #[test]
    // Purpose
    // -------
    // Ensure that the HAC covariance matrix is symmetric up to numerical
    // tolerance for generic inputs.
    //
    // Given
    // -----
    // - A generic `n×p` score matrix with no special structure.
    //
    // Expect
    // ------
    // - `S[i,j] ≈ S[j,i]` for all i,j.
    fn calculate_avg_scores_cov_returns_symmetric_matrix() {
        // Arrange
        let scores =
            array![[0.5, -1.0, 2.0], [1.0, 0.0, -0.5], [-0.5, 1.5, 0.25], [2.0, -0.5, 1.0]];

        let opts = HACOptions::new(None, KernelType::Parzen, true, true);

        // Act
        let cov = calculate_avg_scores_cov(&opts, &scores);

        // Assert
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert_relative_eq!(cov[[i, j]], cov[[j, i]], epsilon = TOL, max_relative = TOL);
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // Validate that `calculate_avg_scores_cov` matches a direct manual
    // implementation of the HAC formula for a small univariate example
    // with Bartlett kernel, finite bandwidth, and Newey–West scaling.
    //
    // Given
    // -----
    // - A 4×1 score matrix.
    // - `kernel=Bartlett`, `bandwidth=Some(10)`, NW correction enabled.
    //
    // Expect
    // ------
    // - The returned covariance equals the hand-computed HAC estimator
    //   using `L_eff = min(10, n−1) = 3` up to numerical tolerance.
    fn calculate_avg_scores_cov_matches_manual_bartlett_newey_west() {
        // Arrange: univariate scores
        let scores = array![[1.0], [0.5], [-0.25], [2.0]];
        let n = scores.nrows();
        let l_requested: usize = 10;
        let l_eff: usize = std::cmp::min(l_requested, n - 1);

        let opts = HACOptions::new(Some(l_requested), KernelType::Bartlett, false, true);

        // Act: library implementation
        let cov_lib = calculate_avg_scores_cov(&opts, &scores);

        // Manual HAC implementation (1D)
        let mut cov_manual = Array2::<f64>::zeros((1, 1));
        let st = scores.t();

        // Γ₀ = (1/n) Sᵀ S
        cov_manual = cov_manual + st.dot(&scores) * (1.0 / (n as f64));

        // Γ_k terms with Newey–West scaling and Bartlett weights
        for k in 1..=l_eff {
            let ck = 1.0 / ((n - k) as f64); // NW on
            let scores_lagged = scores.slice(s![k.., ..]);
            let scores_leading = scores.slice(s![..n - k, ..]);
            let scores_lagged_t = scores_lagged.t();
            let gamma_k = scores_lagged_t.dot(&scores_leading) * ck;

            let weight = KernelType::Bartlett.weight(k as f64 / (l_eff + 1) as f64);

            cov_manual = cov_manual + gamma_k.clone() * weight;
            cov_manual = cov_manual + gamma_k.t().to_owned() * weight;
        }

        // Assert
        assert_matrices_close(&cov_lib, &cov_manual, 1e-9);
    }
}
