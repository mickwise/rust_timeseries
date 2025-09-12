//! HAC (Heteroskedasticity & Autocorrelation Consistent) score covariance.
//!
//! This module builds a `p×p` covariance matrix of *average* per-observation scores
//! for use in robust (sandwich) standard errors. The estimator is
//!
//! ```text
//! S  =  Γ₀  +  ∑_{k=1}^{L} w_k ( Γ_k + Γ_kᵀ ),
//! Γ₀ = (1/n) S_fullᵀ S_full,
//! Γ_k = c_k · S_{k:}ᵀ S_{:n−k},
//! ```
//!
//! where `S_full` is the `n×p` matrix of scores (rows=time, cols=parameters),
//! `w_k` are kernel weights, and `L` is the bandwidth. The scaling `c_k` is
//!
//! - Newey–West (small-sample) **on**:  `c_k = 1/(n − k)`,
//! - Newey–West **off**:                `c_k = 1/n`.
//!
//! With `L = 0` the estimator reduces to the IID OPG (`(1/n) SᵀS`). When
//! `center = true`, columns are demeaned before both the bandwidth selection and
//! the HAC aggregation so that the plug-in bandwidth “sees” the same series that
//! is ultimately aggregated.
//!
//! ### Conventions
//! - The covariance is on the *average* score scale (to match average log-likelihood).
//! - Kernel taper argument is `k/(L+1)` to avoid division by zero at `k=L`.
//! - Rows = time, columns = components/parameters.
//! - w_k are chosen to be 1 for all k.
use crate::inference::kernel::KernelType;
use ndarray::{Array2, s};
use std::{borrow::Cow, cmp::min};

/// Settings for building the score covariance used in robust (sandwich) SEs.
///
/// The model aggregates per-observation scores into a `p×p` covariance using
/// either the IID outer-product-of-gradients or a HAC estimator with the
/// chosen kernel and bandwidth. Scores are scaled so the covariance matches
/// the *average* log-likelihood convention.
///
/// Behavior:
/// - `bandwidth`: if `None`, a default rule is chosen at compute time; if `Some(k)`,
///   it is truncated by the available sample length.
/// - `center`: optionally center scores before aggregation (typically unnecessary
///   at the MLE).
/// - `small_sample_correction`: apply a finite-sample scaling to the covariance.
///
/// Defaults: `bandwidth = None`, `kernel = Bartlett`, `center = false`,
/// `small_sample_correction = true`.
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
    /// Construct a `HACOptions`.
    ///
    /// No validation is performed here; `bandwidth` (if provided) is truncated to
    /// `n−1` at compute time.
    ///
    /// # Arguments
    /// - `bandwidth`: optional user-chosen `L`.
    /// - `kernel`: taper family.
    /// - `center`: whether to demean columns prior to estimation.
    /// - `small_sample_correction`: Newey–West finite-sample scaling toggle.
    ///
    /// # Returns
    /// A `HACOptions` instance with the given settings.
    pub fn new(
        bandwidth: Option<usize>, kernel: KernelType, center: bool, small_sample_correction: bool,
    ) -> HACOptions {
        HACOptions { bandwidth, kernel, center, small_sample_correction }
    }

    /// Default HAC settings.
    ///
    /// Returns options with:
    /// - `bandwidth=None` (use plug-in selection),
    /// - `kernel=Bartlett`,
    /// - `center=false`,
    /// - `small_sample_correction=true`.
    pub fn default() -> HACOptions {
        HACOptions {
            bandwidth: None,
            kernel: KernelType::Bartlett,
            center: false,
            small_sample_correction: true,
        }
    }
}

/// Build a `p×p` HAC covariance matrix of average scores.
///
/// Steps:
/// 1. If `center=true`, demean each column of `raw_scores`.
/// 2. Choose the bandwidth `L`: if `bandwidth` is `Some`, use `min(bw, n−1)`;
///    otherwise compute a plug-in bandwidth from the **same** series matrix
///    passed to HAC (centered if applicable).
/// 3. Accumulate
///    ```text
///    S  =  Γ₀  +  ∑_{k=1}^{L} w_k ( Γ_k + Γ_kᵀ ),
///    Γ₀ = (1/n) Sᵀ S,
///    Γ_k = c_k · S_{k:}ᵀ S_{:n−k},
///    ```
///    where `c_k = 1/(n−k)` if `small_sample_correction` is `true`, else `1/n`.
///
/// # Arguments
/// - `hac_opts`: kernel, centering, small-sample, and bandwidth policy.
/// - `raw_scores`: `n×p` matrix (rows=time, cols=parameters or components).
///
/// # Returns
/// A symmetric `p×p` matrix, the HAC covariance of the *average* per-observation
/// scores, suitable for the sandwich variance `w_iᵀ S w_i`.
///
/// # Notes
/// - With `L=0` this reduces to the IID OPG (`(1/n) SᵀS`).
/// - The taper weight uses `k/(L+1)` so the maximum lag is well-defined.
pub fn calculate_avg_scores(hac_opts: &HACOptions, raw_scores: &Array2<f64>) -> Array2<f64> {
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

/// Add a single lag component to the HAC accumulator.
///
/// For `lag=0`:
/// ```text
/// avg_scores += (1/n) Sᵀ S
/// ```
///
/// For `lag>0`:
/// ```text
/// Γ_k = c_k · S_{k:}ᵀ S_{:n−k},   c_k = 1/(n−k)  (NW on),  or  1/n (NW off)
/// avg_scores += w_k · ( Γ_k + Γ_kᵀ )
/// ```
/// where `w_k = kernel.weight(k/(L+1))`.
///
/// # Arguments
/// - `avg_scores`: running `p×p` accumulator (updated in place).
/// - `scores`: `n×p` series matrix used for HAC (centered if requested).
/// - `lag`: integer `k` in `0..=L`.
/// - `bandwidth`: total `L`, used only to compute the taper argument.
/// - `hac_opts`: to obtain the kernel and the small-sample policy.
///
/// # Performance
/// Uses one cross-product per `k>0` and adds its transpose via a view to avoid
/// extra allocations. No temporary `p×p` buffers are created beyond the dot product.
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
