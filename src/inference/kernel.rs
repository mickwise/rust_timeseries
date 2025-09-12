//! Kernel taps and plug-in bandwidths for HAC estimation.
//!
//! This module provides:
//! - A `KernelType` enum with common HAC tapers (IID, Bartlett/Newey–West, Parzen, QS).
//! - Per-lag weights `w(x)` where `x = k/(L+1)`.
//! - A plug-in bandwidth selector `optimal_bandwidth` that estimates AR(1) column-wise
//!   on the provided series matrix and computes Andrews-style `α(q)` (with `q=1` for
//!   Bartlett, `q=2` for Parzen and QS). If the plug-in fails (e.g., near-unit root or
//!   tiny denominator), it falls back to the rule-of-thumb `⌊n^{1/4}⌉`.
//!
//! Conventions:
//! - Input `series_mat` is `n×p` (rows=time, cols=series/components).
//! - The plug-in uses the **same** series that HAC will aggregate (e.g., centered if
//!   centering is requested upstream).
use crate::{
    inference::errors::{InferenceError, InferenceResult},
    optimization::numerical_stability::transformations::{GENERAL_TOL, STATIONARITY_MARGIN},
};
use arima::estimate;
use ndarray::Array2;

/// HAC taper family.
///
/// - `IID`: no serial correlation; only `k=0` contributes (weight=1 at 0, else 0).
/// - `Bartlett`: triangular (Newey–West) kernel, compact support on |x|≤1.
/// - `Parzen`: smoother compact-support kernel with heavier down-weighting at high lags.
/// - `QuadraticSpectral`: infinite-support taper with high large-sample efficiency.
///
/// The taper argument is taken as `x = k/(L+1)` to avoid divide-by-zero at `k=L`.
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
    /// # Arguments
    /// - `input`: real number, typically `x = k/(L+1)` where `k` is the lag and `L` the bandwidth.
    ///
    /// # Returns
    /// Kernel value `w(x)`. For `IID`, returns 1.0 at `x=0` and 0.0 otherwise.
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

    /// Plug-in bandwidth selection for the given kernel.
    ///
    /// Strategy:
    /// - For `Bartlett`: compute `α(1)` from AR(1) fits per column and use
    ///   `L ≈ 1.1447 · (n·α)^{1/3}`.
    /// - For `Parzen`: compute `α(2)` and use `L ≈ 2.6614 · (n·α)^{1/5}`.
    /// - For `QuadraticSpectral`: compute `α(2)` and use `L ≈ 1.3221 · (n·α)^{1/5}`.
    /// - For `IID`: return `0`.
    ///
    /// If the plug-in step errors (e.g., stationarity violated or tiny denominator),
    /// falls back to `round(n^{1/4})`.
    ///
    /// # Arguments
    /// - `series_mat`: `n×p` matrix (rows=time). Use the same series you will pass
    ///   into the HAC aggregator (e.g., centered if applicable).
    ///
    /// # Returns
    /// Non-negative integer bandwidth `L`, truncated by the caller (e.g., to `n−1`).
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

/// Compute the Andrews plug-in `α(q)` by aggregating across columns.
///
/// For each column:
/// 1. Fit AR(1) via `arima::estimate::fit` to obtain `(intercept, φ)`.
/// 2. Enforce stationarity via a small safety margin; if violated, return an error.
/// 3. Compute residuals and `σ²` (sample variance with `n−1` in the denominator).
/// 4. Accumulate the numerator/denominator terms for `q=1` or `q=2`.
///
/// After iterating all columns, validate the denominator (against `GENERAL_TOL`);
/// if too small, return an error so callers can fall back. Otherwise return `α = num/den`.
///
/// # Arguments
/// - `series_mat`: `n×p` series (rows=time).
/// - `ord`: plug-in order `q` (1 for Bartlett, 2 for Parzen/QS).
///
/// # Errors
/// - `StationarityViolated { φ }` if `|φ|` is too close to 1.
/// - `OrderNotSupported { ord }` for unsupported `q`.
/// - `DenominatorTooSmall { denominator }` if the final sum is numerically tiny.
///
/// # Returns
/// A positive scalar `α(q)` used by `optimal_bandwidth`.
fn calc_opt_bandwidth_param(series_mat: &Array2<f64>, ord: usize) -> InferenceResult<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for col in series_mat.columns() {
        let col_vec = col.to_vec();
        let coeff = estimate::fit(&col_vec, 1, 0, 0)?;
        let intercept = coeff[0];
        let phi = coeff[1];
        if phi.abs() >= 1.0 - STATIONARITY_MARGIN {
            return Err(InferenceError::StationarityViolated { phi });
        }
        let phi_squared = phi * phi;
        let residuals = estimate::residuals(&col_vec, intercept, Some(&[phi]), None)?;
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
