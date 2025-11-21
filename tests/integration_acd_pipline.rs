//! Integration tests for ACD duration models and inference.
//!
//! Purpose
//! -------
//! - Validate the end-to-end ACD pipeline: from validated duration data,
//!   through model construction and MLE fitting, to classical and HAC
//!   standard errors and forecasting.
//! - Exercise realistic parameter regimes (shapes, scales, innovations,
//!   and optimizer settings) rather than toy edge cases only.
//!
//! Coverage
//! --------
//! - `duration::core`:
//!   - `ACDData` construction with and without a `t0` offset.
//!   - `ACDShape` validation for admissible vs invalid (p, q, n).
//! - `duration::models::acd::ACDModel`:
//!   - Model construction, fitting, standard errors, and forecasting.
//! - `inference::hac` and `inference::kernel`:
//!   - Classical vs HAC standard-error paths, including non-default
//!     kernels and bandwidth settings.
//! - `optimization::loglik_optimizer`:
//!   - Use of LBFGS + line search via `MLEOptions` and `Tolerances`.
//!
//! Exclusions
//! ----------
//! - Fine-grained validation of low-level building blocks
//!   (guards, validation routines, numerical
//!   stability helpers) — these are covered by unit tests.
//! - Python bindings, serialization, or user-facing API wrappers — those
//!   are expected to be tested at a higher integration or system level.
//! - Exhaustive stress testing over extreme sample sizes and parameter
//!   grids — those belong in targeted performance and property tests.
use ndarray::{Array1, array};
use rust_timeseries::{
    duration::{
        core::{
            data::{ACDData, ACDMeta},
            guards::PsiGuards,
            init::Init,
            innovations::ACDInnovation,
            options::ACDOptions,
            shape::ACDShape,
            units::ACDUnit,
        },
        models::acd::ACDModel,
    },
    inference::{hac::HACOptions, kernel::KernelType},
    optimization::loglik_optimizer::{MLEOptions, Tolerances, traits::LineSearcher},
};

/// Purpose
/// -------
/// Construct a strictly positive, trending `ACDData` series that
/// introduces mild serial structure for HAC vs classical SE comparisons.
///
/// Parameters
/// ----------
/// - `n`: Length of the series; must be `> 0`.
/// - `base`: Baseline level for the first observation; should be
///   strictly positive.
/// - `slope`: Per-step increment; can be positive or negative, but the
///   resulting series is clamped to remain strictly positive.
///
/// Returns
/// -------
/// - An `ACDData` instance with:
///   - `x_t = max(base + slope · t, base)` for `t = 0,…,n−1`,
///   - units set to `ACDUnit::Seconds`,
///   - no `t0` offset and no log transformation.
///
/// Invariants
/// ----------
/// - Ensures all durations are finite and strictly positive before
///   calling `ACDData::new`, so creation should succeed for reasonable
///   `base` and `slope`.
///
/// Usage
/// -----
/// - Used by integration tests that need a simple but non-degenerate
///   series to:
///   - highlight differences between classical and HAC SEs, and
///   - exercise the optimizer on mildly trending data.
fn make_trending_data(n: usize, base: f64, slope: f64) -> ACDData {
    let data = Array1::from_iter((0..n).map(|t| {
        let x = base + slope * (t as f64);
        if x <= 0.0 { base } else { x }
    }));
    let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
    ACDData::new(data, None, meta).expect("ACDData::new should succeed for positive, finite series")
}

/// Purpose
/// -------
/// Provide a stable, documented baseline `ACDOptions` configuration for
/// integration tests that should reflect "typical" user settings.
///
/// Configuration
/// -------------
/// - Initialization:
///   - `Init::uncond_mean()` — start from the unconditional mean.
/// - Optimizer tolerances (`Tolerances`):
///   - `tol_cost = Some(1e-6)`
///   - `tol_grad = None`
///   - `max_iter = Some(200)`
/// - Optimizer (`MLEOptions`):
///   - Line search: `LineSearcher::MoreThuente`
///   - Default L-BFGS memory (no explicit override).
/// - Psi guards (`PsiGuards`):
///   - Bounds `(1e-6, 1e6)` to keep conditional means strictly positive
///     and numerically well-behaved.
///
/// Returns
/// -------
/// - An `ACDOptions` instance suitable for most integration tests, with
///   tolerances and guards chosen to balance robustness and runtime.
///
/// Invariants
/// ----------
/// - Panics if any of the underlying constructors reject the supplied
///   parameters; this is treated as a test-time configuration error,
///   not a runtime error path to be exercised.
fn default_acd_options() -> ACDOptions {
    let init = Init::uncond_mean();
    let tols = Tolerances::new(Some(1e-6), None, Some(200))
        .expect("Tolerances::new should accept positive tolerances");
    let mle_opts = MLEOptions::new(tols, LineSearcher::MoreThuente, None)
        .expect("MLEOptions::new should succeed with reasonable tolerances");
    let psi_guards = PsiGuards::new((1e-6, 1e6))
        .expect("PsiGuards::new should accept positive, finite bounds (min < max)");
    ACDOptions::new(init, mle_opts, psi_guards)
}

/// Purpose
/// -------
/// Provide an alternate, more aggressive `ACDOptions` configuration to
/// exercise additional optimizer and guard code paths in integration
/// tests.
///
/// Configuration
/// -------------
/// - Initialization:
///   - `Init::uncond_mean()`.
/// - Optimizer tolerances (`Tolerances`):
///   - `tol_cost = Some(1e-8)`
///   - `tol_grad = Some(1e-6)`
///   - `max_iter = Some(50)`
/// - Optimizer (`MLEOptions`):
///   - Line search: `LineSearcher::MoreThuente`
///   - Explicit L-BFGS memory: `Some(5)`.
/// - Psi guards (`PsiGuards`):
///   - Narrower bounds `(1e-4, 1e3)` to test clamping and guard behavior.
///
/// Returns
/// -------
/// - An `ACDOptions` instance that stresses the optimizer and guard
///   logic more than the default configuration.
///
/// Invariants
/// ----------
/// - As with `default_acd_options`, any failure in constructing
///   tolerances, optimizer, or guards is treated as a test configuration
///   error rather than a behavior under test.
///
/// Usage
/// -----
/// - Used by integration tests that verify the ACD API behaves sensibly
///   under tighter tolerances, explicit memory settings, and narrower
///   psi bounds (e.g., robust SEs remain finite and forecasts positive).
fn tuned_acd_options() -> ACDOptions {
    let init = Init::uncond_mean();
    let tols = Tolerances::new(Some(1e-8), Some(1e-6), Some(50))
        .expect("Tolerances::new should accept tighter tolerances");
    let mle_opts = MLEOptions::new(tols, LineSearcher::MoreThuente, Some(5))
        .expect("MLEOptions::new should succeed with explicit L-BFGS memory");
    let psi_guards =
        PsiGuards::new((1e-4, 1e3)).expect("PsiGuards::new should accept narrow yet valid bounds");
    ACDOptions::new(init, mle_opts, psi_guards)
}

/// Purpose
/// -------
/// Provide a reusable helper that wires together trending data, shape
/// validation, model construction, and MLE fitting into a single step
/// for integration tests.
///
/// Parameters
/// ----------
/// - `p`: Number of conditional expected duration lags; must satisfy `0 ≤ p < n`.
/// - `q`: Number of duration lags; must satisfy `0 ≤ q < n`.
/// - `n`: Sample size; must satisfy `n ≥ 1` and `p + q > 0`.
/// - `base`: Baseline duration level for `make_trending_data`.
/// - `slope`: Linear trend component for `make_trending_data`.
/// - `innovation`: `ACDInnovation` family (e.g., Exponential, Weibull).
/// - `opts`: Reference to an `ACDOptions` configuration used for fitting.
///
/// Returns
/// -------
/// - `(model, data, theta_dim)` where:
///   - `model`: Fitted `ACDModel` instance.
///   - `data`: The `ACDData` used for estimation (trending series).
///   - `theta_dim`: Implied parameter dimension `1 + p + q`.
///
/// Invariants
/// ----------
/// - Panics if:
///   - `ACDShape::new(p, q, n)` rejects the supplied orders, or
///   - `ACDModel::fit` fails on the synthetic data.
/// - Uses a zero vector of length `1 + p + q` as the initial parameter
///   guess, assuming that the optimizer will move to a valid region.
///
/// Usage
/// -----
/// - Used by multiple integration tests to avoid duplicating boilerplate
///   around `ACDData` creation, shape validation, model construction,
///   and fitting.
fn fit_acd_model(
    p: usize, q: usize, n: usize, base: f64, slope: f64, innovation: ACDInnovation,
    opts: &ACDOptions,
) -> (ACDModel, ACDData, usize) {
    let data = make_trending_data(n, base, slope);
    let shape = ACDShape::new(p, q, n).expect("ACDShape::new should accept p, q < n and p + q > 0");
    let mut model = ACDModel::new(shape.clone(), innovation, opts.clone(), n);
    let theta_dim = 1 + p + q;
    let theta0 = Array1::from_elem(theta_dim, 0.0);
    model.fit(theta0, &data).expect("ACDModel::fit should succeed on synthetic trending data");
    (model, data, theta_dim)
}

#[test]
// Purpose
// -------
// Ensure the ACD public API supports fitting, classical and HAC standard
// error computation, and forecasting across multiple shapes, scales,
// and innovation families without panicking and with sane outputs.
//
// Given
// -----
// - Synthetic trending duration series of length `n = 128` at several
//   base levels.
// - A grid of shapes: (p, q) ∈ {(1,0), (0,1), (1,1), (2,1)}.
// - Two innovation families: Exponential and Weibull with valid shape.
// - Baseline `ACDOptions` from `default_acd_options()`.
// - Default `HACOptions` for robust SEs.
//
// Expect
// ------
// - `fit_acd_model` succeeds for every (shape, base, innovation) combo.
// - Classical SEs:
//   - Have length `1 + p + q`.
//   - Contain only finite, non-negative values.
// - HAC SEs:
//   - Have length `1 + p + q`.
//   - Contain only finite values (non-negativity is empirically expected
//     but not asserted here).
// - Forecasts for horizon `h = 5` are finite and strictly positive for
//   all configurations.
fn acd_api_supports_multiple_shapes_scales_and_innovations() {
    let shapes: &[(usize, usize)] = &[(1, 0), (0, 1), (1, 1), (2, 1)];
    let bases: &[f64] = &[0.5, 1.0, 5.0];
    let slope_factor: f64 = 0.01;
    let innovations: &[ACDInnovation] =
        &[ACDInnovation::exponential(), ACDInnovation::weibull(1.5).expect("valid Weibull shape")];
    let n = 128;
    let opts = default_acd_options();
    let hac_opts = HACOptions::default(); // Bartlett kernel, plug-in bandwidth, no centering, NW correction
    for &(p, q) in shapes {
        for &base in bases {
            let slope = slope_factor * base;
            for &innovation in innovations {
                let (mut model, data, theta_dim) =
                    fit_acd_model(p, q, n, base, slope, innovation, &opts);
                // classical SEs must be non-negative and finite
                let se_classical = model
                    .standard_errors(&data, None)
                    .expect("classical SEs should succeed after fit");
                assert_eq!(se_classical.len(), theta_dim);
                assert!(se_classical.iter().all(|v| v.is_finite() && *v >= 0.0));
                // HAC SEs: check length and finiteness only
                let se_hac = model
                    .standard_errors(&data, Some(&hac_opts))
                    .expect("HAC SEs should succeed after fit");
                assert_eq!(se_hac.len(), theta_dim);
                assert!(se_hac.iter().all(|v| v.is_finite()));
                // Forecast should be positive and finite
                let h_forecast =
                    model.forecast(5, &data).expect("forecast should succeed after fit");
                assert!(h_forecast.is_finite() && h_forecast > 0.0);
            }
        }
    }
}

#[test]
// Purpose
// -------
// Verify that HAC standard errors differ from classical SEs on a
// trending series, demonstrating that the robust path is numerically
// active and not simply returning the IID result.
//
// Given
// -----
// - A trending `ACDData` series with `n = 512` and mild positive slope.
// - A shape (p, q) = (1, 1) and Exponential innovation.
// - Baseline `ACDOptions` for fitting.
// - Classical SEs computed with `standard_errors(&data, None)`.
// - HAC SEs computed with:
//   - `HACOptions::new(None, KernelType::Bartlett, true, true)`
//   - Plug-in bandwidth, Bartlett kernel, centering enabled, and
//     Newey–West small-sample correction.
//
// Expect
// ------
// - Fitting succeeds and yields finite parameter estimates.
// - Both classical and HAC SE vectors:
//   - Have equal length.
//   - Contain only finite entries.
// - At least one coordinate of HAC SEs differs materially from the
//   corresponding classical SE (not all coordinates are equal up to
//   numerical noise at `1e-10`).
fn hac_standard_errors_differ_from_classical_on_trending_series() {
    let n = 512;
    let data = make_trending_data(n, 1.0, 0.002);
    let p = 1;
    let q = 1;
    let shape = ACDShape::new(p, q, n).expect("ACDShape::new should accept p, q < n and p + q > 0");
    let opts = default_acd_options();
    let innovation = ACDInnovation::exponential();
    let mut model = ACDModel::new(shape.clone(), innovation, opts, n);
    let theta_dim = 1 + p + q;
    let theta0 = Array1::from_elem(theta_dim, 0.0);
    model.fit(theta0, &data).expect("fit should succeed");
    let se_classical = model.standard_errors(&data, None).expect("classical SEs");
    let hac_opts = HACOptions::new(None, KernelType::Bartlett, true, true);
    let se_hac = model.standard_errors(&data, Some(&hac_opts)).expect("HAC SEs");
    assert_eq!(se_classical.len(), se_hac.len());
    for v in se_classical.iter().chain(se_hac.iter()) {
        assert!(v.is_finite(), "all SE values should be finite");
    }
    // they should differ
    let all_equal = se_hac.iter().zip(se_classical.iter()).all(|(h, c)| (h - c).abs() < 1e-10);
    assert!(!all_equal, "HAC and classical SEs should differ on trending series");
}

#[test]
// Purpose
// -------
// Verify that the ACD API behaves well under a non-default set of
// `ACDOptions` and HAC settings, including tighter tolerances, explicit
// L-BFGS memory, and a non-Bartlett kernel.
//
// Given
// -----
// - A trending `ACDData` series with `n = 256`, base 2.5, and slope 0.01.
// - Shape (p, q) = (1, 1).
// - Weibull innovation with valid shape parameter.
// - `tuned_acd_options()` providing:
//   - Tighter cost/gradient tolerances.
//   - Reduced `max_iter`.
//   - Explicit LBFGS memory and narrower psi guards.
// - HAC configuration:
//   - `HACOptions::new(Some(5), KernelType::QuadraticSpectral, false, false)`
//   - Fixed bandwidth, QS kernel, no centering, no small-sample scaling.
//
// Expect
// ------
// - `ACDModel::fit` converges without error under tuned options.
// - Classical SEs:
//   - Have length `1 + p + q`.
//   - Are finite and non-negative.
// - HAC SEs:
//   - Have length `1 + p + q`.
//   - Are finite.
// - Forecasts for horizon `h = 10` are finite and strictly positive.
fn acd_api_respects_tuned_acdoptions() {
    let n = 256;
    let p = 1;
    let q = 1;
    let data = make_trending_data(n, 2.5, 0.01);
    let shape = ACDShape::new(p, q, n).expect("ACDShape::new should accept p, q < n and p + q > 0");
    let opts = tuned_acd_options();
    let innovation = ACDInnovation::weibull(1.3).expect("valid Weibull shape");
    let mut model = ACDModel::new(shape.clone(), innovation, opts, n);
    let theta_dim = 1 + p + q;
    let theta0 = array![0.1, -0.05, 0.02];
    model.fit(theta0, &data).expect("fit should succeed with tuned options");
    let se_classical = model.standard_errors(&data, None).expect("classical SEs tuned");
    assert_eq!(se_classical.len(), theta_dim);
    assert!(se_classical.iter().all(|v| v.is_finite() && *v >= 0.0));
    let hac_opts = HACOptions::new(Some(5), KernelType::QuadraticSpectral, false, false);
    let se_hac = model.standard_errors(&data, Some(&hac_opts)).expect("HAC SEs tuned");
    assert_eq!(se_hac.len(), theta_dim);
    assert!(se_hac.iter().all(|v| v.is_finite()));
    let forecast_val = model.forecast(10, &data).expect("forecast tuned");
    assert!(forecast_val.is_finite() && forecast_val > 0.0);
}

// Purpose
// -------
// Confirm that the ACD shape constructor rejects the degenerate case
// (p, q) = (0, 0) as an invalid model specification.
//
// Given
// -----
// - Sample size `n = 10`.
// - Shape request `ACDShape::new(0, 0, n)`.
//
// Expect
// ------
// - The constructor returns `Err(ACDError::InvalidModelShape { .. })`.
// - No shape object is created for (0, 0), enforcing the constraint
//   that at least one of p or q must be strictly positive.
#[test]
fn acd_model_handles_t0_offset() {
    let n = 200;
    let t0 = Some(50);
    let data_series = Array1::from_iter((0..n).map(|i| 1.0 + 0.01 * (i as f64)));
    let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
    let data =
        ACDData::new(data_series, t0, meta).expect("ACDData::new should accept t0 within range");
    let p = 1;
    let q = 1;
    let shape = ACDShape::new(p, q, n).expect("ACDShape::new should accept p, q < n and p + q > 0");
    let opts = default_acd_options();
    let innovation = ACDInnovation::exponential();
    let mut model = ACDModel::new(shape.clone(), innovation, opts, n);
    let theta_dim = 1 + p + q;
    let theta0 = Array1::from_elem(theta_dim, 0.0);
    model.fit(theta0, &data).expect("fit should succeed with t0 > 0");
    let se_classical = model.standard_errors(&data, None).expect("classical SEs with t0");
    assert_eq!(se_classical.len(), theta_dim);
    assert!(se_classical.iter().all(|v| v.is_finite() && *v >= 0.0));
    let forecast_val = model.forecast(3, &data).expect("forecast with t0");
    assert!(forecast_val.is_finite() && forecast_val > 0.0);
}
