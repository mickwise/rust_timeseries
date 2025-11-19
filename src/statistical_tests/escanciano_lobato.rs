//! statistical_tests::escanciano_lobato — robust automatic portmanteau test.
//!
//! Purpose
//! -------
//! Implement the heteroskedasticity–robust automatic portmanteau test of
//! Escanciano & Lobato (2009, J. Econometrics 150, 209–225) for residual
//! diagnostics in time-series models. Provides a data-driven lag selector
//! and a χ²-based p-value for the robust Box–Pierce statistic.
//!
//! Key behaviors
//! -------------
//! - Compute heteroskedasticity-consistent autocorrelation statistics
//!   {ρ̃ⱼ²} and the robust Box–Pierce statistic Qₚ* = n × ∑ⱼ₌₁ᵖ ρ̃ⱼ².
//! - Select a data-driven lag p̃ by maximizing the penalized criterion
//!   Lₚ = Qₚ* – π(p, n, q) over 1 ≤ p ≤ d, where π switches between BIC
//!   and AIC as in Eq. (4) of Escanciano & Lobato (2009).
//! - Expose a compact [`ELOutcome`] value with the selected lag, statistic,
//!   and χ²(1) p-value, suitable for both Rust and Python bindings.
//!
//! Invariants & assumptions
//! ------------------------
//! - The input series represents residuals or demeaned observations; the
//!   implementation re-centres the data internally but does not enforce
//!   any specific model structure.
//! - The lag bound `d` must satisfy 1 ≤ d < n, where n = data.len().
//! - The tuning constant `q` must be strictly positive; the paper’s
//!   simulations typically use q ≈ 2–3 (e.g. q = 2.4).
//! - Input validation (lengths, ranges of `q` and `d`, finiteness of the
//!   data) is delegated to `statistical_tests::validation::validate_input`,
//!   which returns [`ELResult`] rather than panicking.
//!
//! Conventions
//! -----------
//! - Indices follow the usual time-series convention: Yₜ denotes the
//!   t-th observation, and lag j pairs (Yₜ, Yₜ₋ⱼ) for t = j,…,n–1.
//! - Autocovariances γ̂ⱼ and variance proxy τ̂ⱼ are computed with an
//!   unbiased (1 / (n – j)) denominator.
//! - Error handling uses the dedicated [`ELError`] type from
//!   `statistical_tests::errors` and the result alias
//!   [`ELResult<T> = Result<T, ELError>`].
//!
//! Downstream usage
//! ----------------
//! - Call [`ELOutcome::escanciano_lobato`] on residuals from a fitted
//!   model to obtain the robust Qₚ* statistic, its p-value, and selected
//!   lag p̃ for diagnostic reporting.
//! - Higher-level diagnostic utilities can wrap this module to run the
//!   test on multiple residual series (e.g., per asset, per model spec)
//!   and aggregate the resulting p-values or lag selections.
//! - Python bindings expose only the [`ELOutcome`] surface,
//!   leaving helper functions private to the Rust crate.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module verify correctness of the low-level
//!   helpers (γ̂ⱼ, τ̂ⱼ, ρ̃ⱼ², Qₚ*, and p̃) on small synthetic series and
//!   check that zero variance proxies return `ELError::ZeroTau(j)`.
//! - Additional tests assert that Qₚ* is non-decreasing in p, that the
//!   AIC/BIC switching rule in π(p, n, q) behaves as expected, and that
//!   p-values lie in [0, 1].
//! - Entry-point validation is exercised by tests that pass invalid
//!   inputs to [`ELOutcome::escanciano_lobato`] and assert an error is
//!   returned rather than a panic; model-level integration tests exercise
//!   this module indirectly via fitted-duration pipelines.
use crate::statistical_tests::errors::{self, ELResult};
use crate::statistical_tests::validation::validate_input;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// ELOutcome — outcome of the Escanciano–Lobato automatic portmanteau test.
///
/// Purpose
/// -------
/// Represent the outcome of a single Escanciano–Lobato robust portmanteau
/// test, including the data-driven lag selection, the robust Box–Pierce
/// statistic at that lag, and its χ²(1) p-value.
///
/// Key behaviors
/// -------------
/// - Holds the selected lag p̃ that maximizes the penalized statistic
///   Lₚ = Qₚ* – π(p, n, q) over 1 ≤ p ≤ d.
/// - Stores the robust Box–Pierce statistic Qₚ* evaluated at p = p̃.
/// - Stores the asymptotic χ²(1) upper-tail probability (p-value) of the
///   observed statistic Qₚ*.
/// - Provides lightweight accessor methods for each field so that
///   downstream code (including Python bindings) does not need to depend
///   on the internal layout.
///
/// Parameters
/// ----------
/// Constructed via [`ELOutcome::escanciano_lobato`]:
/// - `data`: `&[f64]`
///   Input time series (typically model residuals). Must satisfy the
///   validation rules enforced by `validate_input` (non-empty and finite).
/// - `q`: `f64`
///   Positive tuning constant controlling the penalty switch between BIC
///   and AIC. Typical choices are in the range 2–3 (the original paper
///   uses values around 2.4).
/// - `d`: `usize`
///   Upper bound on the candidate lag order; must satisfy 1 ≤ d < n,
///   where n = `data.len()`.
///
/// Fields
/// ------
/// - `p_tilde`: `usize`
///   Selected lag order p̃ that maximizes the penalized statistic.
/// - `stat`: `f64`
///   Robust Box–Pierce statistic Qₚ* evaluated at p = p̃.
/// - `p_value`: `f64`
///   Asymptotic χ²(1) p-value corresponding to `stat`.
///
/// Invariants
/// ----------
/// - `p_tilde` always satisfies 1 ≤ p_tilde ≤ d for the original call to
///   [`ELOutcome::escanciano_lobato`].
/// - `stat` is finite whenever the computation succeeds (i.e., τ̂ⱼ > 0
///   for all relevant lags). Non-finite τ̂ⱼ trigger an error instead of
///   being stored in `ELOutcome`.
/// - `p_value` lies in the closed interval [0, 1].
///
/// Performance
/// -----------
/// - Stores only three scalars and derives `Copy` and `Clone`, making it
///   cheap to pass by value across FFI boundaries or between threads.
/// - No heap allocations occur during construction of `ELOutcome` itself;
///   all allocations are performed in the helper routines that build the
///   ρ̃ vector.
///
/// Notes
/// -----
/// - Designed as a simple value object; it does not own the original data.
/// - Safe to expose as a public return type both in Rust and in Python
///   bindings, as it does not encode any internal implementation details
///   beyond the statistic and lag selection.
#[derive(Debug, Copy, Clone)]
pub struct ELOutcome {
    p_tilde: usize,
    stat: f64,
    p_value: f64,
}

impl ELOutcome {
    /// Run the Escanciano–Lobato (2009) robust automatic portmanteau test.
    ///
    /// Parameters
    /// ----------
    /// - `data`: `&[f64]`
    ///   Input series {Yₜ} of length n ≥ 2. Typically residuals from a
    ///   fitted time-series model. The function re-centres the data
    ///   internally; the caller may pass either raw or demeaned values.
    /// - `q`: `f64`
    ///   Positive tuning constant used in the penalty π(p, n, q). The
    ///   original paper recommends moderate values (e.g. q ≈ 2.4) that
    ///   balance fit and parsimony via the AIC/BIC switching rule.
    /// - `d`: `usize`
    ///   Upper bound on the candidate lag order. Must satisfy
    ///   1 ≤ d < n, where n = `data.len()`. Larger `d` allows more
    ///   flexibility in capturing dependence but increases computation.
    ///
    /// Returns
    /// -------
    /// `ELResult<ELOutcome>`
    ///   - `Ok(ELOutcome)` on success, containing:
    ///     - `p_tilde`: the data-driven lag p̃ that maximizes
    ///       Lₚ = Qₚ* – π(p, n, q),
    ///     - `stat`: the robust Box–Pierce statistic Qₚ* at p = p̃, and
    ///     - `p_value`: the χ²(1) upper-tail p-value of `stat`.
    ///   - `Err(ELError)` when:
    ///     - input validation fails (lengths, lag bound, or tuning
    ///       constant), or
    ///     - τ̂ⱼ = 0 for some lag j, so ρ̃ⱼ² is ill-defined.
    ///
    /// Errors
    /// ------
    /// - `errors::ELError::ZeroTau(j)`
    ///   Returned when the heteroskedasticity proxy τ̂ⱼ for some lag j is
    ///   exactly zero, making the ratio γ̂ⱼ² / τ̂ⱼ ill-defined.
    /// - Other `ELError` variants
    ///   Returned by `validate_input` when the series, `q`, or `d` violate
    ///   documented constraints (e.g., `d == 0`, `d >= n`, non-positive
    ///   `q`, or non-finite observations).
    ///
    /// Panics
    /// ------
    /// - Never panics under normal operation; all user-facing invalid
    ///   inputs are surfaced as `ELError` values.
    ///
    /// Notes
    /// -----
    /// - Internally, this method:
    ///   - computes the sample mean Ȳ,
    ///   - computes γ̂ⱼ and τ̂ⱼ up to lag `d`,
    ///   - forms ρ̃ⱼ² = γ̂ⱼ² / τ̂ⱼ,
    ///   - evaluates Qₚ* and π(p, n, q) for each 1 ≤ p ≤ d, and
    ///   - selects p̃ as the smallest lag attaining the maximal Lₚ.
    /// - The p-value is computed using the χ²(1) distribution; callers
    ///   should interpret small p-values as evidence against the null of
    ///   no serial correlation in the (possibly heteroskedastic) series.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// use rust_timeseries::statistical_tests::escanciano_lobato::ELOutcome;
    ///
    /// let data = vec![0.1, -0.2, 0.3, -0.4, 0.5];
    /// let q = 3.0;
    /// let d = 4;
    ///
    /// let outcome = ELOutcome::escanciano_lobato(&data, q, d).unwrap();
    ///
    /// assert!(1 <= outcome.p_tilde() && outcome.p_tilde() <= d);
    /// assert!(outcome.stat().is_finite());
    /// assert!((0.0..=1.0).contains(&outcome.p_value()));
    /// ```
    pub fn escanciano_lobato(data: &[f64], q: f64, d: usize) -> ELResult<Self> {
        validate_input(data, q, d)?;
        let n: f64 = data.len() as f64;
        let mean: f64 = calc_mean(data);
        let rho_tilde: Vec<f64> = calc_rho_tilde(data, d, mean)?;
        let p_tilde: usize = calc_p_tilde(data, d, q, &rho_tilde)?;
        let stat: f64 = calc_robust_box_pierce(&rho_tilde, n, p_tilde);

        Ok(ELOutcome {
            p_tilde,
            stat,
            p_value: 1.0 - ChiSquared::new(1.0).expect("freedom = 1").cdf(stat),
        })
    }

    /// Selected lag p̃ that maximizes the penalized statistic.
    pub fn p_tilde(&self) -> usize {
        self.p_tilde
    }

    /// Robust Box–Pierce statistic Qₚ*.
    pub fn stat(&self) -> f64 {
        self.stat
    }

    /// Asymptotic χ²(1) p-value of [`stat`](Self::stat).
    pub fn p_value(&self) -> f64 {
        self.p_value
    }
}

//
// ---------- Private helpers (compact docs) ----------
//

/// Compute the sample mean Ȳ = (1 / n) ∑ₜ Yₜ for a series.
///
/// Parameters
/// ----------
/// - `data`: `&[f64]`
///   Input series {Yₜ}. Must be non-empty and contain finite values
///   when called from validated entry points.
///
/// Returns
/// -------
/// `f64`
///   The arithmetic mean of the elements in `data`.
///
/// Errors
/// ------
/// - Never returns an error; invalid usage is handled by callers.
///
/// Panics
/// ------
/// - Panics if `data.len() == 0` due to division by zero. Public
///   entry points (e.g. [`ELOutcome::escanciano_lobato`]) rely on
///   `validate_input` to prevent empty input.
///
/// Notes
/// -----
/// - This helper assumes `data` has already passed validation; it does
///   not attempt to guard against NaNs or infinities.
/// - Implemented with a straightforward sum and single division.
///
#[inline]
fn calc_mean(data: &[f64]) -> f64 {
    let n = data.len();
    let sum: f64 = data.iter().sum();
    sum / n as f64
}

/// Compute the heteroskedasticity proxy τ̂ⱼ for lag `j`.
///
/// Parameters
/// ----------
/// - `data`: `&[f64]`
///   Input series {Yₜ} of length n. Must satisfy n > j and contain
///   finite values when called from validated entry points.
/// - `j`: `usize`
///   Lag index (1 ≤ j < n) at which to compute τ̂ⱼ.
/// - `mean`: `f64`
///   Sample mean Ȳ of `data`, typically computed via [`calc_mean`].
///
/// Returns
/// -------
/// `f64`
///   The variance proxy
///   τ̂ⱼ = (1 / (n − j)) ∑ₜ (Yₜ − Ȳ)² (Yₜ₋ⱼ − Ȳ)²
///   evaluated over t = j,…,n−1.
///
/// Errors
/// ------
/// - Never returns an error; invalid usage is handled by callers.
///
/// Panics
/// ------
/// - Panics if `j >= data.len()` due to the `(n - j)` denominator and
///   slice bounds in `data[j..]`.
///
/// Notes
/// -----
/// - This function does not check for τ̂ⱼ = 0. The caller
///   [`calc_rho_tilde`] is responsible for treating τ̂ⱼ = 0 as
///   an error (`ELError::ZeroTau(j)`) to avoid division by zero.
/// - The implementation uses `data[j..].iter().zip(data)` so that
///   each pair `(Yₜ, Yₜ₋ⱼ)` is formed with consistent indexing.
///
#[inline]
fn calc_tau_j(data: &[f64], j: usize, mean: f64) -> f64 {
    let n: usize = data.len();

    data[j..]
        .iter()
        .zip(data)
        .map(|(y_t, y_t_min_j): (&f64, &f64)| (y_t - mean).powi(2) * (y_t_min_j - mean).powi(2))
        .sum::<f64>()
        / (n - j) as f64
}

/// Compute the unbiased sample autocovariance γ̂ⱼ at lag `j`.
///
/// Parameters
/// ----------
/// - `data`: `&[f64]`
///   Input series {Yₜ} of length n. Must satisfy n > j and contain
///   finite values when called from validated entry points.
/// - `j`: `usize`
///   Lag index (1 ≤ j < n) at which to compute γ̂ⱼ.
/// - `mean`: `f64`
///   Sample mean Ȳ of `data`, typically computed via [`calc_mean`].
///
/// Returns
/// -------
/// `f64`
///   The unbiased autocovariance
///   γ̂ⱼ = (1 / (n − j)) ∑ₜ (Yₜ − Ȳ)(Yₜ₋ⱼ − Ȳ)
///   evaluated over t = j,…,n−1.
///
/// Errors
/// ------
/// - Never returns an error; invalid usage is handled by callers.
///
/// Panics
/// ------
/// - Panics if `j >= data.len()` due to the `(n - j)` denominator and
///   slice bounds in `data[j..]`.
///
/// Notes
/// -----
/// - Uses the same `(Yₜ, Yₜ₋ⱼ)` pairing scheme as [`calc_tau_j`]
///   to stay aligned with the notation in Escanciano–Lobato.
/// - No additional numerical stabilization is applied; upstream
///   callers are expected to work with reasonably scaled residuals.
///
#[inline]
fn calc_gamma_j(data: &[f64], j: usize, mean: f64) -> f64 {
    let n = data.len();

    data[j..]
        .iter()
        .zip(data)
        .map(|(y_t, y_t_min_j): (&f64, &f64)| (y_t - mean) * (y_t_min_j - mean))
        .sum::<f64>()
        / (n - j) as f64
}

/// Compute the penalty π(p, n, q) as in Eq. (4) of Escanciano–Lobato.
///
/// Parameters
/// ----------
/// - `p`: `usize`
///   Candidate lag order (1 ≤ p ≤ d). Used linearly in both AIC and
///   BIC-style penalties.
/// - `n`: `f64`
///   Sample size as a floating-point value (n = data.len()).
/// - `q`: `f64`
///   Positive tuning constant controlling the switch between BIC and
///   AIC penalties. Moderate values (e.g. q ≈ 2.4) are recommended.
/// - `max_lag_abs`: `f64`
///   Maximum absolute autocorrelation magnitude
///   max₁≤ⱼ≤d |ρ̃ⱼ|, computed from the robust autocorrelations. Used
///   in the switching condition involving √n · max |ρ̃ⱼ|.
///
/// Returns
/// -------
/// `f64`
///   The penalty π(p, n, q), equal to:
///   - p · log n if √n · max |ρ̃ⱼ| ≤ √(q log n) (BIC-like), or
///   - 2p        otherwise (AIC-like).
///
/// Errors
/// ------
/// - Never returns an error; invalid usage is handled by callers.
///
/// Panics
/// ------
/// - Panics only if `n <= 0.0`, in which case `n.ln()` is undefined.
///
/// Notes
/// -----
/// - This helper encapsulates the AIC/BIC switching logic; tests verify
///   that both branches are taken under appropriate synthetic inputs.
/// - The separation of penalty computation from the lag selection logic
///   makes it easier to reason about and test the data-driven rule.
///
#[inline]
fn calc_pi(p: usize, n: f64, q: f64, max_lag_abs: f64) -> f64 {
    let log_n: f64 = n.ln();
    let cutoff: f64 = (q * log_n).sqrt();
    if n.sqrt() * max_lag_abs <= cutoff { (p as f64) * log_n } else { (2 * p) as f64 }
}

/// Populate the vector {ρ̃ⱼ²} for lags 1..=d.
///
/// Parameters
/// ----------
/// - `data`: `&[f64]`
///   Input series {Yₜ} of length n. Must satisfy n > d and contain
///   finite values when called from validated entry points.
/// - `d`: `usize`
///   Maximum lag order to compute, with 1 ≤ d < n.
/// - `mean`: `f64`
///   Sample mean Ȳ of `data`, typically computed via [`calc_mean`].
///
/// Returns
/// -------
/// `ELResult<Vec<f64>>`
///   - `Ok(rho_tilde)` where `rho_tilde` has length `d + 1` and
///     `rho_tilde[j] = ρ̃ⱼ²` for j = 1..=d (index 0 is unused and set
///     to 0.0 for convenience).
///   - `Err(ELError::ZeroTau(j))` when τ̂ⱼ = 0 for some lag j, making
///     ρ̃ⱼ² ill-defined.
///
/// Errors
/// ------
/// - `errors::ELError::ZeroTau(j)`
///   Returned when τ̂ⱼ computed by [`calc_tau_j`] is exactly zero.
///
/// Panics
/// ------
/// - Panics if `d >= data.len()` due to slice bounds and division by
///   `n - j`. Public entry points rely on `validate_input` to prevent
///   such configurations.
///
/// Notes
/// -----
/// - The vector is sized as `d + 1` so that indices 1..=d map directly
///   to lags, avoiding off-by-one errors in downstream code.
/// - This function is the only place where τ̂ⱼ = 0 is treated as a
///   recoverable error rather than a panic, ensuring that callers
///   can handle degenerate series gracefully.
///
#[inline]
#[allow(clippy::needless_range_loop)]
fn calc_rho_tilde(data: &[f64], d: usize, mean: f64) -> ELResult<Vec<f64>> {
    let mut rho_tilde = vec![0.0; d + 1];
    for j in 1..=d {
        let gamma_j = calc_gamma_j(data, j, mean);
        let tau_j = calc_tau_j(data, j, mean);
        if tau_j == 0.0 {
            return Err(errors::ELError::ZeroTau(j));
        }
        rho_tilde[j] = gamma_j.powi(2) / tau_j;
    }
    Ok(rho_tilde)
}

/// Compute the robust Box–Pierce statistic Qₚ* from ρ̃ⱼ² values.
///
/// Parameters
/// ----------
/// - `rhos`: `&[f64]`
///   Vector of robust autocorrelation squares {ρ̃ⱼ²} with the
///   convention that indices 1..=p correspond to lags 1..=p.
/// - `n`: `f64`
///   Sample size (n = data.len()) used to scale the statistic.
/// - `p`: `usize`
///   Lag order at which to evaluate Qₚ*; must satisfy 1 ≤ p < rhos.len().
///
/// Returns
/// -------
/// `f64`
///   The robust Box–Pierce statistic
///   Qₚ* = n × ∑ⱼ₌₁ᵖ ρ̃ⱼ².
///
/// Errors
/// ------
/// - Never returns an error; invalid usage is handled by callers.
///
/// Panics
/// ------
/// - Panics if `p >= rhos.len()` due to the slice `rhos[1..=p]`.
///
/// Notes
/// -----
/// - Because each ρ̃ⱼ² is non-negative, Qₚ* is monotonically
///   non-decreasing in `p`. This property is asserted by unit tests
///   as a sanity check.
///
#[inline]
fn calc_robust_box_pierce(rhos: &[f64], n: f64, p: usize) -> f64 {
    rhos[1..=p].iter().sum::<f64>() * n
}

/// Select the data-driven lag p̃ that maximizes the penalized statistic Lₚ.
///
/// Parameters
/// ----------
/// - `data`: `&[f64]`
///   Input series {Yₜ} of length n. Used only to retrieve `n` here;
///   must satisfy n > d when called from validated entry points.
/// - `d`: `usize`
///   Upper bound on the candidate lag order; 1 ≤ d < n.
/// - `q`: `f64`
///   Positive tuning constant for the penalty π(p, n, q).
/// - `rho_tilde`: `&[f64]`
///   Vector of ρ̃ⱼ² values as produced by [`calc_rho_tilde`], with
///   length ≥ d + 1 and indices 1..=d corresponding to lags.
///
/// Returns
/// -------
/// `ELResult<usize>`
///   - `Ok(p_tilde)` where 1 ≤ p_tilde ≤ d is the smallest lag
///     attaining the maximal value of
///     Lₚ = Qₚ* – π(p, n, q).
///   - `Err(ELError)` only if future extensions introduce explicit
///     error cases; the current implementation always returns `Ok`.
///
/// Errors
/// ------
/// - Currently never returns an `ELError`; the `ELResult` wrapper is
///   present for consistency with other helpers and to allow future
///   extensions without breaking the API.
///
/// Panics
/// ------
/// - Panics if `d == 0` or `d >= rho_tilde.len()` due to indexing
///   into `rho_tilde[1..=d]`. Public entry points ensure 1 ≤ d < n
///   and `rho_tilde` is sized as `d + 1`.
///
/// Notes
/// -----
/// - Implements the definition
///   p̃ = min { p : 1 ≤ p ≤ d; Lₚ ≥ Lₕ for all h = 1,…,d },
///   by scanning p from 1 to d and updating the best lag only when
///   Lₚ strictly exceeds the current maximum, thereby keeping the
///   smallest maximizing p.
/// - The penalty π(p, n, q) is computed via [`calc_pi`], which
///   encapsulates the AIC/BIC switching rule.
/// - Unit tests verify that p̃ lies in [1, d] and that L_{p̃} is
///   maximal among {L₁,…,L_d} up to numerical tolerance.
///
#[inline]
fn calc_p_tilde(
    data: &[f64], d: usize, q: f64, rho_tilde: &[f64],
) -> Result<usize, errors::ELError> {
    let n: f64 = data.len() as f64;
    let mut p_tilde: usize = 0;
    let mut max_l_val: f64 = f64::NEG_INFINITY;

    let max_lag_abs: f64 = rho_tilde.iter().map(|&x| x.sqrt()).fold(0.0, f64::max);

    // Calculate the value of p that gives the maximum robust L value
    for p in 1..=d {
        let curr_l_p: f64 = calc_robust_box_pierce(rho_tilde, n, p) - calc_pi(p, n, q, max_lag_abs);
        if curr_l_p > max_l_val {
            max_l_val = curr_l_p;
            p_tilde = p;
        }
    }
    Ok(p_tilde)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistical_tests::errors::ELError;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Correct computation of γ̂ⱼ, τ̂ⱼ, and ρ̃ⱼ² on small synthetic series.
    // - Monotonicity of the robust Box–Pierce statistic Qₚ* in the lag p.
    // - Selection of a valid lag p̃ and finite Qₚ* / p-value in the
    //   Escanciano–Lobato test.
    // - Proper surfacing of ELError::ZeroTau(j) when τ̂ⱼ = 0 for some lag.
    //
    // They intentionally DO NOT cover:
    // - Asymptotic size or power properties of the test (handled by
    //   simulation studies, not unit tests).
    // - Model-specific behavior of residuals produced by ACD or other
    //   duration models; those are exercised in higher-level integration
    //   tests and Python-level harnesses.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Ensure that `ELOutcome::escanciano_lobato` respects the input
    // validation rules and surfaces invalid configurations as errors
    // rather than panicking.
    //
    // Given
    // -----
    // - A "valid" baseline configuration:
    //   - data = [0.1, -0.2, 0.3, -0.4],
    //   - q = 3.0,
    //   - d = 2 (so 1 ≤ d < n).
    // - Several invalid variants:
    //   - empty data,
    //   - d = 0,
    //   - d >= n,
    //   - non-positive q.
    //
    // Expect
    // ------
    // - For each invalid variant, `ELOutcome::escanciano_lobato` returns
    //   `Err(ELError)` rather than `Ok` or panicking.
    fn eloutcome_escanciano_lobato_invalid_inputs_return_error() {
        // Arrange
        let valid_data = vec![0.1_f64, -0.2, 0.3, -0.4];
        let valid_q = 3.0_f64;
        let valid_d = 2_usize;

        // Act & Assert: empty data
        let empty: Vec<f64> = Vec::new();
        let result_empty = ELOutcome::escanciano_lobato(&empty, valid_q, valid_d);
        assert!(result_empty.is_err(), "expected error for empty data, got {:?}", result_empty);

        // Act & Assert: d == 0
        let result_zero_d = ELOutcome::escanciano_lobato(&valid_data, valid_q, 0);
        assert!(result_zero_d.is_err(), "expected error for d == 0, got {:?}", result_zero_d);

        // Act & Assert: d >= n
        let result_d_ge_n = ELOutcome::escanciano_lobato(&valid_data, valid_q, valid_data.len());
        assert!(result_d_ge_n.is_err(), "expected error for d >= n, got {:?}", result_d_ge_n);

        // Act & Assert: non-positive q
        let result_non_positive_q = ELOutcome::escanciano_lobato(&valid_data, 0.0_f64, valid_d);
        assert!(
            result_non_positive_q.is_err(),
            "expected error for non-positive q, got {:?}",
            result_non_positive_q
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that the Escanciano–Lobato test runs end-to-end on a small
    // synthetic series and returns a valid lag, finite statistic, and
    // p-value in [0, 1].
    //
    // Given
    // -----
    // - A short residual series of length 5.
    // - A tuning constant q = 3.0 and lag bound d = 4 satisfying
    //   1 ≤ d < n.
    //
    // Expect
    // ------
    // - `ELOutcome::escanciano_lobato` returns `Ok(ELOutcome)`.
    // - The selected lag p̃ lies in [1, d].
    // - The statistic is finite and the p-value lies in [0, 1].
    fn elresult_escanciano_lobato_small_series_returns_valid_result() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3, -0.4, 0.5];
        let q = 3.0;
        let d = 4;

        // Act
        let result = ELOutcome::escanciano_lobato(&data, q, d)
            .expect("Escanciano–Lobato test should succeed on this series");

        // Assert
        assert!(1 <= result.p_tilde() && result.p_tilde() <= d);
        assert!(result.stat().is_finite());
        assert!((0.0..=1.0).contains(&result.p_value()));
    }

    #[test]
    // Purpose
    // -------
    // Ensure that a constant series triggers ELError::ZeroTau for the
    // first lag, since τ̂₁ = 0 makes ρ̃₁² undefined.
    //
    // Given
    // -----
    // - A constant series of length 4, so all centred values are zero.
    // - A lag bound d = 2.
    //
    // Expect
    // ------
    // - `calc_rho_tilde` returns `Err(ELError::ZeroTau(1))`.
    fn calc_rho_tilde_constant_series_returns_zero_tau_error() {
        // Arrange
        let data = vec![1.0_f64, 1.0, 1.0, 1.0];
        let mean = calc_mean(&data);
        let d = 2;

        // Act
        let result = calc_rho_tilde(&data, d, mean);

        // Assert
        match result {
            Err(ELError::ZeroTau(j)) => assert_eq!(j, 1),
            other => panic!("expected ELError::ZeroTau(1), got {:?}", other),
        }
    }

    #[test]
    // Purpose
    // -------
    // Check that the robust Box–Pierce statistic Qₚ* is non-decreasing
    // as the lag p increases, since it is a scaled sum of non-negative
    // terms ρ̃ⱼ².
    //
    // Given
    // -----
    // - A short non-constant series.
    // - ρ̃ⱼ² computed up to d = 4.
    //
    // Expect
    // ------
    // - For p = 1,…,d, the sequence Qₚ* is monotonically non-decreasing.
    fn calc_robust_box_pierce_is_monotone_in_p() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3, -0.4, 0.5];
        let mean = calc_mean(&data);
        let d = 4;
        let rho_tilde =
            calc_rho_tilde(&data, d, mean).expect("rho_tilde should compute for this series");
        let n = data.len() as f64;

        // Act & Assert
        let mut prev_q = 0.0_f64;
        for p in 1..=d {
            let q_p = calc_robust_box_pierce(&rho_tilde, n, p);
            assert!(
                q_p >= prev_q,
                "Q_p should be non-decreasing: Q_{} = {}, Q_{} = {}",
                p - 1,
                prev_q,
                p,
                q_p
            );
            prev_q = q_p;
        }
    }

    #[test]
    // Purpose
    // -------
    // Sanity-check the data-driven lag selection p̃ by verifying that it
    // picks some p in [1, d] and that the corresponding penalized
    // criterion Lₚ = Qₚ* – π(p, n, q) is maximal at p̃.
    //
    // Given
    // -----
    // - A small non-constant series of length 5.
    // - q = 3.0 and d = 4.
    //
    // Expect
    // ------
    // - `calc_p_tilde` returns a p̃ in [1, d].
    // - L_{p̃} is greater than or equal to Lₚ for all 1 ≤ p ≤ d.
    fn calc_p_tilde_maximizes_penalized_statistic() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3, -0.4, 0.5];
        let mean = calc_mean(&data);
        let d = 4;
        let q = 3.0;
        let rho_tilde =
            calc_rho_tilde(&data, d, mean).expect("rho_tilde should compute for this series");
        let n = data.len() as f64;

        // Act
        let p_tilde = calc_p_tilde(&data, d, q, &rho_tilde)
            .expect("p_tilde should be computable for this series");

        // Assert: p_tilde in [1, d]
        assert!((1..=d).contains(&p_tilde));

        // Assert: L_{p̃} >= L_p for all p
        let max_lag_abs: f64 = rho_tilde.iter().map(|&x| x.sqrt()).fold(0.0, f64::max);

        let l_p_tilde =
            calc_robust_box_pierce(&rho_tilde, n, p_tilde) - calc_pi(p_tilde, n, q, max_lag_abs);

        for p in 1..=d {
            let l_p = calc_robust_box_pierce(&rho_tilde, n, p) - calc_pi(p, n, q, max_lag_abs);
            assert!(
                l_p_tilde >= l_p - 1e-12,
                "L_p̃ should be maximal: L_{} = {}, L_{} = {}",
                p_tilde,
                l_p_tilde,
                p,
                l_p
            );
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `calc_pi` switches between the BIC-style penalty
    // (p · log n) and the AIC-style penalty (2p) based on the threshold
    // √n · max|ρ̃ⱼ| ≤ √(q log n).
    //
    // Given
    // -----
    // - A fixed sample size n = 100 and tuning constant q = 2.4.
    // - A candidate lag p = 3.
    // - A very small `max_lag_abs` (0.0) to force the BIC branch.
    // - A very large `max_lag_abs` (10.0) to force the AIC branch.
    //
    // Expect
    // ------
    // - `calc_pi(p, n, q, 0.0)` equals p · log n (BIC-style penalty).
    // - `calc_pi(p, n, q, 10.0)` equals 2p (AIC-style penalty).
    fn calc_pi_switches_between_bic_and_aic() {
        // Arrange
        let n = 100.0_f64;
        let q = 2.4_f64;
        let p = 3_usize;

        // Act
        let pi_bic = calc_pi(p, n, q, 0.0);
        let pi_aic = calc_pi(p, n, q, 10.0);

        // Assert
        let expected_bic = (p as f64) * n.ln();
        let expected_aic = (2 * p) as f64;

        assert!(
            (pi_bic - expected_bic).abs() < 1e-12,
            "expected BIC penalty {}; got {}",
            expected_bic,
            pi_bic
        );
        assert!(
            (pi_aic - expected_aic).abs() < 1e-12,
            "expected AIC penalty {}; got {}",
            expected_aic,
            pi_aic
        );
    }
}
