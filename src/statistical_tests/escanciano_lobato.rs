//! Robust automatic portmanteau test following
//! **Escanciano & Lobato (2009, J. Econometrics 150 pp. 209–225)**.
//!
//! Implements the heteroskedasticity–consistent statistic:
//!
//!     Qₚ* = n × ∑ⱼ₌₁ᵖ ρ̃ⱼ²
//!
//! along with the data-driven lag selector:
//!
//!     p̃ = arg max₁≤p≤d Lₚ
//!
//! where Lₚ = Qₚ* – π(p,n,q) and π(p,n,q) switches between BIC and AIC
//! penalties according to Eq. (4) of the paper.

use crate::statistical_tests::stat_tests_errors;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of the Escanciano–Lobato automatic portmanteau test.
///
/// * `p_tilde` — data-driven number of autocorrelation lags  
/// * `stat`    — robust Box–Pierce statistic Qₚ* at p = p̃  
/// * `p_value` — asymptotic χ²(1) tail probability of `stat`
#[derive(Debug, Copy, Clone)]
pub struct ELResult {
    /// Selected lag p̃ that maximizes the penalized statistic.
    p_tilde: usize,
    /// Robust Box–Pierce statistic Qₚ* at p = p̃.
    stat: f64,
    /// Asymptotic χ²(1) tail probability of `stat`.
    p_value: f64,
}

impl ELResult {
    /// Run the automatic portmanteau test of Escanciano & Lobato (2009).
    ///
    /// # Arguments
    /// * `data` — slice of centred (or raw) observations {Yₜ}  
    /// * `q`    — positive tuning constant in the penalty switch (paper suggests 3–4)  
    /// * `d`    — upper bound on candidate lags, 1 ≤ d < data.len()
    ///
    /// # Errors
    /// * `ELError::ZeroTau(j)` when the heteroskedasticity proxy τ̂ⱼ is zero.
    ///
    /// # Returns
    /// [`ELResult`] with statistic, p-value, and selected lag.
    ///
    /// # Examples
    ///
    /// ```rust, no_run
    /// use rust_pyo3_timeseries::tests::escanciano_lobato::ELResult;
    ///
    /// let data = vec![0.1, -0.2, 0.3, -0.4, 0.5];
    /// let result = ELResult::escanciano_lobato(&data, 3.0, 4).unwrap();
    ///
    /// assert!(1 <= result.p_tilde() && result.p_tilde() <= 4);
    /// println!("Q* = {:.3}, p-value = {:.3}", result.stat(), result.p_value());
    /// ```
    pub fn escanciano_lobato(
        data: &[f64],
        q: f64,
        d: usize,
    ) -> Result<Self, stat_tests_errors::ELError> {
        let n: f64 = data.len() as f64;
        let mean: f64 = calc_mean(data);
        let rho_tilde: Vec<f64> = calc_rho_tilde(data, d, mean)?;
        let p_tilde: usize = calc_p_tilde(data, d, q, &rho_tilde)?;
        let stat: f64 = calc_robust_box_pierce(&rho_tilde, n, p_tilde);

        Ok(ELResult {
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

/// Sample mean Ȳ = (1 / n) ∑ₜ Yₜ.
#[inline]
fn calc_mean(data: &[f64]) -> f64 {
    let n = data.len();
    let sum: f64 = data.iter().sum();
    sum / n as f64
}

/// Heteroskedasticity proxy  
/// τ̂ⱼ = (1 / (n – j)) ∑ₜ (Yₜ – Ȳ)² (Yₜ₋ⱼ – Ȳ)².
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

/// Unbiased autocovariance  
/// γ̂ⱼ = (1 / (n – j)) ∑ₜ (Yₜ – Ȳ)(Yₜ₋ⱼ – Ȳ).
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

/// Penalty π(p,n,q) (Eq. 4).  Switches between BIC (p · log n)
/// and AIC (2 p) using the condition  
/// √n · maxⱼ |ρ̃ⱼ| ≤ √(q · log n).
#[inline]
fn calc_pi(p: usize, n: f64, q: f64, max_lag_abs: f64) -> f64 {
    let log_n: f64 = n.ln();
    let cutoff: f64 = (q * log_n).sqrt();
    if n.sqrt() * max_lag_abs <= cutoff {
        (p as f64) * log_n
    } else {
        (2 * p) as f64
    }
}

/// Populate the vector {ρ̃ⱼ²}₍ⱼ₌₁₎ᵈ with  
/// ρ̃ⱼ² = γ̂ⱼ² ⁄ τ̂ⱼ.
#[inline]
fn calc_rho_tilde(
    data: &[f64],
    d: usize,
    mean: f64,
) -> Result<Vec<f64>, stat_tests_errors::ELError> {
    let mut rho_tilde = vec![0.0; d + 1];
    for j in 1..=d {
        let gamma_j = calc_gamma_j(data, j, mean);
        let tau_j = calc_tau_j(data, j, mean);
        if tau_j == 0.0 {
            return Err(stat_tests_errors::ELError::ZeroTau(j));
        }
        rho_tilde[j] = gamma_j.powi(2) / tau_j;
    }
    Ok(rho_tilde)
}

/// Robust Box–Pierce statistic  
/// Qₚ* = n × ∑ⱼ₌₁ᵖ ρ̃ⱼ².
#[inline]
fn calc_robust_box_pierce(rhos: &[f64], n: f64, p: usize) -> f64 {
    rhos[1..=p].iter().sum::<f64>() * n
}

/// Automatic lag selector p̃ = min { p : Lₚ ≥ Lₕ ∀ h ≤ d },  
/// where Lₚ = Qₚ* – π(p,n,q).
#[inline]
fn calc_p_tilde(
    data: &[f64],
    d: usize,
    q: f64,
    rho_tilde: &[f64],
) -> Result<usize, stat_tests_errors::ELError> {
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
