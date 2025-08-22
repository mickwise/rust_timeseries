use ndarray::{Array1, s};

use crate::duration::duration_errors::{ParamError, ParamResult};

/// Consts
const STATIONARITY_MARGIN: f64 = 1e-6; // Small value to ensure strict stationarity

/// Model-space parameters for an ACD(p, q) model.
///
/// This is the "natural" constrained space used by the recursion and likelihood:
/// - `omega > 0`
/// - `alpha[i] >= 0` for i = 0..p-1  (coeffs on lagged durations)
/// - `beta[j]  >= 0` for j = 0..q-1  (coeffs on lagged ψ)
/// - `sum(alpha) + sum(beta) < 1`    (strict stationarity)
///
/// With unit-mean innovations, the unconditional mean of durations is
/// `mu = omega / (1.0 - sum(alpha) - sum(beta))`.
///
/// The struct stores values that already satisfy these constraints; use
/// [`ACDParams::new`] to validate an instance built by hand.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDParams {
    pub omega: f64,
    pub slack: f64,
    pub alpha: Array1<f64>,
    pub beta: Array1<f64>,
}

impl ACDParams {
    pub fn to_theta(&self, p: usize, q: usize) -> Array1<f64> {
        let mut theta = Array1::<f64>::zeros(self.alpha.len() + self.beta.len() + 2);
        theta[0] = (self.omega.exp() - 1.0).ln();
        // theta[1..p + 1] = self.alpha.mapv(|x | (x/(1.0 - STATIONARITY_MARGIN)).ln());
        todo!()
    }

    /// Return the implied unconditional mean of durations,
    /// `mu = omega / (1 - sum(alpha) - sum(beta))`.
    ///
    /// Assumes the instance is already validated and strictly stationary.
    pub fn uncond_mean(&self) -> f64 {
        let sum_alpha = self.alpha.sum();
        let sum_beta = self.beta.sum();
        self.omega / (1.0 - sum_alpha - sum_beta)
    }
}

pub struct ACDTheta {
    pub theta: Array1<f64>,
}

impl ACDTheta {
    pub fn to_params(&self, p: usize, q: usize) -> ParamResult<ACDParams> {
        if self.theta.len() != p + q + 2 {
            return Err(ParamError::ThetaLengthMismatch {
                expected: p + q + 1,
                actual: self.theta.len(),
            });
        }

        let omega = (1.0 + self.theta[0].exp()).ln();
        let coeffs = safe_softmax(&self.theta.slice(s![1..]).to_owned());
        let slack = coeffs.last().unwrap() * (1.0 - STATIONARITY_MARGIN);
        let alpha = coeffs
            .slice(s![0..p])
            .to_owned()
            .mapv_into(|x| x * (1.0 - STATIONARITY_MARGIN));
        let beta = coeffs
            .slice(s![p..p + q])
            .to_owned()
            .mapv_into(|x| x * (1.0 - STATIONARITY_MARGIN));

        Ok(ACDParams {
            omega,
            alpha,
            beta,
            slack,
        })
    }
}

/// ---- Helper Methods ----
fn safe_softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_x = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max_x).exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}
