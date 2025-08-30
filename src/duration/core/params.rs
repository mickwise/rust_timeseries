//! Parameterization for ACD(p, q) models.
//!
//! This module defines the **constrained model-space** parameters [`ACDParams`]
//! and the **unconstrained optimizer-space** parameters [`ACDTheta`] together
//! with stable, invertible mappings between the two.
//!
//! Key ideas
//! ---------
//! - **Model space (`ACDParams`)**: parameters live on their natural domain
//!   (ω > 0, αᵢ ≥ 0, βⱼ ≥ 0, ∑α + ∑β < 1). The ψ-recursion and likelihood are
//!   evaluated in this space.
//! - **Optimizer space (`ACDTheta`)**: parameters are mapped to ℝᵏ via
//!   softplus/softmax so numerical optimizers can search without constraints.
//! - **Stationarity margin**: we enforce strict stationarity with a tiny buffer
//!   (default 1e-6) to avoid boundary pathologies.
//!
//! The bidirectional mapping (`ACDParams` ↔ `ACDTheta`) guarantees every optimizer
//! iterate corresponds to a valid, strictly stationary ACD model.
//!
//! Invariants
//! ----------
//! - ω > 0
//! - α, β elementwise ≥ 0
//! - ∑α + ∑β < 1 − margin
//! - slack ≥ 0 and ∑α + ∑β + slack = 1 − margin
use std::cell::RefCell;

use ndarray::{Array1, Array2};
#[derive(Debug, Clone, PartialEq)]
pub struct ACDScratch {
    /// Scratch buffer for α.
    pub alpha_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for β.
    pub beta_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for ψ.
    pub psi_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for initial durations.
    pub dur_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for derivatives.
    pub deriv_buf: RefCell<Array2<f64>>,
}

impl ACDScratch {
    pub fn new(n: usize, p: usize, q: usize) -> ACDScratch {
        let alpha_buf = RefCell::new(Array1::zeros(q));
        let beta_buf = RefCell::new(Array1::zeros(p));
        let psi_buf = RefCell::new(Array1::zeros(n + p));
        let dur_buf = RefCell::new(Array1::zeros(q));
        let deriv_buf = RefCell::new(Array2::zeros((n + p, 1 + p + q)));
        ACDScratch { alpha_buf, beta_buf, psi_buf, dur_buf, deriv_buf }
    }
}
