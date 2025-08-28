//! Initialization policies for pre-sample lags in ACD(p, q).
//!
//! An ACD(p, q) recursion requires *p* lagged ψ values and *q* lagged duration
//! values before the first observed sample. This module defines [`Init`], which
//! specifies how those pre-sample lags are seeded.
//!
//! ## Policies
//! - [`Init::UncondMean`]: fill ψ and duration lags with the model’s unconditional mean μ.
//! - [`Init::SampleMean`]: fill with the sample mean of the observed durations x̄.
//! - [`Init::Fixed`]: fill with a user-provided strictly positive scalar.
//! - [`Init::FixedVector`]: caller supplies explicit pre-sample vectors of the
//!   correct lengths.
//!
//! ## Invariants
//! - All lag values must be finite and strictly positive.
//! - Vector lengths must match `p` (ψ) and `q` (durations).
use crate::duration::{
    core::validation::{verify_duration_lags, verify_psi_lags},
    errors::{ACDError, ACDResult},
};
use ndarray::Array1;

/// Initialization policy for pre-sample lags in an ACD(p, q) model.
///
/// These policies specify how to fill the missing ψ and duration lags that are
/// required to start the recursion.
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
    /// Initialize all ψ and duration lags with the unconditional mean μ.
    pub const fn uncond_mean() -> Self {
        Init::UncondMean
    }

    /// Initialize all ψ and duration lags with the sample mean x̄.
    pub const fn sample_mean() -> Self {
        Init::SampleMean
    }

    /// Initialize all ψ and duration lags with a fixed positive scalar.
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidInitFixed`] if `value` is not finite or `<= 0`.
    pub fn fixed(value: f64) -> ACDResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(ACDError::InvalidInitFixed { value });
        }
        Ok(Init::Fixed(value))
    }

    /// Initialize with explicit vectors of ψ and duration lags.
    ///
    /// # Errors
    /// - Returns [`ACDError::InvalidPsiLength`] if `psi_lags.len() != p`.
    /// - Returns [`ACDError::InvalidDurationLength`] if `duration_lags.len() != q`.
    /// - Returns element-level errors if any values are non-finite or `<= 0`.
    pub fn fixed_vector(
        psi_lags: Array1<f64>, duration_lags: Array1<f64>, p: usize, q: usize,
    ) -> ACDResult<Self> {
        verify_psi_lags(&psi_lags, q)?;
        verify_duration_lags(&duration_lags, p)?;

        Ok(Init::FixedVector { psi_lags, duration_lags })
    }
}
