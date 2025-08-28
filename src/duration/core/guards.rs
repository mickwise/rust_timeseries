//! ψ-guards for ACD models.
//!
//! This module defines [`PsiGuards`], a simple container that enforces lower
//! and upper bounds on the conditional mean duration process `ψ_t`.
//!
//! ## Purpose
//! During recursion, numerical issues or invalid parameters can cause `ψ_t` to
//! collapse toward 0 or explode. Guards clamp each `ψ_t` into a safe range
//! `[min, max]`, preventing NaNs or overflows in the log-likelihood.
//!
//! ## Invariants
//! - `min < max`
//! - both bounds must be finite
//! - `min > 0` (to keep log terms well-defined)
use crate::duration::errors::{ACDError, ACDResult};

/// Bounds for the ψ recursion during estimation.
///
/// These guards enforce a valid operating range for ψ to prevent divergence
/// during likelihood maximization or log(0) errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsiGuards {
    /// Lower bound for ψ (strictly > 0).
    pub min: f64,
    /// Upper bound for ψ (must be > `min`).
    pub max: f64,
}

impl PsiGuards {
    /// Construct validated ψ bounds.
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidPsiGuards`] if:
    /// - `min >= max`
    /// - either bound is not finite
    /// - `min <= 0.0`
    pub fn new(value: (f64, f64)) -> ACDResult<Self> {
        if value.0 >= value.1 {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must have min < max.",
            });
        }

        if !value.0.is_finite() || !value.1.is_finite() {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must be finite.",
            });
        }

        if value.0 <= 0.0 {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must be strictly positive.",
            });
        }

        Ok(PsiGuards { min: value.0, max: value.1 })
    }
}
