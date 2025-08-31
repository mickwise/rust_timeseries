//! Model order (p, q) for ACD models.
//!
//! In the Engle–Russell / Wikipedia convention:
//! - `q`: number of **duration lags** (coefficients α₁…α_q on past τ’s).
//! - `p`: number of **ψ lags** (coefficients β₁…β_p on past ψ’s).
//!
//! At least one of `p` or `q` must be > 0 to have dynamics.
use crate::duration::errors::{ACDError, ACDResult};

/// Order of the ACD(p, q) model.
///
/// - `q`: number of lagged durations (α terms)  
/// - `p`: number of lagged conditional means ψ (β terms)
///
/// Invariant: not both zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ACDShape {
    pub p: usize,
    pub q: usize,
}

impl ACDShape {
    /// Construct an [`ACDShape`] = ACD(p, q) and validate it against the sample size `n`.
    ///
    /// # Invariants
    /// - Not both zero: at least one of `p` or `q` must be > 0.
    /// - Sufficient data: `p < n` and `q < n` so that the recursion has enough
    ///   in-sample observations to seed and run.
    ///
    /// # Arguments
    /// - `p`: number of ψ lags (β terms).
    /// - `q`: number of duration lags (α terms).
    /// - `n`: number of available observations in the sample you plan to fit.
    ///
    /// # Errors
    /// - [`ACDError::InvalidModelShape`] if `p == 0 && q == 0`.
    /// - [`ACDError::InvalidModelShape`] if `p >= n` or `q >= n` (insufficient sample).
    ///
    /// # Rationale
    /// The ψ–recursion requires `p` prior ψ values and `q` prior durations
    /// (or valid seeds) to be well-defined. Guarding here fails fast on
    /// under-identified specifications so downstream fitting/forecasting
    /// can assume `n >= max(p, q)`.
    pub fn new(p: usize, q: usize, n: usize) -> ACDResult<Self> {
        if p == 0 && q == 0 {
            return Err(ACDError::InvalidModelShape {
                param: p,
                reason: "Both p and q cannot be zero.",
            });
        }
        if p >= n {
            return Err(ACDError::InvalidModelShape {
                param: p,
                reason: "p must be less than the number of observations.",
            });
        }
        if q >= n {
            return Err(ACDError::InvalidModelShape {
                param: q,
                reason: "q must be less than the number of observations.",
            });
        }
        Ok(ACDShape { p, q })
    }
}
