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
    /// Construct an [`ACDShape`] = ACD(p, q).
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidModelShape`] if `p == 0 && q == 0`
    /// (a degenerate model with no dynamics).
    pub fn new(p: usize, q: usize) -> ACDResult<Self> {
        if q == 0 && p == 0 {
            return Err(ACDError::InvalidModelShape { param: p as f64 });
        }
        Ok(ACDShape { p, q })
    }
}
