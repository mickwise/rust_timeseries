use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

/// Occurs when the heteroskedasticity proxy τ̂ⱼ evaluates to zero,
/// i.e. no variation in the j-lag products.
#[derive(Debug)]
pub enum ELError {
    ZeroTau(usize),
}

impl std::error::Error for ELError {}

impl std::fmt::Display for ELError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ELError::ZeroTau(lag) => write!(f, "Zero τ̂ value at lag {lag}"),
        }
    }
}

impl std::convert::From<ELError> for PyErr {
    fn from(err: ELError) -> PyErr {
        PyOSError::new_err(err.to_string())
    }
}
