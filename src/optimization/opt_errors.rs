/// Crate-wide result alias for optimizer operations.
pub type OptResult<T> = Result<T, OptError>;

pub enum OptError {
    InvalidTheta(String),
    InvalidData(String),
    OptimizationFailed(String),
    ConvergenceError(String),
}
