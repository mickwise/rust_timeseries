use ndarray::Array1;

use crate::optimization::opt_errors::OptResult;
pub trait LogLikelihood: Sync + Send {
    type Theta: Clone + Send + Sync + 'static;
    type Data: Sync + 'static;

    fn value(&self, theta: &Self::Theta, data: &Self::Data) -> OptResult<f64>;
    fn grad(&self, theta: &Self::Theta, data: &Self::Data) -> OptResult<Array1<f64>>;
    fn check(&self, theta: &Self::Theta, data: &Self::Data) -> OptResult<()>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimOutcome<T> {
    pub theta_hat: T,
    pub value: f64,
    pub converged: bool,
    pub status: String,
    pub iterations: usize,
    pub fn_evals: usize,
    pub grad_norm: f64,
}

pub struct MLEOptions {
    pub max_iter: usize,
    pub tol_grad: f64,
    pub tol_change: f64,
    pub tol_f: f64,
    pub verbose: bool,
}

pub fn maximize<F: LogLikelihood, T>(
    f: &F,
    theta0: &F::Theta,
    data: &F::Data,
    opts: &MLEOptions,
) -> OptResult<OptimOutcome<T>> {
    todo!("Implement the optimization logic here");
}
