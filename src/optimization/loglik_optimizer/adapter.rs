//! Adapter that exposes a user `LogLikelihood` as an `argmin` problem.
//!
//! We convert a *maximization* of a log-likelihood `ℓ(θ)` into a *minimization*
//! problem by defining the cost as `c(θ) = -ℓ(θ)`. Analytic gradients (if
//! provided by the user) are negated accordingly. If a gradient is not
//! provided, we finite-difference the **cost** closure, so no sign flip is
//! needed in that branch.
use crate::optimization::{
    errors::OptError,
    loglik_optimizer::{
        finite_diff::run_fd_diff,
        traits::LogLikelihood,
        types::{Cost, Grad, Theta},
        validation::validate_grad,
    },
};
use argmin::core::{CostFunction, Error, Gradient};
use finitediff::FiniteDiff;
use std::cell::RefCell;

/// Bridges a user `LogLikelihood` to `argmin`'s `CostFunction` and `Gradient`.
///
/// - `CostFunction::cost` returns `-ℓ(θ)` (negative log-likelihood).
/// - `Gradient::gradient` returns:
///   - `-∇ℓ(θ)` if the user provides an analytic gradient, or
///   - a finite-difference gradient of the cost (no sign flip needed).
#[derive(Debug, Clone)]
pub struct ArgMinAdapter<'a, F: LogLikelihood> {
    pub f: &'a F,
    pub data: &'a F::Data,
}

impl<'a, F: LogLikelihood> CostFunction for ArgMinAdapter<'a, F> {
    type Param = Theta;
    type Output = Cost;

    /// Evaluate the cost `c(θ) = -ℓ(θ)`.
    ///
    /// - Calls the user's `value(θ, data)` and checks the result is finite.
    /// - Returns `Error(NonFiniteCost)` if the value is not finite.
    ///
    /// # Errors
    /// Propagates any `OptError` from the user’s `value` via `?`.
    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        let output = self.f.value(theta, self.data)?;
        if !output.is_finite() {
            return Err((OptError::NonFiniteCost { value: output }).into());
        }
        Ok(-output)
    }
}

impl<'a, F: LogLikelihood> Gradient for ArgMinAdapter<'a, F> {
    type Param = Theta;
    type Gradient = Grad;

    /// Evaluate the gradient of the cost at `θ`.
    ///
    /// Behavior:
    /// - If the user implements `grad(θ, data)`, we validate it and return `-grad`
    ///   (because the cost is `-ℓ`).
    /// - Otherwise, we compute a finite-difference gradient of the **cost**:
    ///   - Try *central* differences first.
    ///   - If any evaluation of the `cost` closure failed (captured via
    ///     `closure_err`), retry with *forward* differences.
    ///   - Validate the FD gradient; if it fails (e.g., non-finite), retry once
    ///     with *forward* differences and validate again.
    ///
    /// Implementation notes:
    /// - The FD closure must return `f64`, so we can’t use `?` inside it; we capture
    ///   the first error in `closure_err` and return `NaN` from the closure. After
    ///   FD, we turn that captured error back into a real error (or switch to
    ///   forward diff).
    ///
    /// # Errors
    /// - Propagates user errors from `grad` (non-`GradientNotImplemented`).
    /// - Propagates any error raised by cost evaluations performed during FD.
    /// - Returns validation errors if the gradient has wrong dimension or
    ///   non-finite entries.
    fn gradient(&self, theta: &Self::Param) -> Result<Self::Gradient, Error> {
        let dim = theta.len();
        match self.f.grad(theta, self.data) {
            Ok(g) => {
                validate_grad(&g, dim)?;
                Ok(-g)
            }
            Err(e) => {
                let closure_err: RefCell<Option<Error>> = RefCell::new(None);
                match e {
                    OptError::GradientNotImplemented => {
                        let cost_func = |theta: &Theta| -> f64 {
                            match self.cost(theta) {
                                Ok(val) => val,
                                Err(e) => {
                                    let mut slot = closure_err.borrow_mut();
                                    if slot.is_none() {
                                        *slot = Some(e);
                                    }
                                    f64::NAN
                                }
                            }
                        };
                        let mut fd_grad = theta.central_diff(&cost_func);
                        if closure_err.borrow().is_some() {
                            fd_grad = run_fd_diff(theta, &cost_func, &closure_err)?;
                            return Ok(fd_grad);
                        }
                        match validate_grad(&fd_grad, dim) {
                            Ok(()) => Ok(fd_grad),
                            Err(_) => {
                                fd_grad = run_fd_diff(theta, &cost_func, &closure_err)?;
                                Ok(fd_grad)
                            }
                        }
                    }
                    _ => Err(e.into()),
                }
            }
        }
    }
}

impl<'a, F: LogLikelihood> ArgMinAdapter<'a, F> {
    /// Construct a new adapter over a user `LogLikelihood` and its data.
    pub fn new(f: &'a F, data: &'a F::Data) -> Self {
        Self { f, data }
    }
}
