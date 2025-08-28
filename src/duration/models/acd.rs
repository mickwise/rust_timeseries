use std::cell::RefCell;

use ndarray::Array1;

use crate::{
    duration::core::{
        data::ACDData, innovations::ACDInnovation, options::ACDOptions, psi::likelihood_driver,
        shape::ACDShape, validation::validate_theta, workspace::WorkSpace,
    },
    optimization::{
        errors::OptResult,
        loglik_optimizer::{Grad, LogLikelihood, Theta},
    },
};

pub struct ACDModel {
    /// ACD(p, q) model order.
    pub shape: ACDShape,
    /// Innovation distribution with unit-mean parametrization.
    pub innovation: ACDInnovation,
    /// Model options.
    pub options: ACDOptions,
    /// Scratch buffer for α.
    pub alpha_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for β.
    pub beta_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for ψ.
    pub psi_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for initial durations.
    pub dur_buf: RefCell<Array1<f64>>,
}

impl ACDModel {
    /// Construct a new [`ACDModel`] instance.
    ///
    /// # Arguments
    /// - `shape`: model order (p, q) with p, q ≥ 0 and p + q > 0.
    /// - `innovation`: innovation distribution with unit-mean parametrization.
    /// - `options`: model options.
    /// - `n`: length of the data.
    ///
    /// # Returns
    /// A new `ACDModel` instance.
    ///
    /// # Panics
    /// This constructor does not panic; invalid shapes should be caught by
    /// `ACDShape::new`.
    pub fn new(
        shape: ACDShape, innovation: ACDInnovation, options: ACDOptions, n: usize,
    ) -> ACDModel {
        let p = shape.p;
        let q = shape.q;
        let alpha_buf = RefCell::new(Array1::zeros(q));
        let beta_buf = RefCell::new(Array1::zeros(p));
        let psi_buf = RefCell::new(Array1::zeros(n + p));
        let dur_buf = RefCell::new(Array1::zeros(q));
        ACDModel { shape, innovation, options, alpha_buf, beta_buf, psi_buf, dur_buf }
    }
}

impl LogLikelihood for ACDModel {
    type Data = ACDData;

    fn value(&self, theta: &Theta, data: &Self::Data) -> OptResult<f64> {
        let mut workspace_alpha = self.alpha_buf.borrow_mut();
        let mut workspace_beta = self.beta_buf.borrow_mut();
        let mut workspace =
            WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &self.shape)?;
        workspace.update_workspace(theta.view(), &self.shape)?;
        Ok(likelihood_driver(&self, &workspace, &data)?)
    }
    fn check(&self, theta: &Theta, _data: &Self::Data) -> OptResult<()> {
        validate_theta(theta.view(), self.shape.p, self.shape.q)?;
        Ok(())
    }
    fn grad(&self, theta: &Theta, data: &Self::Data) -> OptResult<Grad> {
        match self.innovation {
            ACDInnovation::Exponential => {
                todo!()
            }
            _ => Err(crate::optimization::errors::OptError::GradientNotImplemented),
        }
    }
}
