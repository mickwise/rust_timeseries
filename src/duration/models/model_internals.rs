//! ACD(p, q) model internals: score construction and observation traversal.
//!
//! This module provides helper routines for inference in ACD models:
//! - [`with_workspace`] sets up a temporary [`WorkSpace`] and runs a user closure.
//! - [`walk_observations`] iterates through the ψ recursion, feeding each
//!   observation into a `step` closure and finalizing with a `finish` closure.
//! - [`calculate_scores`] builds the full n × (1 + p + q) score matrix
//!   (per-observation gradients in θ-space).
//! - Utility functions [`extract_theta`], [`start_idx`], and [`n_eff`] give
//!   access to fitted parameters and effective sample sizes.
//!
//! All routines are allocation-free except for the final score matrix. They
//! reuse buffers from the model’s [`ACDScratch`] and work seamlessly with the
//! analytic gradient implementation in [`ACDModel`].
use crate::{
    duration::{
        core::{
            data::ACDData,
            psi::{compute_derivative, compute_psi},
            workspace::WorkSpace,
        },
        errors::{ACDError, ACDResult},
        models::acd::ACDModel,
    },
    optimization::numerical_stability::transformations::{safe_logistic, safe_softmax_deriv},
};
use ndarray::{Array1, Array2, ArrayView1, s};

/// Observation entry passed to closures in [`walk_observations`].
///
/// Tuple fields:
/// - `usize`: index of the observation within the effective sample.
/// - `ArrayView1<f64>`: derivative row (∂ψ/∂θ) for this observation.
/// - `f64`: observed data point (duration).
/// - `f64`: ψ value (conditional expectation of duration).
pub type ObsEntry<'a> = (usize, ArrayView1<'a, f64>, f64, f64);

/// Initialize a [`WorkSpace`] and run a user closure inside it.
///
/// This function borrows the model’s scratch buffers, updates the workspace
/// with the given parameter vector, computes ψ recursion and derivatives,
/// and then calls the provided closure with the prepared workspace.
///
/// # Arguments
/// - `model`: fitted [`ACDModel`] providing scratch buffers and shape.
/// - `data`: observed duration series.
/// - `loop_closure`: user closure operating on the prepared [`WorkSpace`].
/// - `theta_hat`: parameter vector (θ̂) to use for workspace update.
///
/// # Returns
/// - `Ok(())` if the closure and ψ/derivative updates succeed.
/// - Propagates [`ACDError`] if workspace update or recursion fails.
pub fn with_workspace<F: FnMut(&WorkSpace) -> ACDResult<()>>(
    model: &ACDModel, data: &ACDData, mut loop_closure: F, theta_hat: ArrayView1<f64>,
) -> ACDResult<()> {
    let mut workspace_alpha = model.scratch_bufs.alpha_buf.borrow_mut();
    let mut workspace_beta = model.scratch_bufs.beta_buf.borrow_mut();
    let mut workspace =
        WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &model.shape)?;
    workspace.update_workspace(theta_hat.view(), &model.shape)?;
    compute_psi(&workspace, data, model);
    compute_derivative(&workspace, data, model);
    loop_closure(&workspace)
}

/// Construct the per-observation score matrix for the ACD model.
///
/// Each row corresponds to the gradient contribution of a single observation,
/// mapped into θ-space (length = 1 + p + q).
///
/// # Method
/// - Iterates over observations with [`walk_observations`].
/// - Accumulates ∂ℓ/∂ψ terms via the innovation’s score function.
/// - Maps into θ-space using `safe_logistic` (ω) and `safe_softmax_deriv` (α, β).
///
/// # Arguments
/// - `model`: fitted [`ACDModel`].
/// - `data`: observed duration series.
///
/// # Returns
/// - An `n_eff × (1 + p + q)` matrix of per-observation scores.
/// - Propagates errors from gradient evaluation or workspace recursion.
pub fn calculate_scores(model: &ACDModel, data: &ACDData) -> ACDResult<Array2<f64>> {
    let p = model.shape.p;
    let q = model.shape.q;
    let innov = &model.innovation;
    let theta_hat = extract_theta(model)?;
    let n_eff = n_eff(data);

    // Closure to compute scores for each observation
    let step_write_score_row = |(idx, deriv_row, data_point, psi): ObsEntry<'_>,
                                state: &mut Array2<f64>|
     -> ACDResult<()> {
        let mut current_row = state.row_mut(idx);
        let innov_grad = innov.one_d_loglik_grad(data_point, psi)?;
        current_row.scaled_add(innov_grad, &deriv_row);
        Ok(())
    };

    let finish_map_score_rows = |workspace: &WorkSpace, state: &mut Array2<f64>| -> ACDResult<()> {
        state.rows_mut().into_iter().for_each(|mut row| {
            safe_softmax_deriv(
                &workspace.alpha,
                &workspace.beta,
                &mut row.slice_mut(s![1..]),
                p,
                q,
            );
            row[0] *= safe_logistic(theta_hat[0]);
        });
        Ok(())
    };

    let mut state = Array2::zeros((n_eff, 1 + p + q));

    walk_observations(
        model,
        data,
        theta_hat.view(),
        &mut state,
        step_write_score_row,
        finish_map_score_rows,
    )?;
    Ok(state)
}

/// Iterate over observations with user-defined step and finish closures.
///
/// For each effective observation, provides an [`ObsEntry`] tuple and allows
/// the user to update arbitrary state. After the loop, the `finish` closure
/// is called once with the final [`WorkSpace`] and state.
///
/// # Arguments
/// - `model`: fitted [`ACDModel`].
/// - `data`: observed duration series.
/// - `theta_hat`: parameter vector used to update the workspace.
/// - `state`: mutable user state updated across iterations.
/// - `step`: closure `(ObsEntry, &mut State) -> ACDResult<()>` applied to each row.
/// - `finish`: closure `(&WorkSpace, &mut State) -> ACDResult<()>` applied once at end.
///
/// # Returns
/// - `Ok(())` on success.
/// - Propagates [`ACDError`] or closure errors.
///
/// # Notes
/// - Traverses only the effective sample (after burn-in from p, q lags).
/// - Runs allocation-free by borrowing model scratch buffers.
pub fn walk_observations<State, Step, Finish>(
    model: &ACDModel, data: &ACDData, theta_hat: ArrayView1<'_, f64>, state: &mut State,
    mut step: Step, mut finish: Finish,
) -> ACDResult<()>
where
    Step: for<'w> FnMut(ObsEntry<'_>, &mut State) -> ACDResult<()>,
    Finish: for<'w> FnMut(&WorkSpace<'w>, &mut State) -> ACDResult<()>,
{
    let loop_closure = |workspace: &WorkSpace| -> ACDResult<()> {
        let deriv_binding = model.scratch_bufs.deriv_buf.borrow();
        let start_idx = start_idx(model, data);
        let psi_derives = deriv_binding.slice(s![start_idx.., ..]);
        let psi_lags_binding: std::cell::Ref<
            '_,
            ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
        > = model.scratch_bufs.psi_buf.borrow();
        let psi_lags = psi_lags_binding.slice(s![start_idx..]);

        psi_derives
            .rows()
            .into_iter()
            .zip(data.data.slice(s![start_idx..]).iter().copied())
            .zip(psi_lags.iter().copied())
            .map(|((row, x), psi)| (row, x, psi))
            .enumerate()
            .map(|(idx, (row, x, psi))| (idx, row, x, psi))
            .try_for_each(|elem| step(elem, state))?;
        finish(&workspace, state)?;
        Ok(())
    };
    with_workspace(model, data, loop_closure, theta_hat.view())?;
    Ok(())
}

/// Compute the index of the first usable observation.
///
/// Accounts for the model order (p) and optional initial offset `t0`.
///
/// # Arguments
/// - `model`: ACD model (for p).
/// - `data`: observed series (for optional t0).
///
/// # Returns
/// - Starting index into `data` for effective observations.
pub fn start_idx(model: &ACDModel, data: &ACDData) -> usize {
    model.shape.p + data.t0.unwrap_or(0)
}

/// Effective sample size after discarding initial offset.
///
/// # Arguments
/// - `data`: observed duration series.
///
/// # Returns
/// - Number of usable observations = total length − t0 (if any).
pub fn n_eff(data: &ACDData) -> usize {
    data.data.len() - data.t0.unwrap_or(0)
}

/// Extract the fitted parameter vector θ̂ from an [`ACDModel`].
///
/// # Arguments
/// - `model`: fitted model with `results` present.
///
/// # Returns
/// - Reference to the fitted parameter vector (owned by `results`).
///
/// # Errors
/// - Returns [`ACDError::ModelNotFitted`] if no fit results are available.
pub fn extract_theta(model: &ACDModel) -> ACDResult<&Array1<f64>> {
    match model.results {
        Some(ref outcome) => Ok(&outcome.theta_hat),
        None => Err(ACDError::ModelNotFitted),
    }
}
