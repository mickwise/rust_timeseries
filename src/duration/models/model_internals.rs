//! ACD(p, q) model internals — workspace wiring, observation traversal, and score construction.
//!
//! Purpose
//! -------
//! Provide low-level helpers around [`ACDModel`] for in-sample inference:
//! wiring a reusable [`WorkSpace`] over shared scratch buffers, traversing the
//! ψ–recursion in lockstep with the observed durations, and assembling a
//! per-observation score matrix in optimizer-space θ.
//!
//! Key behaviors
//! -------------
//! - Initialize a [`WorkSpace`] over the model’s [`ACDScratch`] buffers via
//!   [`with_workspace`], updating `(ω, α, β, slack)` from a θ vector.
//! - Traverse effective observations (those used in the likelihood) with
//!   [`walk_observations`], feeding each `(∂ψ_t/∂θ, x_t, ψ_t)` into a user
//!   closure.
//! - Construct the full `n_eff × (1 + p + q)` score matrix via
//!   [`calculate_scores`], mapping ∂ℓ/∂ψ into θ-space.
//! - Expose small utilities [`extract_theta`], [`start_idx`], and [`n_eff`]
//!   that keep indexing and fitted-parameter access centralized.
//!
//! Invariants & assumptions
//! ------------------------
//! - Duration data in [`ACDData`] are already validated (finite, > 0, non-empty)
//!   and carry an in-sample offset `t0` when applicable.
//! - Scratch buffers in [`ACDScratch`] are shape-consistent with the model
//!   orders `(p, q)` and sample length `n`:
//!   - `psi_buf.len() == n + p`,
//!   - `deriv_buf.shape() == (n + p, 1 + p + q)`.
//! - The ψ–recursion and its derivatives have already been seeded correctly
//!   by [`compute_psi`] and [`compute_derivative`] before observation traversal.
//! - All routines operate on a fitted model: [`extract_theta`] assumes
//!   `model.results.is_some()` on the happy path.
//!
//! Conventions
//! -----------
//! - Time indexing in scratch buffers is 0-based. Pre-sample ψ-lags occupy the
//!   first `p` entries/rows; in-sample ψ and ∂ψ/∂θ live in indices `p..p + n`.
//! - The *effective sample size* `n_eff` is defined as the number of durations
//!   used in the likelihood: `n_eff = data.len() - t0`, independent of `p`.
//!   Lag burn-in is encoded purely in buffer indexing via [`start_idx`].
//! - Score rows are arranged as `θ = (ω, α₁..α_q, β₁..β_p)`, matching the
//!   parameter layout in [`WorkSpace::update`].
//! - Functions return [`ACDResult`]; invalid configurations / domain violations
//!   are surfaced as errors rather than panics.
//!
//! Downstream usage
//! ----------------
//! - Use [`calculate_scores`] inside standard-error or sandwich-variance
//!   calculations whenever per-observation gradients are needed.
//! - Use [`walk_observations`] to build other statistics that require traversing
//!   `(∂ψ_t/∂θ, x_t, ψ_t)` in lockstep (e.g., influence functions, diagnostics).
//! - Treat [`with_workspace`] as the canonical way to bind a θ vector into a
//!   [`WorkSpace`] and run any observation-level loop on top of ψ recursion.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module focus on:
//!   - alignment between `n_eff`, `start_idx`, and `likelihood_driver`
//!     (same number of observations and same pairing of `(x_t, ψ_t)`),
//!   - correct error behavior when `model.results` is `None` in
//!     [`extract_theta`].
//! - Integration tests in [`duration::models::acd::tests`] already exercise
//!   ψ–recursion, likelihood, and gradient correctness.
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

/// ObsEntry — view of a single effective ACD observation.
///
/// Purpose
/// -------
/// Represent the bundle of information passed into step closures during
/// observation traversal: the effective index, ψ-derivative row, observed
/// duration, and ψ value for a single time point.
///
/// Key behaviors
/// -------------
/// - Holds a borrowed `ArrayView1` of the ∂ψ_t/∂θ row without allocating or
///   copying data out of the model’s scratch buffers.
/// - Couples the derivative row with the corresponding duration and ψ_t so
///   closures can construct score contributions without extra lookups.
///
/// Parameters
/// ----------
/// - `idx`: `usize`
///   Effective sample index (0-based) relative to the first usable
///   observation after lag and `t0` burn-in.
/// - `deriv_row`: `ArrayView1<'a, f64>`
///   View into the derivative row ∂ψ_t/∂θ for this observation; length must
///   match the θ dimension `1 + p + q`.
/// - `data_point`: `f64`
///   Observed duration x_t at this effective index; strictly positive.
/// - `psi`: `f64`
///   Conditional expectation ψ_t (model-implied mean duration) for the same
///   time index.
///
/// Fields
/// ------
/// - `idx`: `usize`
///   Effective index into the score matrix / effective sample.
/// - `deriv_row`: `ArrayView1<'a, f64>`
///   Borrowed view of the derivative row in the workspace’s derivative buffer.
/// - `data_point`: `f64`
///   Raw duration value used in the innovation log-likelihood.
/// - `psi`: `f64`
///   Conditional mean duration associated with this observation.
///
/// Invariants
/// ----------
/// - `idx` is strictly less than the effective sample size `n_eff`.
/// - `deriv_row.len()` equals the θ dimension (1 + p + q) of the model.
/// - `data_point` and `psi` correspond to the same underlying time index.
/// - All values are finite; durations and ψ_t are strictly positive.
///
/// Performance
/// -----------
/// - Zero heap allocation; only borrows from existing scratch buffers.
/// - Copying `ObsEntry` is cheap; the heavy data is behind the borrowed view.
///
/// Notes
/// -----
/// - `ObsEntry` is primarily an internal convenience type used by
///   `walk_observations`; callers typically work with it via closures rather
///   than constructing instances directly.
#[derive(Debug, Clone, Copy)]
pub struct ObsEntry<'a> {
    pub idx: usize,
    pub deriv_row: ArrayView1<'a, f64>,
    pub data_point: f64,
    pub psi: f64,
}

/// Initialize an ACD `WorkSpace` and run a user closure inside it.
///
/// Parameters
/// ----------
/// - `model`: `&ACDModel`
///   Fitted ACD model providing shape (p, q), innovation family, and scratch
///   buffers (`ACDScratch`) compatible with the given data.
/// - `data`: `&ACDData`
///   Observed duration series used to drive the ψ recursion and derivative
///   computation. Durations are strictly positive, finite, and ordered.
/// - `loop_closure`: `F`
///   User closure that receives a fully prepared `WorkSpace` (updated for
///   θ̂, with ψ and ∂ψ/∂θ computed) and may inspect its arrays or
///   derived quantities. The closure must not attempt to re-borrow the
///   same scratch buffers mutably while the workspace is alive.
/// - `theta_hat`: `ArrayView1<f64>`
///   Parameter vector θ̂ used to update the workspace. The layout is
///   `(ω, α₁…α_p, β₁…β_q)` and its length must equal `1 + p + q`.
///
/// Returns
/// -------
/// `ACDResult<()>`
///   Returns `Ok(())` if workspace construction, update, ψ recursion,
///   derivative computation, and the user closure all complete successfully.
///   Propagates errors otherwise.
///
/// Errors
/// ------
/// - `ACDError`
///   Returned if `WorkSpace::new`, `workspace.update`, `compute_psi`,
///   `compute_derivative`, or the user closure fail due to inconsistent
///   shapes, invalid inputs, or other model errors.
///
/// Panics
/// ------
/// - May panic at runtime if the model’s scratch buffers are already borrowed
///   in a conflicting way when this function attempts to borrow them, due to
///   `RefCell` dynamic borrow checks.
///
/// Safety
/// ------
/// - This is a safe API; it does not use `unsafe`. Callers must ensure that
///   the same `ACDModel` is not used concurrently in ways that violate its
///   single-owner scratch buffer assumptions.
///
/// Notes
/// -----
/// - Centralizes workspace setup so callers that just need a single pass
///   over data with a given θ̂ do not have to manually manage ψ and
///   derivative recursions.
/// - Performs no heap allocations beyond those already held in the model’s
///   scratch buffers; all heavy arrays are reused.
///
/// Examples
/// --------
/// - Typical usage is to allocate an accumulator, call `with_workspace` with
///   a closure that reads ψ / derivative arrays from the workspace and
///   updates the accumulator, and then inspect the accumulator on return.
pub fn with_workspace<F: FnMut(&WorkSpace) -> ACDResult<()>>(
    model: &ACDModel, data: &ACDData, mut loop_closure: F, theta_hat: ArrayView1<f64>,
) -> ACDResult<()> {
    let mut workspace_alpha = model.scratch_bufs.alpha_buf.borrow_mut();
    let mut workspace_beta = model.scratch_bufs.beta_buf.borrow_mut();
    let mut workspace =
        WorkSpace::new(workspace_alpha.view_mut(), workspace_beta.view_mut(), &model.shape)?;
    workspace.update(theta_hat.view())?;
    compute_psi(&workspace, data, model);
    compute_derivative(&workspace, data, model);
    loop_closure(&workspace)
}

/// Construct the per-observation score matrix in θ-space for an ACD model.
///
/// Parameters
/// ----------
/// - `model`: `&ACDModel`
///   Fitted ACD model whose `results` contain θ̂ and whose `innovation`
///   implements a one-dimensional log-likelihood gradient in ψ.
/// - `data`: `&ACDData`
///   Observed duration series aligned with the model fit. Must have enough
///   observations to support the specified orders p, q after any `t0`
///   offset and lag burn-in.
///
/// Returns
/// -------
/// `ACDResult<Array2<f64>>`
///   - `Ok(scores)` where `scores` has shape
///     `(n_eff, 1 + p + q)` with `n_eff = n - t0` matching the number of
///     terms in the log-likelihood.
///   - `Err(ACDError)` if ψ recursion, derivative computation, or the
///     innovation gradient fails.
///
/// Errors
/// ------
/// - `ACDError::ModelNotFitted`
///   Returned if the model does not contain fit results and θ̂ cannot be
///   extracted.
/// - `ACDError`
///   Propagated from ψ / derivative recursions, workspace initialization, or
///   the innovation gradient evaluation if any of these fail.
///
/// Panics
/// ------
/// - May panic if internal scratch buffer borrowing is violated (for example,
///   if the same `ACDModel` is used concurrently from multiple threads) or
///   if the effective index exceeds the number of allocated rows due to
///   inconsistent buffer sizing.
///
/// Safety
/// ------
/// - Safe API with no `unsafe` blocks. Callers must ensure that the model,
///   data, and θ̂ are mutually consistent in shape, length, and `t0`
///   offset.
///
/// Notes
/// -----
/// - The computation proceeds in two stages:
///   1. Traverse effective observations with `walk_observations`, using the
///      innovation’s score function to accumulate ∂ℓ/∂ψ into each row.
///   2. Map these ψ-space gradients into θ-space using `safe_softmax_deriv`
///      for the (α, β) simplex parameterization and `safe_logistic` for ω.
/// - Allocates exactly one dense `Array2` for the scores; all other
///   intermediate arrays come from `ACDScratch`.
/// - The number of rows in the returned matrix is determined solely by the
///   in-sample window (`t0`), not by the lag order `p`. Lag burn-in is encoded
///   in the scratch buffer indices, not by shortening the sample.
///
/// Examples
/// --------
/// - After fitting an `ACDModel` to an `ACDData` series, call
///   `calculate_scores(&model, &data)` to obtain per-observation scores for
///   sandwich variance estimates or diagnostic plots.
pub fn calculate_scores(model: &ACDModel, data: &ACDData) -> ACDResult<Array2<f64>> {
    let p = model.shape.p;
    let q = model.shape.q;
    let innov = &model.innovation;
    let theta_hat = extract_theta(model)?;
    let n_eff = n_eff(data);

    // Closure to compute scores for each observation
    let step_write_score_row = |entry: ObsEntry<'_>, state: &mut Array2<f64>| -> ACDResult<()> {
        let mut current_row = state.row_mut(entry.idx);
        let innov_grad = innov.one_d_loglik_grad(entry.data_point, entry.psi)?;
        current_row.scaled_add(innov_grad, &entry.deriv_row);
        Ok(())
    };

    let finish_map_score_rows = |workspace: &WorkSpace, state: &mut Array2<f64>| -> ACDResult<()> {
        state.rows_mut().into_iter().for_each(|mut row| {
            safe_softmax_deriv(&workspace.alpha, &workspace.beta, &mut row.slice_mut(s![1..]));
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

/// Traverse effective ACD observations with user-defined step and finish closures.
///
/// Parameters
/// ----------
/// - `model`: `&ACDModel`
///   ACD model providing shape (p, q), innovation family, and scratch buffers
///   for ψ and derivative recursions.
/// - `data`: `&ACDData`
///   Observed duration series to traverse. Must be consistent with the model
///   fit and contain enough observations after `t0`.
/// - `theta_hat`: `ArrayView1<'_, f64>`
///   Parameter vector θ̂ used to update the workspace before traversal.
///   Expected layout is `(ω, α₁…α_p, β₁…β_q)` with length `1 + p + q`.
/// - `state`: `&mut State`
///   Mutable caller-provided state threaded through each step and made
///   available in the finalizer. Typically an accumulator or output buffer.
/// - `step`: `Step`
///   Closure invoked for each effective observation. Receives an `ObsEntry`
///   with `(idx, deriv_row, data_point, psi)` and a mutable reference to
///   `state`, and may update state or return an error.
/// - `finish`: `Finish`
///   Closure called once after all observations have been processed. Receives
///   the final `WorkSpace` and mutable `state` and may perform any final
///   mapping, normalization, or projection.
///
/// Returns
/// -------
/// `ACDResult<()>`
///   Returns `Ok(())` if workspace preparation, ψ / derivative recursions,
///   all `step` invocations, and the `finish` closure complete successfully;
///   otherwise propagates the first error encountered.
///
/// Errors
/// ------
/// - `ACDError`
///   Returned if workspace initialization or the ψ / derivative recursions
///   fail.
/// - Any error variant carried by `ACDResult` from the `step` or `finish`
///   closures if they signal failure.
///
/// Panics
/// ------
/// - May panic on `RefCell` dynamic borrow errors if other parts of the code
///   attempt to borrow the same scratch buffers while traversal is in
///   progress.
/// - May panic if indexing assumptions between `start_idx`, `n_eff`, and
///   scratch buffer sizes are violated due to inconsistent shapes.
///
/// Safety
/// ------
/// - Purely safe API. Callers must ensure that the `ACDModel`, `ACDData`, and
///   θ̂ correspond to the same fit and that no concurrent mutable access
///   to the same model instance occurs.
///
/// Notes
/// -----
/// - Traverses exactly the same observation window as [`likelihood_driver`]:
///   durations from `t0` (or 0 if `None`) up to the end of the series, paired
///   with ψ values from `psi_buf[start_idx..]` where
///   `start_idx = model.shape.p + data.t0.unwrap_or(0)`.
/// - Pre-sample ψ-lags and their derivative rows (indices `0..p` in scratch
///   buffers) are treated as internal initialization state and are *not*
///   exposed as [`ObsEntry`] values.
/// - Runs allocation-free by borrowing model scratch buffers; user state
///   `State` is owned by the caller.
///
/// Examples
/// --------
/// - Use `walk_observations` with a small accumulator struct as `State` to
///   build moment conditions or diagnostic summaries over an ACD fit without
///   allocating additional intermediate arrays.
pub fn walk_observations<State, Step, Finish>(
    model: &ACDModel, data: &ACDData, theta_hat: ArrayView1<'_, f64>, state: &mut State,
    mut step: Step, mut finish: Finish,
) -> ACDResult<()>
where
    Step: FnMut(ObsEntry<'_>, &mut State) -> ACDResult<()>,
    Finish: for<'w> FnMut(&WorkSpace<'w>, &mut State) -> ACDResult<()>,
{
    let loop_closure = |workspace: &WorkSpace| -> ACDResult<()> {
        let deriv_binding = model.scratch_bufs.deriv_buf.borrow();
        let start_idx = start_idx(model, data);
        let data_start_idx = data.t0.unwrap_or(0);
        let psi_derives = deriv_binding.slice(s![start_idx.., ..]);
        let psi_lags_binding = model.scratch_bufs.psi_buf.borrow();
        let psi_lags = psi_lags_binding.slice(s![start_idx..]);
        let data_slice = data.data.slice(s![data_start_idx..]);

        psi_derives
            .rows()
            .into_iter()
            .zip(data_slice.iter().copied())
            .zip(psi_lags.iter().copied())
            .map(|((row, x), psi)| (row, x, psi))
            .enumerate()
            .map(|(idx, (row, x, psi))| ObsEntry { idx, deriv_row: row, data_point: x, psi })
            .try_for_each(|elem| step(elem, state))?;
        finish(&workspace, state)?;
        Ok(())
    };
    with_workspace(model, data, loop_closure, theta_hat.view())?;
    Ok(())
}

/// Compute the index of the first usable ψ / derivative entry.
///
/// Parameters
/// ----------
/// - `model`: `&ACDModel`
///   Provides the ACD order p, which determines how many leading positions in
///   the ψ and derivative buffers are reserved for lag burn-in.
/// - `data`: `&ACDData`
///   Observed duration series, optionally carrying a `t0` offset indicating
///   that the first `t0` raw observations are discarded from the effective
///   sample.
///
/// Returns
/// -------
/// `usize`
///   Starting index into the ψ / derivative buffers for effective
///   observations, equal to `p + t0` where `t0` defaults to 0.
///
/// Errors
/// ------
/// - Never returns an error.
///
/// Panics
/// ------
/// - Never panics under normal usage, assuming `p + t0` does not exceed the
///   length of the ψ / derivative buffers.
///
/// Safety
/// ------
/// - Purely arithmetic helper. Callers must ensure that the returned index
///   is valid for all arrays that will be sliced starting at this position.
///
/// Notes
/// -----
/// - Intended to centralize the definition of “first effective index” so
///   traversal and likelihood routines remain consistent with each other.
pub fn start_idx(model: &ACDModel, data: &ACDData) -> usize {
    model.shape.p + data.t0.unwrap_or(0)
}

/// Effective sample size used in likelihood and score calculations.
///
/// Purpose
/// -------
/// Compute the number of in-sample observations that contribute to the
/// log-likelihood (and hence to the score matrix), taking the optional
/// burn-in index `t0` into account.
///
/// Parameters
/// ----------
/// - `data`: `&ACDData`
///   Duration series and metadata. `data.t0` denotes the index of the first
///   in-sample observation; if `None`, the entire series is in-sample.
///
/// Returns
/// -------
/// `usize`
///   `n_eff = data.data.len() - data.t0.unwrap_or(0)`, i.e. the number of
///   terms in the likelihood sum and the number of rows expected in any
///   per-observation score matrix.
///
/// Errors
/// ------
/// - Never returns an error; this is a pure helper over validated `ACDData`.
///
/// Panics
/// ------
/// - Never panics. Input invariants (`t0 < len`) are already enforced by
///   [`ACDData::new`].
///
/// Notes
/// -----
/// - The ACD lag order `p` does *not* reduce `n_eff`; lag burn-in is handled
///   by indexing into `psi_buf` and `deriv_buf` via [`start_idx`].
/// - This definition matches the effective sample used by
///   [`likelihood_driver`], so that:
///   - `n_eff == duration_series.len()` there, and
///   - rows of the score matrix align 1:1 with likelihood contributions.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::array;
/// # use rust_timeseries::duration::core::data::{ACDData, ACDMeta};
/// # use rust_timeseries::duration::core::units::ACDUnit;
/// # use rust_timeseries::duration::models::acd::internals::n_eff;
/// let data = array![1.0, 2.0, 3.0, 4.0];
/// let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
/// let acd = ACDData::new(data, Some(1), meta).unwrap();
/// assert_eq!(n_eff(&acd), 3);  // observations at indices 1, 2, 3
/// ```
pub fn n_eff(data: &ACDData) -> usize {
    data.data.len() - data.t0.unwrap_or(0)
}

/// Extract the fitted parameter vector θ̂ from an `ACDModel`.
///
/// Parameters
/// ----------
/// - `model`: `&ACDModel`
///   ACD model that may or may not have been fitted. On success, its
///   `results` field must contain a θ̂ vector whose length matches the
///   model’s θ dimension (1 + p + q).
///
/// Returns
/// -------
/// `ACDResult<&Array1<f64>>`
///   On success, returns a reference to the fitted parameter vector θ̂ owned
///   by the model’s `results`. The reference is valid as long as `model` is
///   borrowed for the duration of the call chain.
///
/// Errors
/// ------
/// - `ACDError::ModelNotFitted`
///   Returned if `model.results` is `None`, indicating that no fit has been
///   performed yet.
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - Purely safe API. The returned reference remains valid as long as the
///   borrowed `ACDModel` is not mutably modified in ways that invalidate
///   its `results` field.
///
/// Notes
/// -----
/// - Centralizing θ̂ extraction here keeps error handling consistent and
///   avoids repeating `ModelNotFitted` checks across multiple modules that
///   need access to the fitted parameters.
pub fn extract_theta(model: &ACDModel) -> ACDResult<&Array1<f64>> {
    match model.results {
        Some(ref outcome) => Ok(&outcome.theta_hat),
        None => Err(ACDError::ModelNotFitted),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::core::data::{ACDData, ACDMeta};
    use crate::duration::core::guards::PsiGuards;
    use crate::duration::core::init::Init;
    use crate::duration::core::innovations::ACDInnovation;
    use crate::duration::core::options::ACDOptions;
    use crate::duration::core::params::ACDScratch;
    use crate::duration::core::shape::ACDShape;
    use crate::duration::core::units::ACDUnit;
    use crate::duration::models::acd::ACDModel;
    use crate::optimization::loglik_optimizer::traits::MLEOptions;
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Alignment between `n_eff`, `start_idx`, and `likelihood_driver`.
    // - Construction and error behavior of utilities in this internals module:
    //   `n_eff`, `start_idx`, `with_workspace`, and `extract_theta`.
    // - Correct wiring of `with_workspace` and positive, finite in-sample ψ.
    //
    // They intentionally DO NOT cover:
    // - Numerical correctness of ψ-recursion or derivatives (covered in
    //   duration::core::psi tests).
    // - Optimizer behavior or full ACDModel fitting (covered by acd.rs tests).
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // `n_eff` returns the number of observations used in the likelihood, i.e.
    // `len(data) - t0`, independent of the lag order `p`.
    //
    // Given
    // -----
    // - An `ACDData` with and without a burn-in index `t0`.
    //
    // Expect
    // ------
    // - `n_eff(&data)` equals `data.data.len()` when `t0 == None`.
    // - `n_eff(&data)` equals `data.data.len() - t0` when `t0 == Some(t0)`.
    fn n_eff_respects_t0_and_ignores_p() {
        // Arrange
        let durations = array![1.0, 2.0, 3.0, 4.0];
        let meta = ACDMeta::new(ACDUnit::Seconds, None, false);

        let data_full = ACDData::new(durations.clone(), None, meta.clone()).unwrap();
        let data_burn = ACDData::new(durations.clone(), Some(2), meta).unwrap();

        // Act
        let n_eff_full = n_eff(&data_full);
        let n_eff_burn = n_eff(&data_burn);

        // Assert
        assert_eq!(n_eff_full, 4);
        assert_eq!(n_eff_burn, 2); // indices 2 and 3
    }

    #[test]
    // Purpose
    // -------
    // `start_idx` correctly accounts for both the ACD lag order `p` and the
    // burn-in index `t0`.
    //
    // Given
    // -----
    // - A shape with `p > 0`.
    // - An `ACDData` with and without a non-zero `t0`.
    //
    // Expect
    // ------
    // - `start_idx(model, data)` equals `p` when `t0 == None`.
    // - `start_idx(model, data)` equals `p + t0` when `t0 == Some(t0)`.
    fn start_idx_matches_p_and_t0() {
        // Arrange
        let durations = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
        let shape = ACDShape { p: 2, q: 1 };
        let n = durations.len();

        // minimal, "reasonable" options; mirror patterns from other tests
        let init = Init::uncond_mean(); // or whatever constructor you already use
        let mle_opts = MLEOptions::default(); // if available; otherwise construct explicitly
        let psi_guards = PsiGuards::new((1e-6, 1e6)).unwrap();
        let options = ACDOptions::new(init, mle_opts, psi_guards);

        let scratch = ACDScratch::new(n, shape.p, shape.q);
        let innovation = ACDInnovation::Exponential;
        let model = ACDModel {
            shape,
            innovation,
            options,
            scratch_bufs: scratch,
            results: None,
            fitted_params: None,
            forecast: None,
        };

        let data_full = ACDData::new(durations.clone(), None, meta.clone()).unwrap();
        let data_burn = ACDData::new(durations.clone(), Some(1), meta).unwrap();

        // Act
        let idx_full = start_idx(&model, &data_full);
        let idx_burn = start_idx(&model, &data_burn);

        // Assert
        assert_eq!(idx_full, 2); // p + t0(None => 0)
        assert_eq!(idx_burn, 3); // p + t0(1)
    }

    #[test]
    // Purpose
    // -------
    // `extract_theta` returns an error when the model has not been fitted and
    // `model.results` is `None`.
    //
    // Given
    // -----
    // - An `ACDModel` instance with `results == None`.
    //
    // Expect
    // ------
    // - `extract_theta(&model)` returns `Err(ACDError::ModelNotFitted)`.
    fn extract_theta_errors_on_unfitted_model() {
        // Arrange: same minimal model setup as above, but we don't care about data here.
        let shape = ACDShape { p: 1, q: 1 };
        let n = 10;

        let init = Init::uncond_mean();
        let mle_opts = MLEOptions::default();
        let psi_guards = PsiGuards::new((1e-6, 1e6)).unwrap();
        let options = ACDOptions::new(init, mle_opts, psi_guards);

        let scratch = ACDScratch::new(n, shape.p, shape.q);
        let innovation = ACDInnovation::Exponential;
        let model = ACDModel {
            shape,
            innovation,
            options,
            scratch_bufs: scratch,
            results: None,
            fitted_params: None,
            forecast: None,
        };

        // Act
        let result = extract_theta(&model);

        // Assert
        assert!(matches!(result, Err(ACDError::ModelNotFitted)));
    }

    #[test]
    // Purpose
    // -------
    // `with_workspace` wires the model scratch, ψ recursion, and derivatives
    // without error and produces positive, finite in-sample ψ values.
    //
    // Given
    // -----
    // - A small ACD model with p = 1, q = 1 and a short duration series.
    // - A θ̂ vector of the correct length (1 + p + q).
    //
    // Expect
    // ------
    // - `with_workspace` returns `Ok(())`.
    // - The in-sample ψ segment in `psi_buf` has length `n` and contains
    //   strictly positive, finite values.
    fn with_workspace_runs_and_populates_in_sample_psi() {
        // Arrange
        let durations = array![1.0, 2.0, 3.0, 4.0];
        let n = durations.len();
        let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
        let data = ACDData::new(durations, None, meta).unwrap();

        let shape = ACDShape { p: 1, q: 1 };
        let init = Init::uncond_mean();
        let mle_opts = MLEOptions::default();
        let psi_guards = PsiGuards::new((1e-6, 1e6)).unwrap();
        let options = ACDOptions::new(init, mle_opts, psi_guards);

        let scratch = ACDScratch::new(n, shape.p, shape.q);
        let innovation = ACDInnovation::Exponential;
        let model = ACDModel {
            shape,
            innovation,
            options,
            scratch_bufs: scratch,
            results: None,
            fitted_params: None,
            forecast: None,
        };

        // θ̂ layout: (ω, α₁, β₁)
        let theta_hat = array![0.0, 0.1, -0.2];

        // Act
        let result = with_workspace(
            &model,
            &data,
            |workspace| {
                // Assert inside closure: shapes and ψ positivity.
                let p = model.shape.p;
                let psi_buf = model.scratch_bufs.psi_buf.borrow();
                let psi_slice = psi_buf.slice(s![p..p + n]);

                // Alpha / beta shapes should match p, q.
                assert_eq!(workspace.alpha.len(), model.shape.p);
                assert_eq!(workspace.beta.len(), model.shape.q);

                // In-sample ψ segment should be finite and strictly positive.
                assert_eq!(psi_slice.len(), n);
                assert!(psi_slice.iter().all(|v| v.is_finite() && *v > 0.0));

                Ok(())
            },
            theta_hat.view(),
        );

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `walk_observations` traverses exactly `n_eff` observations and keeps
    // `(idx, data_point, psi)` aligned with the in-sample window.
    //
    // Given
    // -----
    // - A small ACD model with p = 1, q = 1 and a short duration series.
    // - A burn-in index `t0` on the data.
    // - A θ̂ vector of the correct length (1 + p + q).
    //
    // Expect
    // ------
    // - The step closure is called exactly `n_eff(&data)` times.
    // - `ObsEntry.idx` runs from 0 to `n_eff - 1`.
    // - The collected `data_point` values equal the in-sample slice
    //   `data.data[t0..]`.
    fn walk_observations_traverses_n_eff_in_lockstep() {
        // Arrange
        let durations = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = durations.len();
        let t0 = 2;
        let meta = ACDMeta::new(ACDUnit::Seconds, None, false);
        let data = ACDData::new(durations.clone(), Some(t0), meta).unwrap();

        let shape = ACDShape { p: 1, q: 1 };
        let init = Init::uncond_mean();
        let mle_opts = MLEOptions::default();
        let psi_guards = PsiGuards::new((1e-6, 1e6)).unwrap();
        let options = ACDOptions::new(init, mle_opts, psi_guards);

        let scratch = ACDScratch::new(n, shape.p, shape.q);
        let innovation = ACDInnovation::Exponential;
        let model = ACDModel {
            shape,
            innovation,
            options,
            scratch_bufs: scratch,
            results: None,
            fitted_params: None,
            forecast: None,
        };

        // θ̂ layout: (ω, α₁, β₁)
        let theta_hat = array![0.0, 0.1, -0.2];

        #[derive(Default)]
        struct State {
            count: usize,
            xs: Vec<f64>,
            psis: Vec<f64>,
        }

        let mut state = State::default();

        // Act
        let result = walk_observations(
            &model,
            &data,
            theta_hat.view(),
            &mut state,
            |entry, st| {
                // idx should be contiguous 0..n_eff-1
                assert_eq!(entry.idx, st.count);
                st.count += 1;
                st.xs.push(entry.data_point);
                st.psis.push(entry.psi);
                Ok(())
            },
            |_workspace, _st| {
                // No extra work in finish for this test.
                Ok(())
            },
        );

        // Assert
        assert!(result.is_ok());

        let expected_n_eff = n_eff(&data);
        assert_eq!(state.count, expected_n_eff);
        assert_eq!(state.xs.len(), expected_n_eff);
        assert_eq!(state.psis.len(), expected_n_eff);

        let expected_xs: Vec<f64> = durations.slice(s![t0..]).to_vec();
        assert_eq!(state.xs, expected_xs);
    }
}
