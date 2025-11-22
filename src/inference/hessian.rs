//! inference::hessian — Hessian-based covariance matrix utilities.
//!
//! Purpose
//! -------
//! Provide a thin wrapper around finite-difference Hessians that converts
//! them into numerically stable covariance matrix estimates.
//! This module handles conversion between `ndarray` and `nalgebra` types
//! and supports both classical and robust (sandwich) SEs built from the
//! observed information and a score covariance matrix.
//!
//! Key behaviors
//! -------------
//! - Call [`compute_hessian`] on an average log-likelihood gradient to
//!   obtain the observed information matrix `J(θ̂)`.
//! - Copy the resulting `ndarray` Hessian into a `nalgebra::DMatrix`
//!   (`fill_dmatrix`) for eigen-based linear algebra.
//! - Compute classical standard errors from the Moore–Penrose
//!   pseudoinverse of `J(θ̂)`.
//! - Compute robust (sandwich) standard errors when supplied with a
//!   score covariance matrix `S` (IID OPG or HAC) on the average-score
//!   scale.
//! - Expose [`calc_covariance`] to return the full classical or robust
//!   covariance matrix `Var(θ̂) = J⁺ S J⁺` (with `S = J` in the classical case).
//!
//! Invariants & assumptions
//! ------------------------
//! - [`compute_hessian`] returns a finite, square `n×n` matrix with
//!   `n = θ̂.len()`. Symmetry is already enforced upstream via
//!   `symmetrize_hess`; this module does **not** re-symmetrize.
//! - The matrix passed into [`solve_for_se`] and
//!   [`solve_for_se_robust`] is treated as symmetric for the purposes
//!   of `symmetric_eigen`.
//! - When provided, `scores` is an `n×n` symmetric score covariance
//!   matrix for the *average* score, with `n = θ̂.len()`.
//! - Eigenvalues with magnitude at most [`EIGEN_EPS`] are treated as
//!   numerically nonpositive and ignored when constructing pseudoinverse
//!   directions.
//!
//! Conventions
//! -----------
//! - All Hessians are on the **average log-likelihood** scale (not the
//!   sum), so the resulting variances/SEs/covariances correspond to that scaling.
//! - The covariance matrices returned are in the **unconstrained**
//!   parameter space (θ-space) used by the optimizer, **not** in the
//!   model-parameters space `(ω, α, β, slack - space)`.
//!
//! Downstream usage
//! ----------------
//! - Model layers (e.g., ACD models) call [`calc_standard_errors`] after
//!   fitting to obtain classical or robust covariance matrix at the MLE.
//! - [`solve_for_robust_cov_matrix`] use a score covariance matrix `S` produced by
//!   `inference::hac` (e.g., via `inference::hac::calculate_avg_scores_cov`)
//!   and threaded through model-side inference options.
//! - Helper routines [`fill_dmatrix`], [`solve_for_cov_matrix`] , and
//!   [`solve_for_robust_cov_matrix`] are internal utilities; library users
//!   should not need to invoke them directly.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module cover:
//!   - Correct copying of Hessians from `ndarray` into `DMatrix`
//!     without altering symmetry.
//!   - Agreement between classical covariance estimates and an
//!     analytic `J⁺` for simple quadratic objectives.
//!   - Robust covariance behavior when `scores` is chosen to inflate variances
//!     relative to the classical case.
//! - Integration tests at the model layer verify that:
//!   - SEs behave sensibly under reparameterizations.
//!   - Weak identification (near-zero eigenvalues) produces inflated
//!     SEs along those directions.
use crate::optimization::{
    errors::OptResult, loglik_optimizer::finite_diff::compute_hessian,
    numerical_stability::transformations::EIGEN_EPS,
};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// covariance_matrix — classical or HAC-robust covariance in θ-space.
///
/// Purpose
/// -------
/// Compute the full covariance matrix of the **unconstrained** parameter
/// vector θ used by the optimizer, evaluated at the MLE θ̂. By default it
/// returns *classical* covariance based on the observed information; with
/// [`HACOptions`] it returns *robust* (sandwich/HAC) covariance that
/// accounts for serial correlation and heteroskedasticity in the scores.
///
/// Parameters
/// ----------
/// - `data`: `&ACDData`
///   Observed duration series used both to re-evaluate the gradient
///   (for the finite-difference Hessian) and to compute per-observation
///   scores needed for HAC robustification.
/// - `hac_opts`: `Option<&HACOptions>`
///   - `None` for classical covariance based on the observed information
///     matrix `J(θ̂)`.
///   - `Some(opts)` for robust/HAC covariance using `opts.kernel`,
///     `opts.bandwidth`, `opts.center`, and any small-sample corrections.
///
/// Returns
/// -------
/// `ACDResult<Array2<f64>>`
///   On success, an `k×k` covariance matrix `Var(θ̂)` in **θ-space**, where
///   `k = 1 + p + q` is the dimension of the unconstrained optimizer vector.
///   On failure, returns an [`ACDError`] indicating what went wrong.
///
/// Errors
/// ------
/// - [`ACDError::ModelNotFitted`]
///   Returned if no θ̂ is available because [`ACDModel::fit`] has not been
///   called successfully.
/// - Other [`ACDError`] variants
///   Propagated from:
///   - [`ACDModel::grad`] (during Hessian construction),
///   - [`calculate_scores`] / [`calculate_avg_scores_cov`] (for HAC cases),
///   - or any error mapped from the underlying optimization / inference
///     machinery used by [`calc_covariance`].
///
/// Panics
/// ------
/// - Never panics on invalid user inputs; such conditions are surfaced as
///   [`ACDError`] values. Panics would indicate programmer errors
///   (e.g., inconsistent shapes) rather than user-facing failures.
///
/// Notes
/// -----
/// - The covariance is **strictly in θ-space**, i.e. for the unconstrained
///   optimizer parameters, _not_ for model-space parameters `(ω, α, β, slack)`.
///   To obtain covariance for model-space parameters, apply a delta-method
///   transform using the Jacobian of the softplus/softmax mapping.
/// - Classical covariance is the Moore–Penrose pseudoinverse `J⁺` of the
///   observed information matrix `J(θ̂)`, computed via symmetric
///   eigendecomposition with eigenvalue truncation controlled by
///   [`EIGEN_EPS`].
/// - Robust/HAC covariance applies a sandwich formula `Var(θ̂) = J⁺ S J⁺`,
///   where `S` is a HAC estimator of the covariance of the **average score**
///   constructed from per-observation scores and the kernel/bandwidth in
///   [`HACOptions`].
pub fn calc_covariance<F: Fn(&Array1<f64>) -> Array1<f64>>(
    f: &F, theta_hat: &Array1<f64>, scores: Option<&Array2<f64>>,
) -> OptResult<Array2<f64>> {
    let obs_info = compute_hessian(f, theta_hat)?;
    let mut obs_info_nalg = DMatrix::<f64>::zeros(obs_info.nrows(), obs_info.ncols());
    fill_dmatrix(&obs_info, &mut obs_info_nalg);
    match scores {
        Some(s) => Ok(solve_for_robust_cov_matrix(obs_info_nalg, s)),
        None => Ok(solve_for_cov_matrix(obs_info_nalg)),
    }
}
// ---- Helper methods ----

/// fill_dmatrix — copy an `ndarray` Hessian into a `nalgebra::DMatrix`.
///
/// Purpose
/// -------
/// Bridge between `ndarray` and `nalgebra` by copying a square observed
/// information matrix into a `DMatrix<f64>` using column-major writes.
/// This helper does **not** modify symmetry; it assumes that the input
/// has already been symmetrized upstream (e.g., by [`compute_hessian`]).
///
/// Parameters
/// ----------
/// - `obs_info`: `&Array2<f64>`
///   Square `n×n` observed information matrix in `ndarray` form. Expected
///   to be symmetric up to numerical precision.
/// - `obs_info_nalg`: `&mut DMatrix<f64>`
///   Preallocated `n×n` `DMatrix` that will receive the contents of
///   `obs_info`. Must have the same dimensions as `obs_info`.
///
/// Returns
/// -------
/// `()`
///   Mutates `obs_info_nalg` in place; no value is returned.
///
/// Errors
/// ------
/// - `None`  
///   This helper does not return a `Result`. Dimension mismatches are
///   considered programmer errors.
///
/// Panics
/// ------
/// - May panic if `obs_info` and `obs_info_nalg` have inconsistent shapes,
///   due to out-of-bounds indexing.
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that both matrices are
///   `n×n` with matching `n`.
///
/// Notes
/// -----
/// - The copy proceeds column by column, matching the internal storage of
///   `DMatrix` (column-major) and improving cache locality compared to a
///   row-major traversal.
/// - No symmetrization is performed here; any asymmetry present in
///   `obs_info` will be preserved in `obs_info_nalg`.
fn fill_dmatrix(obs_info: &Array2<f64>, obs_info_nalg: &mut DMatrix<f64>) {
    let n = obs_info.ncols();
    for j in 0..n {
        for i in j..n {
            if j == i {
                obs_info_nalg[(i, i)] = obs_info[[i, i]];
            } else {
                obs_info_nalg[(i, j)] = obs_info[[i, j]];
                obs_info_nalg[(j, i)] = obs_info[[j, i]];
            }
        }
    }
}

/// solve_for_cov_matrix — classical covariance from observed information.
///
/// Purpose
/// -------
/// Compute the classical covariance matrix as the Moore–Penrose
/// pseudoinverse `J⁺` of a symmetric observed information matrix
/// `J(θ̂)`, using symmetric eigendecomposition and eigenvalue
/// truncation.
///
/// Parameters
/// ----------
/// - `obs_info_nalg`: `DMatrix<f64>`
///   Symmetric `n×n` observed information matrix, typically produced by
///   [`fill_dmatrix`]. Consumed by the eigendecomposition.
///
/// Returns
/// -------
/// `Array2<f64>`
///   `n×n` classical covariance matrix `J⁺`, where small eigenvalues
///   (≤ [`EIGEN_EPS`]) have been truncated.
///
/// Notes
/// -----
/// - Eigenvalues `λ_k` with `λ_k ≤ EIGEN_EPS` are treated as zero when
///   forming the pseudoinverse, inflating uncertainty along weakly
///   identified directions.
fn solve_for_cov_matrix(obs_info_nalg: DMatrix<f64>) -> Array2<f64> {
    let n = obs_info_nalg.ncols();
    let pseudo_inv = pseudo_inverse(obs_info_nalg);
    let mut cov = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cov[[i, j]] = pseudo_inv[(i, j)];
        }
    }
    cov
}

/// solve_for_robust_cov_matrix — robust (sandwich) covariance.
///
/// Purpose
/// -------
/// Compute robust (sandwich) covariance by combining the pseudoinverse
/// `J⁺` of the observed information matrix `J(θ̂)` with a score
/// covariance matrix `S` on the average-score scale. The resulting
/// covariance is `Var(θ̂) = J⁺ S J⁺`.
///
/// Parameters
/// ----------
/// - `obs_info_nalg`: `DMatrix<f64>`
///   Symmetric `n×n` observed information matrix on the average
///   log-likelihood scale. Consumed by the eigendecomposition.
/// - `scores`: `&Array2<f64>`
///   `n×n` score covariance matrix `S` on the *average-score* scale
///   (IID OPG or HAC). Must be symmetric and conformable with
///   `obs_info_nalg`.
///
/// Returns
/// -------
/// `Array2<f64>`
///   `n×n` robust covariance matrix `Var(θ̂) = J⁺ S J⁺`.
///
/// Notes
/// -----
/// - Eigenvalues `λ_k ≤ EIGEN_EPS` are discarded when constructing
///   `J⁺`, which protects against division by very small eigenvalues
///   while inflating covariance along nearly flat directions.
fn solve_for_robust_cov_matrix(obs_info_nalg: DMatrix<f64>, scores: &Array2<f64>) -> Array2<f64> {
    let n = obs_info_nalg.ncols();
    let pseudo_inv = pseudo_inverse(obs_info_nalg);

    let mut scores_nalg = DMatrix::<f64>::zeros(scores.nrows(), scores.ncols());
    for i in 0..scores.nrows() {
        for j in 0..scores.ncols() {
            scores_nalg[(i, j)] = scores[[i, j]];
        }
    }

    let cov_nalg = &pseudo_inv * &scores_nalg * &pseudo_inv;
    let mut cov = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cov[[i, j]] = cov_nalg[(i, j)];
        }
    }
    cov
}

/// pseudo_inverse — eigen-based Moore–Penrose pseudoinverse of a symmetric matrix.
///
/// Purpose
/// -------
/// Construct the Moore–Penrose pseudoinverse of a symmetric `n×n`
/// matrix using symmetric eigendecomposition and eigenvalue truncation
/// controlled by [`EIGEN_EPS`].
///
/// Parameters
/// ----------
/// - `obs_info_nalg`: `DMatrix<f64>`
///   Symmetric `n×n` matrix to be pseudo-inverted (typically the observed
///   information `J(θ̂)`). Consumed by the eigendecomposition.
///
/// Returns
/// -------
/// `DMatrix<f64>`
///   `n×n` pseudoinverse `J⁺`, where eigenvalues `λ_k ≤ EIGEN_EPS` have
///   been treated as zero.
///
/// Notes
/// -----
/// - The pseudoinverse is constructed as
///   `J⁺ = Σ_{k: λ_k > EIGEN_EPS} (1 / λ_k) q_k q_kᵀ`, where `q_k` are
///   the orthonormal eigenvectors of `J`.
fn pseudo_inverse(obs_info_nalg: DMatrix<f64>) -> DMatrix<f64> {
    let n = obs_info_nalg.ncols();
    let eigen_decomp = obs_info_nalg.symmetric_eigen();
    let q = eigen_decomp.eigenvectors;
    let eigenvals = eigen_decomp.eigenvalues;

    let mut pseudo_inv = DMatrix::<f64>::zeros(n, n);
    for k in 0..n {
        let lambda = eigenvals[k];
        if lambda > EIGEN_EPS {
            let coeff = 1.0 / lambda;
            for i in 0..n {
                for j in 0..n {
                    pseudo_inv[(i, j)] += coeff * q[(i, k)] * q[(j, k)];
                }
            }
        }
    }
    pseudo_inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use ndarray::{Array1, Array2, array};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Correct copying of Hessians from `ndarray` into `DMatrix`.
    // - Classical covariance for simple quadratic objectives with known
    //   analytic information matrices.
    // - Robust (sandwich) covariance inflation when the score covariance
    //   increases variance relative to the classical case.
    //
    // They intentionally DO NOT cover:
    // - End-to-end ACD model inference or HAC bandwidth selection.
    // - Pathological cases where `compute_hessian` itself fails.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `fill_dmatrix` copies entries from an `ndarray` Hessian
    // into a `nalgebra::DMatrix` without altering values or symmetry.
    //
    // Given
    // -----
    // - A small 2×2 symmetric `Array2<f64>` with distinct entries.
    //
    // Expect
    // ------
    // - The corresponding `DMatrix` has identical entries at all positions.
    fn fill_dmatrix_copies_ndarray_into_dmatrix_without_modification() {
        // Arrange
        let obs_info: Array2<f64> = array![[2.0, 0.5], [0.5, 1.0]];
        let mut obs_info_nalg = DMatrix::<f64>::zeros(2, 2);

        // Act
        fill_dmatrix(&obs_info, &mut obs_info_nalg);

        // Assert
        assert_eq!(obs_info_nalg[(0, 0)], 2.0);
        assert_eq!(obs_info_nalg[(0, 1)], 0.5);
        assert_eq!(obs_info_nalg[(1, 0)], 0.5);
        assert_eq!(obs_info_nalg[(1, 1)], 1.0);
    }

    #[test]
    // Purpose
    // -------
    // Check that `calc_covariance` produces a classical covariance matrix
    // equal to the analytic pseudoinverse for a simple diagonal quadratic.
    //
    // Given
    // -----
    // - A diagonal information matrix A = diag(4, 1) encoded via a linear
    //   gradient map g(θ) = A θ.
    // - A generic θ̂ (its value is irrelevant for a constant Hessian).
    //
    // Expect
    // ------
    // - Classical covariance is approximately diag(1/4, 1).
    fn calc_covariance_diagonal_quadratic_matches_analytic_inverse() {
        // Arrange
        let a = array![[4.0, 0.0], [0.0, 1.0]];
        let f = |theta: &Array1<f64>| -> Array1<f64> { a.dot(theta) };
        let theta_hat = array![1.0, -1.0];

        // Act
        let cov_res: OptResult<Array2<f64>> = calc_covariance(&f, &theta_hat, None);

        // Assert
        assert!(cov_res.is_ok());
        let cov = cov_res.unwrap();
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);

        // Analytic J⁺ = diag(1/4, 1)
        assert!((cov[[0, 0]] - 0.25).abs() < 1e-6);
        assert!((cov[[1, 1]] - 1.0).abs() < 1e-6);
        // Off-diagonals should be ~0 for a diagonal A.
        assert!(cov[[0, 1]].abs() < 1e-8);
        assert!(cov[[1, 0]].abs() < 1e-8);
    }

    #[test]
    // Purpose
    // -------
    // Verify that providing a score covariance matrix with larger variance
    // inflates robust covariance relative to classical covariance.
    //
    // Given
    // -----
    // - An identity information matrix J = I_2 encoded via a linear
    //   gradient map g(θ) = I θ.
    // - A score covariance S = 2 I_2, representing doubled variance.
    //
    // Expect
    // ------
    // - Classical covariance is approximately I_2.
    // - Robust covariance is approximately 2 I_2 and strictly dominates
    //   classical along the diagonal.
    fn calc_covariance_robust_inflates_variance_relative_to_classical() {
        // Arrange
        let f = |theta: &Array1<f64>| -> Array1<f64> { theta.clone() };
        let theta_hat = array![0.0, 0.0];
        let scores: Array2<f64> = array![[2.0, 0.0], [0.0, 2.0]];

        // Act
        let cov_classical_res: OptResult<Array2<f64>> = calc_covariance(&f, &theta_hat, None);
        let cov_robust_res: OptResult<Array2<f64>> = calc_covariance(&f, &theta_hat, Some(&scores));

        // Assert
        assert!(cov_classical_res.is_ok());
        assert!(cov_robust_res.is_ok());

        let cov_classical = cov_classical_res.unwrap();
        let cov_robust = cov_robust_res.unwrap();

        // Shapes
        assert_eq!(cov_classical.nrows(), 2);
        assert_eq!(cov_classical.ncols(), 2);
        assert_eq!(cov_robust.nrows(), 2);
        assert_eq!(cov_robust.ncols(), 2);

        // Classical: Var ≈ 1 on both coordinates.
        assert!((cov_classical[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((cov_classical[[1, 1]] - 1.0).abs() < 1e-8);

        // Robust: Var ≈ 2 on both coordinates.
        assert!((cov_robust[[0, 0]] - 2.0).abs() < 1e-8);
        assert!((cov_robust[[1, 1]] - 2.0).abs() < 1e-8);

        // Robust covariance should strictly dominate classical on the diagonal.
        assert!(cov_robust[[0, 0]] > cov_classical[[0, 0]]);
        assert!(cov_robust[[1, 1]] > cov_classical[[1, 1]]);
    }
}
