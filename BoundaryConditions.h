#pragma once
//
// Dirichlet BC enforcement on the monolithic Newton system K * delta = -R.
//
// The unknown delta has length 3*npoin in block layout [u | phi]:
//   delta[0 .. 2*npoin)        -> displacement increments  (node-major)
//   delta[2*npoin .. 3*npoin)  -> phase-field increments
//
// Only displacement DOFs receive Dirichlet BCs here (the standard PFM choice).
// Phase-field DOFs are always free; if you later need bounds on phi
// (irreversibility, 0 <= phi <= 1), that goes in a separate module.
//
// At each Newton iteration of a load step we want, for every constrained DOF r,
//
//     u_total[r]  =  load_factor * v_target[r]                     (1)
//
// Inside the Newton iteration we solve for the increment delta and update
// u_total <- u_total + delta. So the increment we must impose is
//
//     delta_prescribed[r]  =  load_factor * v_target[r]  -  u_current[r]
//
// applyDirichlet rewrites (K, R) so that K * delta = -R returns exactly
// delta[r] = delta_prescribed[r] at every constrained DOF, while keeping the
// free DOFs consistent (column contributions of the constrained DOFs moved to
// the right-hand side).
//

#include "FEMINPUT.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace pfm {

// In-place modification of K and R to enforce Dirichlet BCs on the
// displacement DOFs. After the call:
//
//   * for every constrained DOF r:
//       - row r and column r of K are zero, except K(r, r) = 1
//       - R(r) = -delta_prescribed[r]
//   * for every free DOF i:
//       - R(i) is corrected by  R(i) += sum_{r constrained} K_orig(i, r) * delta_prescribed[r]
//       - K(i, r) and K(r, i) cleaned to zero
//
// load_factor lets a load-stepping loop ramp the prescribed values
// (e.g. 0.1, 0.2, ..., 1.0). Pass 1.0 to apply the BC values as-is.
void applyDirichlet(Eigen::SparseMatrix<double>& K,
                    Eigen::VectorXd&             R,
                    const FemInput&              d,
                    const Eigen::VectorXd&       u_current,
                    double                       load_factor = 1.0);

}  // namespace pfm
