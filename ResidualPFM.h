#pragma once
//
// Residual-only global assembly.
//
// The element-level physics (residual AND tangent) now lives in a single
// shared routine, pfm::elementSystem (StiffnessPFM.h). This header exposes
// only the residual-only global assembly, used by the solver where the
// tangent K is NOT needed -- e.g. the staggered scheme's coupled-residual
// convergence check and the monolithic post-loop residual evaluation.
//
// When both K and R are needed, call pfm::assembleGlobalSystem instead; it
// produces them together in one pass.
//
// External forces are NOT included here. The caller subtracts f_ext (surface
// tractions, point loads) from the assembled internal residual.
//
// DOF ordering: global residual is size 3*npoin, block layout [u | phi]
//   first  2*npoin : displacement entries, node-major
//   last     npoin : phase-field entries
//

#include "FEMINPUT.h"
#include "History.h"
#include "StiffnessPFM.h"      // pfm::elementSystem, pfm::MatParams

#include <Eigen/Dense>

namespace pfm {

// Global internal-force residual of length 3*npoin.
Eigen::VectorXd
assembleGlobalResidual(const FemInput&        d,
                       const Eigen::VectorXd& u_global,
                       const Eigen::VectorXd& phi_global,
                       const HistoryField&    history);

}  // namespace pfm
