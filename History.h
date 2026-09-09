#pragma once
//
// Crack-driving history field for the phase-field fracture model (Miehe 2010).
//
// Irreversibility -- a crack must not heal -- is enforced by driving the
// phase field with a HISTORY variable H instead of the instantaneous positive
// elastic energy density psi^+:
//
//     H(x, t) = max over the load history [0, t] of psi^+(x, tau)
//
// Stored per element, per Gauss point. During a load step the assembly uses
//
//     H_eff(x) = max( H_stored(x), psi^+(u_current)(x) )
//
// so new damage can still form, while H_stored holds the (frozen) maximum
// from all previously converged steps. After a step converges the driver
// calls updateHistory() to fold the converged psi^+ into H_stored.
//
// Without this, the staggered / monolithic iteration can let phi decrease
// when the local strain relaxes (e.g. as the crack-tip field redistributes),
// which is unphysical and hurts convergence.
//

#include "FEMINPUT.h"

#include <Eigen/Dense>

#include <vector>

namespace pfm {

// history[e][g] = stored maximum of psi^+ at Gauss point g of element e.
// The number of Gauss points per element matches fem::gaussPoints(nnode, ngaus).
using HistoryField = std::vector<std::vector<double>>;

// Zero-initialised history sized to the mesh (one slot per Gauss point of
// every element). Call once, before the load-stepping loop.
[[nodiscard]] HistoryField makeHistoryField(const FemInput& d);

// After a CONVERGED load step, raise the stored history to include the
// current state:  history[e][g] = max(history[e][g], psi^+(u) at that point).
// Call once per converged step before advancing to the next load increment.
void updateHistory(const FemInput&        d,
                   const Eigen::VectorXd& u,
                   HistoryField&          history);

}  // namespace pfm
