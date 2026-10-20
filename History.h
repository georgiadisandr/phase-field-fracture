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

#include <string>
#include <vector>

namespace pfm {

// history[e][g] = stored maximum of psi^+ at Gauss point g of element e.
// The number of Gauss points per element matches fem::gaussPoints(nnode, ngaus).
using HistoryField = std::vector<std::vector<double>>;

// Zero-initialised history sized to the mesh (one slot per Gauss point of
// every element). Call once, before the load-stepping loop.
[[nodiscard]] HistoryField makeHistoryField(const FemInput& d);

// Seed the initial cracks of d.initial_cracks ([[initial_history]]) into a
// freshly made history field (Borden et al. 2012). At every Gauss point at
// distance d <= w from a crack segment
//
//     history[e][g] = max(history[e][g], B * H_c * (1 - d / w))
//
// with H_c = Gc/(2 l0) (AT2) or 3 Gc/(16 l0) (AT1) and w = half_width, or
// l0/4 when half_width <= 0; Gc, l0 are those of the element's material.
// Call once, right after makeHistoryField(), before any solve.
struct SeedReport {
    std::string label;        // crack label (group name or initial_history[i])
    int         n_segments = 0;
    int         n_gauss    = 0;   // Gauss points that received H0 > 0
    double      H_max      = 0.0; // largest H0 written
    double      w_min      = 0.0; // half-widths actually used (per material)
    double      w_max      = 0.0;
};
[[nodiscard]] std::vector<SeedReport>
seedInitialCrackHistory(const FemInput& d, HistoryField& history);

// After a CONVERGED load step, raise the stored history to include the
// current state:  history[e][g] = max(history[e][g], psi^+(u) at that point).
// Call once per converged step before advancing to the next load increment.
void updateHistory(const FemInput&        d,
                   const Eigen::VectorXd& u,
                   HistoryField&          history);

}  // namespace pfm
