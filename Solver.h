#pragma once
//
// Newton solver for one quasi-static load step of the phase-field fracture
// problem.
//
// Two solution schemes are available (SolverSettings::scheme):
//
//   * Monolithic -- solve the full coupled [u | phi] Newton system at once.
//                   Fast quadratic convergence near the solution, but the
//                   tangent can be indefinite during crack propagation,
//                   where Newton may diverge.
//
//   * Staggered  -- alternate minimisation: within each load step, repeatedly
//                   (a) solve the displacement sub-problem with phi frozen,
//                   then (b) solve the phase-field sub-problem with u frozen,
//                   until the full coupled residual is small. Each sub-problem
//                   is well-posed, so the scheme is much more robust during
//                   crack growth, at the cost of linear (rather than
//                   quadratic) outer-sweep convergence.
//
// solveStep() runs ONE Newton sequence at the prescribed load_factor and
// honours SolverSettings::scheme.
//

#include "FEMINPUT.h"
#include "History.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

namespace pfm {

// Coupling strategy for the monolithic [u | phi] system.
enum class SolverScheme {
    Monolithic,   // solve u and phi together
    Staggered     // alternate-minimisation: u-solve, then phi-solve, repeat
};

// Stopping criterion for the OUTER staggered cycle (ignored by the monolithic
// scheme and by the inner u/phi Newton sub-solves, which always use the
// residual tolerances tol_rel / tol_abs).
//
//   Residual    -- stop the cycle when the full coupled residual is small:
//                  ||R|| < max(tol_abs, tol_rel*||R0||). Drives the staggered
//                  fixed point to the monolithic solution; can need many sweeps.
//
//   EnergyGamma -- Ambati et al. (2015), Sect. 3.4. Monitor the total energy
//                  E^k of the iterate and stop when the normalized energy-slope
//                  angle gamma drops to a tolerance (default 10 deg):
//                    E_k^Norm = (E^k - E^N)/(E^1 - E^N),   s_k = k/N
//                    gamma    = atan( N*(E^{k-1}-E^k)/(E^1-E^k) ) * 180/pi
//                  evaluated online at the current sweep N. Robust to the very
//                  different scales of E^k and k that make a raw energy/slope
//                  tolerance unusable.
enum class StaggerStop {
    Residual,
    EnergyGamma
};

// How the box constraint  phi_n <= phi <= 1  is enforced in the staggered
// phase-field sub-solve (phi_n = phase field of the last ACCEPTED step).
//
//   None      -- not enforced (original behaviour). With AT2 + the history
//                field phi stays in [0,1] on its own and does not heal.
//   ActiveSet -- primal-dual active set (Hintermueller, Ito & Kunisch 2002;
//                for phase-field fracture: Heister, Wheeler & Wick 2015).
//                The phi-problem is affine for frozen u, so:
//                  1. solve with the current fixed (active) set
//                  2. residual R = K phi - f of the UNMODIFIED system
//                  3. fix free nodes outside [phi_n, 1] at the violated
//                     bound; release fixed nodes whose R has the wrong sign
//                     (lower bound: release if R < 0, upper: if R > 0)
//                  4. repeat until the set no longer changes
//                Fixed nodes are imposed exactly like Dirichlet DOFs, so the
//                sparsity pattern (and the symbolic factorisation) is reused.
//                Staggered scheme only.
enum class PhiBounds { None, ActiveSet };

// What prevents the crack from healing.
//
//   History -- phi is driven by H = max(H_n, psi+) (Miehe 2010); the driver
//              raises H after every accepted step. Original behaviour.
//   Bounds  -- phi is driven by psi+ directly (H is never raised, so
//              max(0, psi+) = psi+), and irreversibility comes ONLY from the
//              lower bound phi >= phi_n. Requires PhiBounds::ActiveSet.
//              Variationally cleaner; the setting needed for formulations
//              that do not keep phi in range on their own (AT1, PF-CZM).
enum class Irreversibility { History, Bounds };

struct SolverSettings {
    // A Newton (sub-)solve terminates when ||R|| < max(tol_abs, tol_rel*||R0||).
    double tol_rel  = 1.0e-5;
    double tol_abs  = 1.0e-10;
    int    max_iter = 30;        // max Newton iterations per (sub-)solve

    // Print residual norms.
    bool   verbose  = true;

    // Coupling scheme. Monolithic by default.
    SolverScheme scheme = SolverScheme::Monolithic;

    // Staggered only: max number of outer (u-solve / phi-solve) sweeps per
    // load step. Ignored when scheme == Monolithic.
    int max_staggered = 50;

    // Staggered only: which criterion ends the outer cycle (see StaggerStop).
    StaggerStop stagger_stop = StaggerStop::Residual;

    // Staggered + EnergyGamma only: stop when the normalized energy-slope angle
    // gamma <= this (degrees). Ambati et al. suggest ~10 deg.
    double stagger_gamma_tol_deg = 10.0;

    // Staggered only: if non-empty, the per-sweep energy history E^k is written
    // to this CSV (columns: load_factor,sweep,E_total,E_elastic,E_fracture,
    // R_norm). One row per outer sweep, across all load steps; filter by
    // load_factor to recover the {E^k} sequence of a single step and reproduce
    // Ambati Fig. 4/6. Empty (default) disables the log.
    std::string energy_csv;

    // Box constraint phi_n <= phi <= 1 in the phi sub-solve (see PhiBounds).
    PhiBounds       phi_bounds      = PhiBounds::None;
    // Healing prevention (see Irreversibility). Read by the driver.
    Irreversibility irreversibility = Irreversibility::History;
    // Active set: max passes (solve / update set) per phi-solve. The phi
    // problem is linear, so 2-5 passes are typical; hitting this limit marks
    // the phi-solve as not converged.
    int phi_active_set_max_iter = 50;
};

struct SolverResult {
    int    iters_used     = 0;     // Newton iterations (monolithic) or
                                   //   staggered sweeps (staggered) at exit
    double final_residual = 0.0;   // ||R|| at exit
    double initial_residual = 0.0; // ||R|| at the first iteration / sweep
    bool   converged      = false;
    // Active-set phi-solve only: nodes held at phi_n / at 1 on exit.
    int    n_at_lower     = 0;
    int    n_at_upper     = 0;
};

// Run one quasi-static step at load_factor * (prescribed BC values), updating
// u and phi in place. Vector sizes:
//   u   : length 2*npoin
//   phi : length npoin
//
// history is the per-Gauss-point crack-driving history (see History.h). It is
// read-only here; the driver should call updateHistory() after a converged
// step. Dispatches to the monolithic or staggered scheme per settings.scheme.
//
SolverResult solveStep(const FemInput&        d,
                       Eigen::VectorXd&       u,
                       Eigen::VectorXd&       phi,
                       const HistoryField&    history,
                       double                 load_factor,
                       const SolverSettings&  settings = SolverSettings{},
                       const Eigen::VectorXd* phi_lower = nullptr);
// phi_lower: the lower bound phi_n for PhiBounds::ActiveSet -- the phase field
// of the last ACCEPTED step (not of the previous sweep). nullptr = 0.

// ===========================================================================
// Dissipation-controlled staggered step (energy-release path following).
//
// The load factor lambda becomes an UNKNOWN of the step, fixed by requiring
// that the crack (fracture) energy of the step grows by exactly dD:
//
//     g(lambda) = Gamma(phi) - Gamma_n - dD = 0,
//     Gamma(phi) = INT Gc ( phi^2/(2 l0) + (l0/2)|grad phi|^2 ) dV   (E_fracture)
//
// Gamma only grows (irreversibility), so it is a valid path parameter through
// the peak and along a snap-back, where lambda itself must decrease.
//
// With the hybrid (isotropic, linear) momentum balance under pure Dirichlet
// loading, the displacement for frozen phi is exactly linear in lambda:
// u = lambda * u_hat(phi), u_hat = solution at lambda = 1. Each sweep runs
//   (a) u-solve with phi^k frozen at lambda = 1 -> u_hat     (1 u-solve)
//   (b) 1-D root-find on lambda: phi = Phi(lambda u_hat) with
//       H = max(H_n, psi+(lambda u_hat)) until Gamma(phi) - Gamma_n = dD
//       (bracketing secant / regula falsi, a few cheap phi-solves)
//   (c) u = lambda * u_hat, usual convergence test
// The crack energy is pinned in EVERY sweep, so the crack cannot run away
// past the peak; lambda drops instead (snap-back). Two cheaper variants were
// tried first and failed in tests: a closed-form lambda from the elastic-
// energy balance (oscillated during crack growth), and correcting lambda only
// after the sweeps settle at fixed lambda (at the peak the staggered iteration
// jumps to the fully-cracked branch, so the target cannot be met).
//
// The step converges when the usual staggered criterion holds AND
// |g| <= tol * dD (or lambda sits on the dlambda_max cap: "capped" step).
// ===========================================================================
struct DissipationControl {
    // ---- inputs ----
    double lambda_n    = 0.0;    // load factor of the last accepted state
    double Gamma_n     = 0.0;    // crack energy (E_fracture) of that state
    double dD          = 0.0;    // crack-energy increase wanted in this step
    double lambda_pred = 0.0;    // sweep-0 predictor
    double slope       = 0.0;    // rough dGamma/dlambda (> 0), 0 = unknown
    double dlambda_max = 0.0;    // cap on |lambda - lambda_n| (> 0)
    double tol         = 1.0e-2; // |g| <= tol * dD
    // Monolithic version only (energy-balance constraint): generalised force
    // conjugate to lambda at the last accepted state, P_n = 2 Psi_el/lambda_n.
    double P_n         = 0.0;
    // ---- outputs ----
    double lambda    = 0.0;      // load factor of the returned state
    double dGamma    = 0.0;      // achieved crack-energy increase
    int    n_bisect  = 0;        // sweeps that fell back to bisection
    bool   capped    = false;    // accepted on the RELOAD cap (g < 0 at
                                 // lambda_n + dlambda_max), |g| > tol
    bool   unload_capped = false;// rejected: dD exceeded even at the largest
                                 // allowed unloading -> driver halves dtau
    int    n_phi_solves = 0;     // phi-solves used (root-finder evaluations)
};

// Dispatches on settings.scheme:
//   Staggered  -- crack-energy constraint (above); needs fem.hybrid = true.
//   Monolithic -- energy-balance constraint on the bordered Newton system
//                 (see solveStepMonolithicDissipation in Solver.cpp); needs
//                 dc.P_n, no phi bounds, works with or without hybrid.
//                 dc.dGamma then reports the energy-balance dissipation.
// u and phi are updated in place; on return u corresponds to dc.lambda.
SolverResult solveStepDissipation(const FemInput&        d,
                                  Eigen::VectorXd&       u,
                                  Eigen::VectorXd&       phi,
                                  const HistoryField&    history,
                                  DissipationControl&    dc,
                                  const SolverSettings&  settings,
                                  const Eigen::VectorXd* phi_lower = nullptr);

}  // namespace pfm
