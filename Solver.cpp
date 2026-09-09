#include "Solver.h"

#include "BoundaryConditions.h"
#include "NeumannBC.h"
#include "Profiling.h"
#include "ResidualPFM.h"
#include "StiffnessPFM.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfm {

namespace {

// Direct linear solvers.
//   * SpLU   -- general unsymmetric LU. The monolithic tangent can become
//               indefinite during crack growth, so it needs the general LU.
//   * SpLDLT -- symmetric LDL^T (Cholesky). The staggered displacement and
//               phase-field sub-problems are symmetric positive-definite, so
//               they use this
using SpLU   = Eigen::SparseLU<Eigen::SparseMatrix<double>>;
using SpLDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;

// ---------------------------------------------------------------------------
// Factorize K and solve  K * delta = rhs.
//
// The sparsity pattern of K is structurally constant for the entire run: the
// mesh is fixed, assembleGlobalSystem() emits every structural entry (zeros
// included), and the Dirichlet- / frozen-DOF sets never change between
// iterations or load steps. So the expensive symbolic analysis
// (analyzePattern) is run EXACTLY ONCE per solver -- guarded by 'analyzed' --
// and every later call only re-runs the cheap numeric factorize().
//
// This is what the project always intended (see the comment in StiffnessPFM)
// but never actually did: the old code called compute(), which redoes the
// full symbolic analysis on every Newton iteration.
//
// 'solver' and 'analyzed' are passed in (held 'static' by the caller) so the
// analyzed pattern survives across iterations, sweeps and load steps.
// ---------------------------------------------------------------------------
template <class Solver>
Eigen::VectorXd factorizeAndSolve(Solver&                            solver,
                                  bool&                              analyzed,
                                  const Eigen::SparseMatrix<double>& K,
                                  const Eigen::VectorXd&             rhs,
                                  const char*                        where)
{
    {
        prof::Scope _t(prof::Factorize);

        //Analyzes sparsity of K matrix if it hasn't already
        if (!analyzed) {
            solver.analyzePattern(K);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error(std::string(where) +
                                         ": symbolic analyzePattern failed");
            analyzed = true;
        }

        //factorize matrix K to either K = L * U or K =L *D L^T
        solver.factorize(K);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error(
                std::string(where) +
                ": numeric factorize failed (matrix singular, or not "
                "positive-definite for the Cholesky sub-solves)");
    }

    //solve the system
    Eigen::VectorXd delta;
    {
        prof::Scope _t(prof::Solve);
        delta = solver.solve(rhs);
    }
    if (solver.info() != Eigen::Success)
        throw std::runtime_error(std::string(where) + ": solve failed");
    if (!delta.allFinite())
        throw std::runtime_error(std::string(where) + ": non-finite increment");
    return delta;
}

// Zero the residual at Dirichlet-constrained u-DOFs. Those entries hold the
// reaction force, which is a valid output but NOT a quantity to drive to
// zero, so they must be excluded from any convergence norm.
void zeroReactions(Eigen::VectorXd& R, const FemInput& d)
{
    for (int i = 0; i < d.nvfix; ++i) {
        const int node = d.nofix[i];
        for (int k = 0; k < d.ndofn; ++k) {
            if (d.iffix[i][k] != 0)
                R(2 * node + k) = 0.0;
        }
    }
}

// ===========================================================================
// Monolithic scheme
// ===========================================================================

SolverResult solveStepMonolithic(const FemInput&        d,
                                 Eigen::VectorXd&       u,
                                 Eigen::VectorXd&       phi,
                                 const HistoryField&    history,
                                 double                 load_factor,
                                 const SolverSettings&  settings)
{
    const int npoin = d.npoin;

    SolverResult res;

    if (settings.verbose) {
        std::cout << "[solver] MONOLITHIC Newton  load_factor = " << load_factor
                  << ",  tol_rel = " << settings.tol_rel
                  << ",  max_iter = " << settings.max_iter << "\n";
    }

    double R_norm_0 = -1.0;

    const Eigen::VectorXd F_ext_full = assembleExternalForce(d); // Newmann BC
    const bool any_neumann           = (F_ext_full.array() != 0.0).any();

    // Persistent LU factorization: the symbolic analysis runs once, then only
    // the numeric factorize() repeats. 'static' keeps it alive across steps,
    // where the sparsity pattern is unchanged.
    static SpLU lu;
    static bool lu_analyzed = false;

    for (int iter = 0; iter < settings.max_iter; ++iter) {
        // One assembly pass produces the residual AND the tangent.
        GlobalSystem sys = assembleGlobalSystem(d, u, phi, history);
        if (any_neumann)
            sys.R.head(2 * npoin).noalias() -= load_factor * F_ext_full;

        if (iter > 0) {
            zeroReactions(sys.R, d);
            const double R_norm = sys.R.norm();
            if (settings.verbose) {
                std::cout << "  iter " << iter
                          << "  |R| = " << R_norm
                          << "   |R|/|R0| = " << (R_norm / R_norm_0) << "\n";
            }
            const double tol = std::max(settings.tol_abs,
                                        settings.tol_rel * R_norm_0);
            if (R_norm < tol) {
                res.iters_used     = iter;
                res.final_residual = R_norm;
                res.converged      = true;
                if (settings.verbose)
                    std::cout << "[solver] converged in " << iter
                              << " iterations\n";
                return res;
            }
        }

        applyDirichlet(sys.K, sys.R, d, u, load_factor);

        if (iter == 0) {
            const double R_norm  = sys.R.norm();
            R_norm_0             = std::max(R_norm, settings.tol_abs);
            res.initial_residual = R_norm;
            if (settings.verbose)
                std::cout << "  iter 0  |R| = " << R_norm
                          << "   |R|/|R0| = 1\n";
        }

        const Eigen::VectorXd delta =
            factorizeAndSolve(lu, lu_analyzed, sys.K, -sys.R, "solveStep");

        u   += delta.head(2 * npoin);
        phi += delta.tail(npoin);

        res.iters_used     = iter + 1;
        res.final_residual = sys.R.norm();
    }

    {
        Eigen::VectorXd R_end = assembleGlobalResidual(d, u, phi, history);
        if (any_neumann)
            R_end.head(2 * npoin).noalias() -= load_factor * F_ext_full;
        zeroReactions(R_end, d);
        res.final_residual = R_end.norm();
    }

    res.converged = false;
    if (settings.verbose)
        std::cout << "[solver] did NOT converge in " << res.iters_used
                  << " iterations  (|R| = " << res.final_residual << ")\n";
    return res;
}

// ===========================================================================
// Staggered scheme: sub-solves
// ===========================================================================

// Newton sub-solve of the DISPLACEMENT block with phi held fixed.
// The full monolithic K and R are assembled, the Dirichlet BCs are applied,
// then the phi-block is frozen (identity) so the linear solve returns a zero
// phi-increment and the exact u sub-problem increment. u is updated in place.
//
SolverResult subsolveU(const FemInput&        d,
                       Eigen::VectorXd&       u,
                       const Eigen::VectorXd& phi,
                       const HistoryField&    history,
                       double                 load_factor,
                       const Eigen::VectorXd& F_ext,
                       bool                   any_neumann,
                       const SolverSettings&  settings)
{
    (void)load_factor;        // used below via the F_ext correction
    SolverResult res;
    double R0 = -1.0;

    // The u sub-problem is symmetric positive-definite -> Cholesky. 'static'
    // keeps the once-only symbolic analysis alive across sweeps and load steps.
    // The matrix is now 2*npoin (u-block only), not the old 3*npoin coupled
    // system with a frozen identity phi-block.
    static SpLDLT ldlt;
    static bool   ldlt_analyzed = false;

    for (int iter = 0; iter < settings.max_iter; ++iter) {
        USystem sys = assembleU_System(d, u, phi, history);
        if (any_neumann)
            sys.R.noalias() -= load_factor * F_ext;

        // Convergence check (skip iter 0 -- u may not yet satisfy the
        // Dirichlet target so the residual is not meaningful).
        if (iter > 0) {
            Eigen::VectorXd Ru = sys.R;
            zeroReactions(Ru, d);
            const double Rn  = Ru.norm();
            const double tol = std::max(settings.tol_abs,
                                        settings.tol_rel * R0);
            if (Rn < tol) {
                res.iters_used     = iter;
                res.final_residual = Rn;
                res.converged      = true;
                return res;
            }
        }

        applyDirichlet(sys.K, sys.R, d, u, load_factor);

        if (iter == 0) {
            R0                   = std::max(sys.R.norm(), settings.tol_abs);
            res.initial_residual = R0;
        }

        const Eigen::VectorXd delta =
            factorizeAndSolve(ldlt, ldlt_analyzed, sys.K, -sys.R, "subsolveU");

        u += delta;

        res.iters_used     = iter + 1;
        res.final_residual = sys.R.norm();
    }

    res.converged = false;
    return res;
}

// Newton sub-solve of the PHASE-FIELD block with u held fixed.
// With u (hence psi^+ and the history H) frozen the phase-field residual is
// affine in phi, so this converges in a single iteration; the loop is kept
// generic for robustness. The u-block is frozen (identity).
SolverResult subsolvePhi(const FemInput&        d,
                         const Eigen::VectorXd& u,
                         Eigen::VectorXd&       phi,
                         const HistoryField&    history,
                         const SolverSettings&  settings)
{
    SolverResult res;
    double R0 = -1.0;

    // The phi sub-problem is symmetric positive-definite -> Cholesky, with
    // the symbolic analysis reused across calls. Matrix size is npoin only,
    // not 3*npoin with a frozen u-block.
    static SpLDLT ldltP;
    static bool   ldltP_analyzed = false;

    for (int iter = 0; iter < settings.max_iter; ++iter) {
        PhiSystem sys = assemblePhi_System(d, u, phi, history);
        const double Rn = sys.R.norm();

        if (iter == 0) {
            R0                   = std::max(Rn, settings.tol_abs);
            res.initial_residual = R0;
        }
        const double tol = std::max(settings.tol_abs, settings.tol_rel * R0);
        if (Rn < tol) {
            res.iters_used     = iter;
            res.final_residual = Rn;
            res.converged      = true;
            return res;
        }

        const Eigen::VectorXd delta =
            factorizeAndSolve(ldltP, ldltP_analyzed, sys.K, -sys.R, "subsolvePhi");

        phi += delta;

        res.iters_used     = iter + 1;
        res.final_residual = sys.R.norm();
    }

    res.converged = false;
    return res;
}

// ===========================================================================
// Staggered scheme: outer sweep
// ===========================================================================

SolverResult solveStepStaggered(const FemInput&        d,
                                Eigen::VectorXd&       u,
                                Eigen::VectorXd&       phi,
                                const HistoryField&    history,
                                double                 load_factor,
                                const SolverSettings&  settings)
{
    const int npoin = d.npoin;
    const int n_u   = 2 * npoin;

    SolverResult res;

    const Eigen::VectorXd F_ext = assembleExternalForce(d);
    const bool any_neumann      = (F_ext.array() != 0.0).any();

    if (settings.verbose)
        std::cout << "[solver] STAGGERED  load_factor = " << load_factor
                  << "  (max " << settings.max_staggered << " sweeps)\n";

    // Optional per-sweep energy history E^k (Ambati Sect. 3.4). Opened once on
    // first use (the LU/LDLT factorizations above use the same static-state
    // pattern); header written once; one row appended per sweep with the
    // load_factor column so a single step's {E^k} can be filtered out later.
    static std::ofstream energy_csv;
    static bool          energy_csv_open = false;
    if (!settings.energy_csv.empty() && !energy_csv_open) {
        energy_csv.open(settings.energy_csv, std::ios::out | std::ios::trunc);
        if (energy_csv)
            energy_csv << "load_factor,sweep,E_total,E_elastic,E_fracture,R_norm\n";
        energy_csv_open = true;
    }

    double R0 = -1.0;

    // Outer-cycle stopping criterion. The inner u/phi Newton sub-solves always
    // use the residual tolerances; this only selects how the OUTER sweep loop
    // decides it is done.
    const bool   use_gamma = (settings.stagger_stop == StaggerStop::EnergyGamma);
    constexpr double PI    = 3.14159265358979323846;  // M_PI is not portable (MSVC)
    double E_first = 0.0;   // E^1  (first sweep of this cycle)
    double E_prev  = 0.0;   // E^{k-1}

    for (int sweep = 0; sweep < settings.max_staggered; ++sweep) {
        // (a) displacement sub-solve with phi frozen
        const SolverResult ru = subsolveU(d, u, phi, history, load_factor,
                                          F_ext, any_neumann, settings);

        // Early bail-out: if the inner u-Newton failed to converge inside
        // max_iter iterations, u has drifted into an unreliable state. Any
        // further staggered sweeps would feed that bad u into the phase field
        // via H_eff = max(H_stored, psi^+(u)) and burn time on a polluted
        // trajectory. Return immediately so the driver (main.cpp) can
        // subdivide the load increment.
        if (!ru.converged) {
            if (settings.verbose)
                std::cout << "  [solver] aborting staggered loop at sweep "
                          << sweep << ": inner u-Newton did not converge "
                          << "in " << ru.iters_used << " iterations\n";
            res.iters_used     = sweep + 1;
            res.final_residual = ru.final_residual;
            if (sweep == 0) res.initial_residual = ru.initial_residual;
            res.converged      = false;
            return res;
        }

        // (b) phase-field sub-solve with u frozen
        const SolverResult rp = subsolvePhi(d, u, phi, history, settings);

        // The phi sub-problem is nearly affine, so failure here is rare --
        // but if it happens, same logic applies: bail rather than iterate
        // on a polluted phi.
        if (!rp.converged) {
            if (settings.verbose)
                std::cout << "  [solver] aborting staggered loop at sweep "
                          << sweep << ": inner phi-Newton did not converge "
                          << "in " << rp.iters_used << " iterations\n";
            res.iters_used     = sweep + 1;
            res.final_residual = rp.final_residual;
            if (sweep == 0) res.initial_residual = ru.initial_residual;
            res.converged      = false;
            return res;
        }

        // Full coupled residual: measures whether the alternating split has
        // settled (it converges as the phase field stops changing).
        Eigen::VectorXd R = assembleGlobalResidual(d, u, phi, history);
        if (any_neumann)
            R.head(n_u).noalias() -= load_factor * F_ext;
        zeroReactions(R, d);
        const double Rn = R.norm();

        if (sweep == 0) {
            // Baseline scale: the first u sub-solve's iter-0 residual reflects
            // the (load_factor) BC violation, mirroring the monolithic R0.
            R0                   = std::max(ru.initial_residual, settings.tol_abs);
            res.initial_residual = ru.initial_residual;
        }

        res.iters_used     = sweep + 1;
        res.final_residual = Rn;

        // Total energy of the current iterate E^k = E(u^k, phi^k). The natural
        // scalar for monitoring staggered convergence (decreases monotonically
        // toward the minimizer). Computed only when something consumes it
        // (verbose print or the CSV log) to avoid the extra mesh pass otherwise.
        const bool want_energy =
            settings.verbose || !settings.energy_csv.empty() || use_gamma;
        EnergyParts E;
        if (want_energy) E = computeEnergy(d, u, phi);

        if (energy_csv_open && energy_csv) {
            energy_csv << load_factor << ',' << (sweep + 1) << ','
                       << E.total() << ',' << E.elastic << ','
                       << E.fracture << ',' << Rn << '\n';
            energy_csv.flush();
        }

        // Normalized energy-slope angle gamma (Ambati Sect. 3.4), evaluated
        // online at the current sweep N = sweep+1 using E^1, E^{k-1}, E^k:
        //   gamma = atan( N * (E^{k-1}-E^k)/(E^1-E^k) ) * 180/pi.
        // Needs at least 2 sweeps; a flat energy (negligible total drop) is
        // treated as already at the minimizer.1q       
        double gamma_deg         = std::numeric_limits<double>::infinity();
        bool   energy_converged  = false;
        if (use_gamma) {
            const double Ek = E.total();
            if (sweep == 0) {
                E_first = Ek;
            } else {
                const double num   = E_prev  - Ek;   // last-sweep energy drop
                const double den   = E_first - Ek;   // total drop so far
                const double floor = std::max(1e-30, 1e-12 * std::abs(E_first));
                if (std::abs(den) <= floor) {
                    energy_converged = true;         // energy flat -> converged
                    gamma_deg        = 0.0;
                } else {
                    const int    N     = sweep + 1;
                    const double slope = static_cast<double>(N) * num / den;
                    gamma_deg          = std::atan(slope) * 180.0 / PI;
                    energy_converged   = (gamma_deg <= settings.stagger_gamma_tol_deg);
                }
            }
            E_prev = Ek;
        }

        if (settings.verbose) {
            std::cout << "  solver sweep " << sweep
                      << "  |R| = " << Rn
                      << "   |R|/|R0| = " << (Rn / R0)
                      << "   E = " << E.total();
            if (use_gamma) std::cout << "   gamma = " << gamma_deg << " deg";
            std::cout << "   (u: " << ru.iters_used << " Newton it"
                      << (ru.converged ? "" : ", NOT conv")
                      << ";  phi: " << rp.iters_used << " it"
                      << (rp.converged ? "" : ", NOT conv")
                      << ")\n";
        }

        bool converged;
        if (use_gamma) {
            converged = energy_converged;            // needs sweep >= 1
        } else {
            const double tol = std::max(settings.tol_abs, settings.tol_rel * R0);
            converged = (Rn < tol);
        }
        if (converged) {
            res.converged = true;
            if (settings.verbose)
                std::cout << "[solver] staggered converged in "
                          << (sweep + 1) << " sweeps "
                          << (use_gamma ? "(gamma criterion)" : "(residual criterion)")
                          << "\n";
            return res;
        }
    }

    res.converged = false;
    if (settings.verbose)
        std::cout << "[solver] staggered did NOT converge in "
                  << res.iters_used << " sweeps  (|R| = "
                  << res.final_residual << ")\n";
    return res;
}

}  // namespace

// ===========================================================================
// Public API: dispatch on settings.scheme
// ===========================================================================

SolverResult solveStep(const FemInput&        d,
                       Eigen::VectorXd&       u,
                       Eigen::VectorXd&       phi,
                       const HistoryField&    history,
                       double                 load_factor,
                       const SolverSettings&  settings)
{
    const int npoin = d.npoin;
    if (u.size()   != 2 * npoin)
        throw std::runtime_error("solveStep: u size != 2*npoin");
    if (phi.size() != npoin)
        throw std::runtime_error("solveStep: phi size != npoin");

    if (settings.scheme == SolverScheme::Staggered)
        return solveStepStaggered(d, u, phi, history, load_factor, settings);
    return solveStepMonolithic(d, u, phi, history, load_factor, settings);
}

}  // namespace pfm
