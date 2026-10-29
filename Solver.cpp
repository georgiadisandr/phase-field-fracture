#include "Solver.h"

#include "BoundaryConditions.h"
#include "NeumannBC.h"
#include "Profiling.h"
#include "ResidualPFM.h"
#include "StiffnessPFM.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#ifdef PFM_USE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
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
//
// Build with -DPFM_USE_CHOLMOD=ON to use SuiteSparse CHOLMOD for the SPD
// sub-problems instead of Eigen's own SimplicialLDLT. It MUST be the
// *supernodal* variant: it factors dense column blocks through BLAS, which on
// a ~100k-dof Q4 mesh was ~8x faster per factorization (single thread) than
// SimplicialLDLT. CHOLMOD's *simplicial* mode is no faster than Eigen's.
//
// Measured on the Windows/MSYS2 build (i7-10870H, single thread):
//   K_u   (104k dofs): Eigen 1.57 s   CHOLMOD supernodal 0.64 s
//   K_phi ( 52k dofs): Eigen 0.12 s   CHOLMOD supernodal 0.34 s
// so CHOLMOD is used for the displacement block only; the phase-field block
// stays on Eigen, where it is faster.
using SpLDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
#ifdef PFM_USE_CHOLMOD
using SpCholU = Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>>;
constexpr const char* kSpdSolverName = "u: CHOLMOD supernodal LLT, phi: Eigen SimplicialLDLT";
#else
using SpCholU = SpLDLT;
constexpr const char* kSpdSolverName = "u: Eigen SimplicialLDLT, phi: Eigen SimplicialLDLT";
#endif

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
    static SpCholU ldlt;
    static bool    ldlt_analyzed = false;
    if (!ldlt_analyzed)
        std::cout << "[solver] SPD sub-solver: " << kSpdSolverName << "\n";

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

// ---------------------------------------------------------------------------
// Phase-field sub-solve with the box constraint  lo <= phi <= 1  (lo = phi_n)
// by the primal-dual active set method. See PhiBounds in Solver.h.
//
// With u frozen the phi residual is AFFINE in phi:  R(x) = R0 + K (x - phi0)
// with K constant (H_eff = max(H, psi+(u)) does not depend on phi), so the
// system is assembled ONCE and every pass is one factorisation + solve.
//
// Fixed nodes are imposed like Dirichlet DOFs, in place (identity row and
// column, prescribed increment moved to the right-hand side), on a copy of
// the assembled matrix -- the sparsity pattern is unchanged, so the once-only
// symbolic analysis is reused for every pass.
//
// The active set is warm-started from the previous call: between staggered
// sweeps (and load steps) it barely changes, so most calls settle in 1-2
// passes. The update rule corrects any stale entry.
// ---------------------------------------------------------------------------
// Active-set state per node, kept between calls (warm start) and read by the
// staggered sweep: 0 free, -1 held at the lower bound phi_n, +1 held at 1.
std::vector<signed char> g_phi_active;

SolverResult subsolvePhiActiveSet(const FemInput&        d,
                                  const Eigen::VectorXd& u,
                                  Eigen::VectorXd&       phi,
                                  const HistoryField&    history,
                                  const SolverSettings&  settings,
                                  const Eigen::VectorXd* phi_lower)
{
    const int n = d.npoin;
    SolverResult res;

    static SpLDLT ldltA;
    static bool   ldltA_analyzed = false;
    std::vector<signed char>& state = g_phi_active;
    const bool fresh_state = (static_cast<int>(state.size()) != n);
    if (fresh_state) state.assign(n, 0);

    PhiSystem sys = assemblePhi_System(d, u, phi, history);
    const Eigen::SparseMatrix<double> K0   = sys.K;   // assembler-owned: copy
    const Eigen::VectorXd             R0   = sys.R;
    const Eigen::VectorXd             phi0 = phi;
    res.initial_residual = R0.norm();

    // Bounds: lo = clamp(phi_n, 0, 1), hi = 1.
    Eigen::VectorXd lo = Eigen::VectorXd::Zero(n);
    if (phi_lower) {
        if (phi_lower->size() != n)
            throw std::runtime_error("subsolvePhiActiveSet: phi_lower size != npoin");
        lo = phi_lower->cwiseMax(0.0).cwiseMin(1.0);
    }
    const double hi = 1.0;

    // Tolerances: x in phi units; R relative to the stiffness scale (R for a
    // unit change of phi at one node is ~ the diagonal of K).
    double kmax = 0.0;
    for (int j = 0; j < K0.outerSize(); ++j)
        for (Eigen::SparseMatrix<double>::InnerIterator it(K0, j); it; ++it)
            if (it.row() == it.col()) kmax = std::max(kmax, std::abs(it.value()));
    const double tol_x = 1e-12;
    const double tol_R = 1e-12 * std::max(kmax, 1e-300);

    // First call of the run: start with every node that sits ON its lower
    // bound held there (at t = 0 that is all of them). The update rule
    // releases the ones that want to grow. This matters for AT1: with no
    // damage yet (H = 0) its phi matrix is only the Laplacian term, which is
    // singular without any fixed node -- an all-free first pass would fail
    // to factorise.
    if (fresh_state)
        for (int i = 0; i < n; ++i)
            if (phi0(i) <= lo(i) + tol_x) state[i] = -1;

    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd x = phi0, Rx = R0, rhs, dfix(n);
    std::vector<char> mask(n);
    bool settled = false;
    int  pass    = 0;

    for (pass = 0; pass < settings.phi_active_set_max_iter; ++pass) {
        // Fixed values: increment that puts each fixed node on its bound.
        for (int i = 0; i < n; ++i) {
            mask[i] = (state[i] != 0);
            dfix(i) = (state[i] < 0) ? lo(i) - phi0(i)
                    : (state[i] > 0) ? hi    - phi0(i) : 0.0;
        }
        // K delta = -R0 with the fixed DOFs eliminated in place.
        K   = K0;
        rhs = -R0;
        for (int j = 0; j < K.outerSize(); ++j) {
            const bool jc = mask[j] != 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
                const int i = static_cast<int>(it.row());
                if (jc) {
                    if (i == j) { it.valueRef() = 1.0; continue; }
                    if (!mask[i]) rhs(i) -= it.value() * dfix(j);
                    it.valueRef() = 0.0;
                } else if (mask[i]) {
                    it.valueRef() = 0.0;
                }
            }
        }
        for (int i = 0; i < n; ++i) if (mask[i]) rhs(i) = dfix(i);

        const Eigen::VectorXd delta =
            factorizeAndSolve(ldltA, ldltA_analyzed, K, rhs, "subsolvePhi(active set)");
        x  = phi0 + delta;
        for (int i = 0; i < n; ++i)               // exact bound values
            if (state[i] < 0) x(i) = lo(i); else if (state[i] > 0) x(i) = hi;
        Rx = R0 + K0 * (x - phi0);                // residual of the ORIGINAL system

        // Update the whole set at once.
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            signed char ns = state[i];
            if (state[i] == 0) {
                if      (x(i) < lo(i) - tol_x) ns = -1;
                else if (x(i) > hi    + tol_x) ns = +1;
            } else if (state[i] < 0) {
                if (Rx(i) < -tol_R) ns = 0;       // equations want phi higher
            } else {
                if (Rx(i) >  tol_R) ns = 0;       // equations want phi lower
            }
            if (ns != state[i]) { state[i] = ns; changed = true; }
        }
        if (!changed) { settled = true; ++pass; break; }
    }

    phi = x;
    double r2 = 0.0;
    for (int i = 0; i < n; ++i) {
        if (state[i] == 0) r2 += Rx(i) * Rx(i);
        else if (state[i] < 0) ++res.n_at_lower;
        else ++res.n_at_upper;
    }
    res.iters_used     = pass;                    // active-set passes
    res.final_residual = std::sqrt(r2);           // free-node residual
    res.converged      = settled;
    if (!settled && settings.verbose)
        std::cout << "  [phi-bounds] active set did not settle in "
                  << settings.phi_active_set_max_iter << " passes\n";
    return res;
}

// Newton sub-solve of the PHASE-FIELD block with u held fixed.
// With u (hence psi^+ and the history H) frozen the phase-field residual is
// affine in phi, so this converges in a single iteration; the loop is kept
// generic for robustness. The u-block is frozen (identity).
// With settings.phi_bounds == ActiveSet the box-constrained solve above is
// used instead.
SolverResult subsolvePhi(const FemInput&        d,
                         const Eigen::VectorXd& u,
                         Eigen::VectorXd&       phi,
                         const HistoryField&    history,
                         const SolverSettings&  settings,
                         const Eigen::VectorXd* phi_lower = nullptr)
{
    if (settings.phi_bounds == PhiBounds::ActiveSet)
        return subsolvePhiActiveSet(d, u, phi, history, settings, phi_lower);

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
                                const SolverSettings&  settings,
                                const Eigen::VectorXd* phi_lower = nullptr)
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
        const SolverResult rp = subsolvePhi(d, u, phi, history, settings, phi_lower);

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
        // Nodes held at a phi bound carry the constraint's reaction (the
        // Lagrange multiplier) in their residual, by design -- exclude them
        // from the convergence norm exactly like Dirichlet reactions.
        if (settings.phi_bounds == PhiBounds::ActiveSet &&
            static_cast<int>(g_phi_active.size()) == npoin)
            for (int i = 0; i < npoin; ++i)
                if (g_phi_active[i] != 0) R(n_u + i) = 0.0;
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
                      << ";  phi: " << rp.iters_used
                      << (settings.phi_bounds == PhiBounds::ActiveSet
                              ? " active-set passes" : " it")
                      << (rp.converged ? "" : ", NOT conv");
            if (settings.phi_bounds == PhiBounds::ActiveSet)
                std::cout << ";  at phi_n: " << rp.n_at_lower
                          << ", at 1: " << rp.n_at_upper;
            std::cout << ")\n";
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
                       const SolverSettings&  settings,
                       const Eigen::VectorXd* phi_lower)
{
    const int npoin = d.npoin;
    if (u.size()   != 2 * npoin)
        throw std::runtime_error("solveStep: u size != 2*npoin");
    if (phi.size() != npoin)
        throw std::runtime_error("solveStep: phi size != npoin");

    if (settings.scheme == SolverScheme::Staggered)
        return solveStepStaggered(d, u, phi, history, load_factor, settings,
                                  phi_lower);
    if (settings.phi_bounds != PhiBounds::None)
        throw std::runtime_error(
            "solveStep: phi bounds (active set) need the staggered scheme");
    return solveStepMonolithic(d, u, phi, history, load_factor, settings);
}

}  // namespace pfm
