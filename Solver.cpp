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
// Dissipation control: the phase-field step of one sweep
//
// With phi^k frozen the u-solve gave u_hat (lambda = 1). Find lambda such that
// the phase field solved with u = lambda * u_hat raises the crack energy by
// exactly dD:
//
//     g(lambda) = Gamma( Phi(lambda * u_hat) ) - Gamma_n - dD = 0
//
// Phi is the (affine, one-iteration) phi-solve with H = max(H_n, psi+(u)).
// For a FIXED u_hat this is a consistent scalar function, increasing in lambda
// (more load -> larger H -> more damage), so a bracketing root-finder is safe.
// Each evaluation costs one phi-solve -- the cheap block; the expensive
// u-solve is done once per sweep. Pinning Gamma inside every sweep is what
// keeps the crack from running away past the peak (with lambda fixed while
// the sweeps settle, the staggered iteration jumps to the fully-cracked
// branch and the target cannot be met).
//
// On return phi = Phi(lam * u_hat) for the returned lam.
// ===========================================================================
struct DissPhiStep {
    SolverResult rp;           // phi-solve result at the returned lambda
    // Stopped on the dlambda_max cap with |g| > tol:
    //   +1  RELOAD cap: g < 0 at lambda_n + dlambda_max -- even the largest
    //       allowed load increase dissipates less than dD (elastic reloading,
    //       broken specimen). Acceptable: the step then acts like a
    //       displacement-controlled step of size dissipation_max_du.
    //   -1  UNLOAD cap: g > 0 at lambda_n - dlambda_max -- even the largest
    //       allowed unloading dissipates MORE than dD. Never acceptable: the
    //       crack would run further than the step allows. The step is
    //       rejected and the driver halves dtau.
    int  cap_side = 0;
    int  evals  = 0;           // phi-solves used
    int  bisect = 0;           // bisection fallbacks
};

DissPhiStep dissipationPhiStep(const FemInput&        d,
                               const Eigen::VectorXd& u_hat,
                               Eigen::VectorXd&       phi,
                               const HistoryField&    history,
                               const SolverSettings&  settings,
                               const DissipationControl& dc,
                               double&                lam,
                               double&                slope_mem,
                               const Eigen::VectorXd* phi_lower)
{
    const double lmax = dc.lambda_n + dc.dlambda_max;
    const double lmin = std::max(dc.lambda_n - dc.dlambda_max, 0.05 * dc.lambda_n);
    // Re-target tightly in every sweep (much tighter than the acceptance
    // tolerance dc.tol): letting g drift inside the acceptance band leaves
    // lambda frozen, and at the peak the sweeps then creep toward the
    // runaway branch instead of converging.
    const double gtol = 1e-7 * dc.dD;
    const Eigen::VectorXd phi0 = phi;

    DissPhiStep     out;
    Eigen::VectorXd ut, ph;
    const auto clampL = [&](double L) { return std::min(std::max(L, lmin), lmax); };
    const auto eval   = [&](double L) {
        ph = phi0;
        ut = L * u_hat;
        out.rp = subsolvePhi(d, ut, ph, history, settings, phi_lower);
        ++out.evals;
        return computeEnergy(d, ut, ph).fracture - dc.Gamma_n - dc.dD;
    };

    bool   have_lo = false, have_hi = false;
    double lo = 0.0, glo = 0.0, hi = 0.0, ghi = 0.0;
    const auto note = [&](double L, double g) {
        if (g < 0.0) { if (!have_lo || L > lo) { lo = L; glo = g; have_lo = true; } }
        else         { if (!have_hi || L < hi) { hi = L; ghi = g; have_hi = true; } }
    };

    double x  = clampL(lam);
    double gx = eval(x);
    note(x, gx);
    double xp = 0.0, gp = 0.0;
    bool   have_p = false;
    constexpr int kMaxEval = 30;

    while (std::abs(gx) > gtol && out.evals < kMaxEval && out.rp.converged) {
        // On the cap with g still asking to go beyond it: target unreachable.
        if (gx < 0.0 && x >= lmax) { out.cap_side = +1; break; }
        if (gx > 0.0 && x <= lmin) { out.cap_side = -1; break; }
        double c;
        if (have_lo && have_hi) {
            // regula falsi on the bracket, bisection when it hugs an end
            c = lo - glo * (hi - lo) / (ghi - glo);
            const double w = hi - lo;
            if (!(c > lo + 0.05 * w && c < hi - 0.05 * w)) {
                c = 0.5 * (lo + hi);
                ++out.bisect;
            }
        } else {
            double sl = 0.0;
            if (have_p && std::abs(x - xp) > 1e-15) sl = (gx - gp) / (x - xp);
            if (!(sl > 0.0)) sl = slope_mem;
            c = (sl > 0.0) ? x - gx / sl
                           : x + (gx < 0.0 ? 0.25 : -0.25) * dc.dlambda_max;
            c = clampL(c);
        }
        if (std::abs(c - x) <= 1e-14 * std::max(1.0, std::abs(x))) break;
        xp = x;  gp = gx;  have_p = true;
        x  = c;  gx = eval(x);
        note(x, gx);
        if (std::abs(x - xp) > 1e-15) {
            const double sl = (gx - gp) / (x - xp);
            if (sl > 0.0) slope_mem = sl;
        }
    }
    lam = x;
    phi = ph;
    return out;
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
                                DissipationControl*    dc = nullptr,
                                const Eigen::VectorXd* phi_lower = nullptr)
{
    const int npoin = d.npoin;
    const int n_u   = 2 * npoin;

    SolverResult res;

    const Eigen::VectorXd F_ext = assembleExternalForce(d);
    const bool any_neumann      = (F_ext.array() != 0.0).any();

    // Dissipation control (dc != nullptr): the load factor is an unknown of
    // the step (see DissipationControl in Solver.h). Without dc, lam stays
    // equal to load_factor and nothing below changes.
    double lam       = load_factor;  // lambda that u currently corresponds to
    double slope_mem = 0.0;          // last dGamma/dlambda seen (root-finder)
    int    unload_cap_streak = 0;    // consecutive sweeps stopped on the unload cap
    if (dc) {
        if (any_neumann)
            throw std::runtime_error(
                "dissipation control: Neumann tractions / point loads are not "
                "supported (pure Dirichlet loading only)");
        lam             = dc->lambda_pred;
        slope_mem       = dc->slope;
        dc->n_bisect    = 0;
        dc->lambda      = lam;
        dc->dGamma      = 0.0;
        dc->capped      = false;
        dc->unload_capped = false;
        dc->n_phi_solves = 0;
    }

    if (settings.verbose) {
        if (dc)
            std::cout << "[solver] STAGGERED, DISSIPATION CONTROL  dD = " << dc->dD
                      << ",  lambda_n = " << dc->lambda_n
                      << ",  predictor = " << dc->lambda_pred
                      << "  (max " << settings.max_staggered << " sweeps)\n";
        else
            std::cout << "[solver] STAGGERED  load_factor = " << load_factor
                      << "  (max " << settings.max_staggered << " sweeps)\n";
    }

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
        // (a) displacement sub-solve with phi frozen. Under dissipation
        //     control it is solved at lambda = 1 (u_hat) and scaled, so u is
        //     ALWAYS consistent with dc->lambda on return.
        const SolverResult ru =
            dc ? subsolveU(d, u, phi, history, 1.0, F_ext, false, settings)
               : subsolveU(d, u, phi, history, load_factor,
                           F_ext, any_neumann, settings);
        Eigen::VectorXd u_hat;
        if (dc) {
            u_hat      = u;
            u         *= lam;
            dc->lambda = lam;
        }

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

        // (b) phase-field sub-solve with u frozen. Under dissipation control
        //     lambda is chosen here so that this phi-solve meets the
        //     crack-energy target (see dissipationPhiStep).
        SolverResult rp;
        bool         sweep_capped = false;     // reload cap only (acceptable)
        int          phi_evals    = 1;
        if (dc) {
            const DissPhiStep o = dissipationPhiStep(d, u_hat, phi, history,
                                                     settings, *dc, lam, slope_mem,
                                                     phi_lower);
            rp            = o.rp;
            sweep_capped  = (o.cap_side > 0);
            unload_cap_streak = (o.cap_side < 0) ? unload_cap_streak + 1 : 0;
            phi_evals     = o.evals;
            u             = lam * u_hat;
            dc->lambda    = lam;
            dc->n_bisect += o.bisect;
            dc->n_phi_solves += o.evals;
            // dD not reachable even at the largest allowed unloading, in 3
            // sweeps in a row: the step is too large for the snap-back here.
            // Reject it now (instead of burning max_staggered sweeps) so the
            // driver rolls back and halves dtau.
            if (unload_cap_streak >= 3) {
                dc->unload_capped  = true;
                res.iters_used     = sweep + 1;
                res.final_residual = rp.final_residual;
                if (sweep == 0) res.initial_residual = ru.initial_residual;
                res.converged      = false;
                if (settings.verbose)
                    std::cout << "  [solver] dissipation step rejected at sweep "
                              << sweep << ": even the largest allowed unloading "
                                 "(dissipation_max_du) dissipates more than dD "
                                 "-- step too large here\n";
                return res;
            }
        } else {
            rp = subsolvePhi(d, u, phi, history, settings, phi_lower);
        }

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
            // Dissipation control: the u-solve ran at lambda = 1, so its
            // initial residual measures the BC jump (1 - lambda_n), not this
            // step. And near the peak the step is driven by crack growth at
            // an almost constant lambda, so a BC-jump scale would make the
            // relative tolerance absurdly strict. Use the coupled residual
            // after the first sweep as the reference instead.
            if (dc) {
                R0                   = std::max(Rn, settings.tol_abs);
                res.initial_residual = R0;
            }
        }

        res.iters_used     = sweep + 1;
        res.final_residual = Rn;

        // Total energy of the current iterate E^k = E(u^k, phi^k). The natural
        // scalar for monitoring staggered convergence (decreases monotonically
        // toward the minimizer). Computed only when something consumes it
        // (verbose print or the CSV log) to avoid the extra mesh pass otherwise.
        const bool want_energy =
            settings.verbose || !settings.energy_csv.empty() || use_gamma
            || dc != nullptr;
        EnergyParts E;
        if (want_energy) E = computeEnergy(d, u, phi);

        if (energy_csv_open && energy_csv) {
            energy_csv << lam << ',' << (sweep + 1) << ','
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
            if (dc) {
                const auto prec = std::cout.precision();
                std::cout << "   lambda = " << std::setprecision(10) << lam
                          << std::setprecision(4) << "   dGamma/dD = "
                          << (E.fracture - dc->Gamma_n) / dc->dD
                          << std::setprecision(prec)
                          << "   phi-solves = " << phi_evals
                          << (sweep_capped ? " [reload cap]"
                              : unload_cap_streak > 0 ? " [UNLOAD cap]" : "");
            }
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
        // Dissipation control: the crack-energy constraint must hold too (it
        // does by construction unless the root-finder hit the step cap).
        // Only the RELOAD cap may be accepted with |g| > tol (sweep_capped);
        // an unload-capped sweep can never converge here.
        if (dc) {
            const double g    = E.fracture - dc->Gamma_n - dc->dD;
            const bool   g_ok = std::abs(g) <= dc->tol * dc->dD;
            dc->dGamma        = E.fracture - dc->Gamma_n;
            dc->capped        = converged && !g_ok && sweep_capped;
            converged         = converged && (g_ok || sweep_capped);
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

// ===========================================================================
// Monolithic dissipation control (energy-balance constraint)
//
// Unknowns: x = [u_free | phi] and the load factor lambda; the prescribed
// displacements follow the load, u_p = lambda * u_hat_p (u_hat_p = the
// Dirichlet values of the TOML at full load). One extra equation fixes the
// energy released in the step (Gutierrez 2004; Verhoosel et al. 2009):
//
//     g(x, lambda) = 1/2 ( P_n lambda - P(x, lambda) lambda_n ) - dD = 0
//
// with P = u_hat_p . f_int,p the generalised force conjugate to lambda (the
// reaction at the loaded DOFs projected on the load pattern). At equilibrium
// P lambda = 2 Psi_el, so this is external work minus stored energy.
//
// Newton on the bordered system, solved with ONE factorisation and two
// back-substitutions (Batoz & Dhatt):
//
//     [ K    r_lam ] [dx  ]     [R]        r_lam = dR/dlambda = K_{:,p} u_hat_p
//     [ c^T  c_lam ] [dlam] = - [g]        c     = dg/dx      = -lambda_n/2 K_{p,:}^T u_hat_p
//                                          c_lam = dg/dlambda = P_n/2 - lambda_n/2 u_hat_p^T K_pp u_hat_p
//
//     K dx_R = -R,  K dx_l = -r_lam,  dlam = -(g + c.dx_R)/(c.dx_l + c_lam),
//     dx = dx_R + dlam dx_l,  du_p = dlam u_hat_p
//
// (K is the full coupled tangent with the Dirichlet rows/columns replaced by
// identity; r_lam, c and c_lam come from the UNMODIFIED K.)
//
// The update is damped by a backtracking line search on the merit
//     m = sqrt( (|R_free|/R_ref)^2 + (g/dD)^2 ).
// lambda is kept within lambda_n +- dlambda_max; the cap rules match the
// staggered version (reload cap accepted, unload cap rejected).
// Does not need the hybrid formulation. AT2 without phi bounds only.
// ===========================================================================
SolverResult solveStepMonolithicDissipation(const FemInput&        d,
                                            Eigen::VectorXd&       u,
                                            Eigen::VectorXd&       phi,
                                            const HistoryField&    history,
                                            DissipationControl&    dc,
                                            const SolverSettings&  settings)
{
    const int npoin = d.npoin;
    const int n_u   = 2 * npoin;
    const int ntot  = 3 * npoin;
    SolverResult res;

    static SpLU lu;
    static bool lu_analyzed = false;

    // Load pattern w (full length; nonzero only on loaded Dirichlet DOFs) and
    // the constrained-DOF mask.
    Eigen::VectorXd   w = Eigen::VectorXd::Zero(ntot);
    std::vector<char> cmask(ntot, 0);
    for (int i = 0; i < d.nvfix; ++i)
        for (int k = 0; k < d.ndofn; ++k)
            if (d.iffix[i][k] != 0) {
                const int gdof = 2 * d.nofix[i] + k;
                cmask[gdof] = 1;
                w(gdof)     = d.fixed[i][k];
            }
    const auto setPrescribed = [&](Eigen::VectorXd& uu, double L) {
        for (int i = 0; i < d.nvfix; ++i)
            for (int k = 0; k < d.ndofn; ++k)
                if (d.iffix[i][k] != 0)
                    uu(2 * d.nofix[i] + k) = L * d.fixed[i][k];
    };
    const auto freeNorm = [&](const Eigen::VectorXd& R) {
        double s2 = 0.0;
        for (int i = 0; i < ntot; ++i) if (!cmask[i]) s2 += R(i) * R(i);
        return std::sqrt(s2);
    };
    const auto gOf = [&](double L, double P) {
        return 0.5 * (dc.P_n * L - P * dc.lambda_n) - dc.dD;
    };

    const double lmax = dc.lambda_n + dc.dlambda_max;
    const double lmin = std::max(dc.lambda_n - dc.dlambda_max, 0.05 * dc.lambda_n);

    double lam = std::min(std::max(dc.lambda_pred, lmin), lmax);
    setPrescribed(u, lam);
    dc.capped = false;  dc.unload_capped = false;  dc.lambda = lam;

    if (settings.verbose)
        std::cout << "[solver] MONOLITHIC, DISSIPATION CONTROL (energy balance)  dD = "
                  << dc.dD << ",  lambda_n = " << dc.lambda_n
                  << ",  predictor = " << lam
                  << "  (max " << settings.max_iter << " Newton its)\n";

    double R_ref = -1.0;
    int    cap_side = 0;          // +1 lambda on lmax, -1 on lmin
    const double eps_cap = 1e-12 * std::max(1.0, std::abs(lmax));

    for (int iter = 0; iter < settings.max_iter; ++iter) {
        GlobalSystem sys = assembleGlobalSystem(d, u, phi, history);
        const double P  = w.dot(sys.R);
        const double g  = gOf(lam, P);
        const double Rn = freeNorm(sys.R);
        if (iter == 0) {
            double rr = 0.0;
            for (int i = 0; i < ntot; ++i) if (cmask[i]) rr += sys.R(i) * sys.R(i);
            R_ref = std::max({Rn, 1e-3 * std::sqrt(rr), settings.tol_abs});
            res.initial_residual = Rn;
        }
        const double tolR = std::max(settings.tol_abs, settings.tol_rel * R_ref);
        const bool   R_ok = (Rn < tolR);
        const bool   g_ok = std::abs(g) <= dc.tol * dc.dD;
        cap_side = (std::abs(lam - lmax) <= eps_cap) ? +1
                 : (std::abs(lam - lmin) <= eps_cap) ? -1 : 0;

        res.iters_used     = iter;
        res.final_residual = Rn;
        dc.lambda          = lam;
        dc.dGamma          = g + dc.dD;     // released energy so far

        if (settings.verbose) {
            const auto prec = std::cout.precision();
            std::cout << "  iter " << iter << "  |R| = " << Rn
                      << "   |R|/|Rref| = " << Rn / R_ref
                      << std::setprecision(10) << "   lambda = " << lam
                      << std::setprecision(4) << "   dD_bal/dD = " << (g + dc.dD) / dc.dD
                      << std::setprecision(prec)
                      << (cap_side > 0 ? " [reload cap]" : cap_side < 0 ? " [unload cap]" : "")
                      << "\n";
        }

        if (R_ok && g_ok) { res.converged = true; break; }
        if (R_ok && cap_side > 0 && g < 0.0) {           // reload cap: accept
            dc.capped = true;  res.converged = true;  break;
        }
        if (R_ok && cap_side < 0 && g > 0.0) {           // unload cap: reject
            dc.unload_capped = true;
            if (settings.verbose)
                std::cout << "  [solver] dissipation step rejected: even the "
                             "largest allowed unloading (dissipation_max_du) "
                             "releases more than dD -- step too large here\n";
            res.converged = false;
            return res;
        }

        // Border terms from the UNMODIFIED tangent.
        const Eigen::VectorXd Kw  = sys.K * w;
        const Eigen::VectorXd KTw = sys.K.transpose() * w;
        const double c_lam = 0.5 * dc.P_n - 0.5 * dc.lambda_n * w.dot(Kw);
        Eigen::VectorXd r_lam = Kw, c = -0.5 * dc.lambda_n * KTw;
        for (int i = 0; i < ntot; ++i) if (cmask[i]) { r_lam(i) = 0.0; c(i) = 0.0; }

        // u_p already equals lam*u_hat_p, so the prescribed increment is 0:
        // applyDirichlet only replaces the constrained rows/columns.
        applyDirichlet(sys.K, sys.R, d, u, lam);
        const Eigen::VectorXd dxR =
            factorizeAndSolve(lu, lu_analyzed, sys.K, -sys.R, "monolithic dissipation");
        Eigen::VectorXd dxL;
        {
            prof::Scope _t(prof::Solve);
            dxL = lu.solve(-r_lam);
        }
        if (lu.info() != Eigen::Success || !dxL.allFinite())
            throw std::runtime_error("monolithic dissipation: second solve failed");

        const double den = c.dot(dxL) + c_lam;
        double dlam = (std::abs(den) > 1e-300) ? -(g + c.dot(dxR)) / den : 0.0;
        if (!std::isfinite(dlam)) dlam = 0.0;
        if (lam + dlam > lmax) dlam = lmax - lam;      // keep lambda within the cap
        if (lam + dlam < lmin) dlam = lmin - lam;
        const Eigen::VectorXd dx = dxR + dlam * dxL;

        // Backtracking line search on the merit function.
        const double m0 = std::hypot(Rn / R_ref, g / dc.dD);
        double alpha = 1.0;
        Eigen::VectorXd u_t, phi_t;
        double lam_t = lam;
        for (int ls = 0; ls < 6; ++ls) {
            u_t   = u   + alpha * dx.head(n_u);
            phi_t = phi + alpha * dx.tail(npoin);
            lam_t = lam + alpha * dlam;
            setPrescribed(u_t, lam_t);
            const Eigen::VectorXd Rt = assembleGlobalResidual(d, u_t, phi_t, history);
            const double mt = std::hypot(freeNorm(Rt) / R_ref,
                                         gOf(lam_t, w.dot(Rt)) / dc.dD);
            if (mt <= (1.0 - 1e-4 * alpha) * m0) break;
            if (ls < 5) alpha *= 0.5;                  // last try is accepted as is
        }
        if (settings.verbose && alpha < 1.0)
            std::cout << "    line search: alpha = " << alpha << "\n";
        u = u_t;  phi = phi_t;  lam = lam_t;
        res.iters_used = iter + 1;
    }

    dc.lambda = lam;
    if (!res.converged && settings.verbose)
        std::cout << "[solver] monolithic dissipation step did NOT converge in "
                  << settings.max_iter << " iterations\n";
    else if (settings.verbose)
        std::cout << "[solver] converged in " << res.iters_used << " iterations"
                  << (dc.capped ? " (on the reload cap)" : "") << "\n";
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
                                  nullptr, phi_lower);
    if (settings.phi_bounds != PhiBounds::None)
        throw std::runtime_error(
            "solveStep: phi bounds (active set) need the staggered scheme");
    return solveStepMonolithic(d, u, phi, history, load_factor, settings);
}

SolverResult solveStepDissipation(const FemInput&        d,
                                  Eigen::VectorXd&       u,
                                  Eigen::VectorXd&       phi,
                                  const HistoryField&    history,
                                  DissipationControl&    dc,
                                  const SolverSettings&  settings,
                                  const Eigen::VectorXd* phi_lower)
{
    const int npoin = d.npoin;
    if (u.size()   != 2 * npoin)
        throw std::runtime_error("solveStepDissipation: u size != 2*npoin");
    if (phi.size() != npoin)
        throw std::runtime_error("solveStepDissipation: phi size != npoin");
    if (dc.lambda_n <= 0.0 || dc.dD <= 0.0 || dc.dlambda_max <= 0.0)
        throw std::runtime_error(
            "solveStepDissipation: needs lambda_n > 0, dD > 0 and "
            "dlambda_max > 0");
    if (settings.scheme == SolverScheme::Monolithic) {
        if (settings.phi_bounds != PhiBounds::None)
            throw std::runtime_error(
                "solveStepDissipation: monolithic dissipation control does not "
                "support phi bounds");
        if (dc.P_n <= 0.0)
            throw std::runtime_error(
                "solveStepDissipation: monolithic version needs P_n > 0");
        return solveStepMonolithicDissipation(d, u, phi, history, dc, settings);
    }
    if (!d.hybrid)
        throw std::runtime_error(
            "solveStepDissipation: the staggered version needs fem.hybrid = true");
    return solveStepStaggered(d, u, phi, history, dc.lambda_pred, settings, &dc,
                              phi_lower);
}

}  // namespace pfm
