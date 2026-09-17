#include "BoundaryConditions.h"

#include "Profiling.h"

#include <stdexcept>
#include <vector>
#include <Eigen/Sparse>

namespace pfm {

void applyDirichlet(Eigen::SparseMatrix<double>& K,
                    Eigen::VectorXd&             R,
                    const FemInput&              d,
                    const Eigen::VectorXd&       u_current,
                    double                       load_factor)
{
    const int npoin = d.npoin;//all node from mesh
    const int total = static_cast<int>(K.rows()); //size of all field variables 

    if (K.cols() != total)
        throw std::runtime_error("applyDirichlet: K must be square");
    if (R.size() != total)
        throw std::runtime_error("applyDirichlet: R size != K rows");
    if (u_current.size() != 2 * npoin)
        throw std::runtime_error("applyDirichlet: u_current size != 2*npoin");

    prof::Scope _t(prof::Dirichlet);

    // -----------------------------------------------------------------------
    // 1. Gather constraints. mask[r] == true means dof r is constrained.
    //    delta_prescribed[r] is the increment we must impose on dof r.
    // -----------------------------------------------------------------------
    std::vector<char>   mask(total, 0);              // 0/1, faster than vector<bool>
    std::vector<double> delta_prescribed(total, 0.0);

    for (int i = 0; i < d.nvfix; ++i) {
        const int node = d.nofix[i];
        for (int k = 0; k < d.ndofn; ++k) {
            if (d.iffix[i][k] == 0) continue;//jump to next itteration

            const int    gdof    = 2 * node + k;     // u-dof in global numbering
            const double v_target = load_factor * d.fixed[i][k];
            mask[gdof]             = 1;
            delta_prescribed[gdof] = v_target - u_current(gdof);
        }
    }

    // Early-out if nothing is constrained.
    bool any = false;
    for (char c : mask) if (c) { any = true; break; }
    if (!any) return;

    // -----------------------------------------------------------------------
    // 2. Rewrite the constrained rows/columns IN PLACE.
    //
    //    The sparsity pattern is fixed for the whole run (StiffnessPFM builds
    //    it once), so there is nothing to rebuild here: the constrained
    //    entries are simply overwritten with 0, the constrained diagonals with
    //    1, and the RHS is corrected for the prescribed increment. Every
    //    constrained dof belongs to at least one element, so its diagonal
    //    exists in the pattern; if it does not, that is a bug, not a case to
    //    paper over -- hence the throw.
    //
    //    The result is numerically identical to the old triplet rebuild; it
    //    only keeps the zeroed entries stored (as explicit zeros) instead of
    //    dropping them from the pattern.
    // -----------------------------------------------------------------------
    for (int j = 0; j < K.outerSize(); ++j) {
        const bool j_constrained = (mask[j] != 0);
        bool       diag_seen     = false;

        for (Eigen::SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
            const int i = static_cast<int>(it.row());

            if (j_constrained) {
                if (i == j) {                       // constrained diagonal -> 1
                    it.valueRef() = 1.0;
                    diag_seen     = true;
                    continue;
                }
                if (!mask[i])                       // free row, constrained col
                    R(i) += it.value() * delta_prescribed[j];
                it.valueRef() = 0.0;                // drop the coupling
            } else if (mask[i]) {                   // constrained row, free col
                it.valueRef() = 0.0;
            }
        }

        if (j_constrained && !diag_seen)
            throw std::runtime_error(
                "applyDirichlet: constrained dof has no stored diagonal -- the "
                "matrix pattern does not cover it");
    }

    // Constrained DOFs: prescribed increment on the RHS (K(r,r) is now 1).
    for (int r = 0; r < total; ++r)
        if (mask[r]) R(r) = -delta_prescribed[r];
}

}  // namespace pfm
