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
    // 2. Build a new K with constrained rows/columns replaced by identity
    //    rows/columns, applying the RHS correction for the displacement
    //    increment.  We use a triplet rebuild so the resulting pattern is
    //    exactly what SparseLU needs (no structural zeros, no ghost entries).
    // -----------------------------------------------------------------------
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(K.nonZeros()));

    for (int j = 0; j < K.outerSize(); ++j) {
        const bool j_constrained = (mask[j] != 0);
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
            const int    i = static_cast<int>(it.row());
            const double v = it.value();
            if (mask[i]) { continue; }                  // constrained row: skip
            if (j_constrained) {                        // free row, constrained col
                R(i) += v * delta_prescribed[j];        // RHS correction
                continue;
            }
            trips.emplace_back(i, j, v);                // free-free entry
        }
    }
    // Constrained DOFs: identity diagonal + prescribed increment in RHS.
    for (int r = 0; r < total; ++r) {
        if (!mask[r]) continue;
        trips.emplace_back(r, r, 1.0);
        R(r) = -delta_prescribed[r];
    }

    Eigen::SparseMatrix<double> K_new(total, total);
    K_new.setFromTriplets(trips.begin(), trips.end());
    K_new.makeCompressed();
    K.swap(K_new);
}

}  // namespace pfm
