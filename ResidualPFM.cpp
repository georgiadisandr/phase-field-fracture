#include "ResidualPFM.h"

#include "Profiling.h"

#include <stdexcept>
#include <vector>

namespace pfm {

// Residual-only assembly. Loops the elements calling the shared
// elementSystem() with want_K = false, so the tangent block products are
// skipped but the residual physics is identical to assembleGlobalSystem().
Eigen::VectorXd
assembleGlobalResidual(const FemInput&        d,
                       const Eigen::VectorXd& u_global,
                       const Eigen::VectorXd& phi_global,
                       const HistoryField&    history)
{
    const int npoin = d.npoin;
    const int total = 3 * npoin;                // 2*npoin + npoin

    if (u_global.size()   != 2 * npoin)
        throw std::runtime_error("assembleGlobalResidual: u_global size != 2*npoin");
    if (phi_global.size() != npoin)
        throw std::runtime_error("assembleGlobalResidual: phi_global size != npoin");
    if (static_cast<int>(history.size()) != d.nelem)
        throw std::runtime_error("assembleGlobalResidual: history size != nelem");

    prof::Scope _t(prof::ElemAssembly);

    Eigen::VectorXd R = Eigen::VectorXd::Zero(total);

    std::vector<int> nodes;
    Eigen::VectorXd  u_elem, phi_elem, re;
    Eigen::MatrixXd  Ke;                        // unused (want_K = false)

    for (int e = 0; e < d.nelem; ++e) {
        nodes.assign(d.conn.begin() + d.offset[e],
                     d.conn.begin() + d.offset[e + 1]);
        const int nnode = static_cast<int>(nodes.size());
        const int ndofu = 2 * nnode;

        u_elem.resize(ndofu);
        phi_elem.resize(nnode);
        for (int i = 0; i < nnode; ++i) {
            const int node = nodes[i];
            u_elem(2 * i)     = u_global(2 * node);
            u_elem(2 * i + 1) = u_global(2 * node + 1);
            phi_elem(i)       = phi_global(node);
        }

        elementSystem(nodes, d.matno[e], d, u_elem, phi_elem,
                      history[e], /*want_K=*/false, Ke, re);

        // Scatter the element residual into the global vector.
        for (int i = 0; i < nnode; ++i) {
            const int node = nodes[i];
            R(2 * node)         += re(2 * i);
            R(2 * node + 1)     += re(2 * i + 1);
            R(2 * npoin + node) += re(ndofu + i);
        }
    }

    return R;
}

}  // namespace pfm
