#include "History.h"

#include "ConstitutiveModel.h"
#include "ElementUtils.h"
#include "Jacob2.h"
#include "Profiling.h"
#include "ShapeFunc.h"
#include "StiffnessPFM.h"      // for pfm::MatParams

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pfm {

HistoryField makeHistoryField(const FemInput& d)
{
    HistoryField history(d.nelem);
    for (int e = 0; e < d.nelem; ++e) {
        const int nnode = d.nnode_of(e);
        const int ngp   = static_cast<int>(
                              fem::gaussPoints(nnode, d.ngaus).size());
        history[e].assign(ngp, 0.0);
    }
    return history;
}

void updateHistory(const FemInput&        d,
                   const Eigen::VectorXd& u,
                   HistoryField&          history)
{
    if (u.size() != 2 * d.npoin)
        throw std::runtime_error("updateHistory: u size != 2*npoin");
    if (static_cast<int>(history.size()) != d.nelem)
        throw std::runtime_error("updateHistory: history size != nelem");

    prof::Scope _t(prof::History);

    const int ndime = d.ndime;
    std::vector<int> nodes;

    for (int e = 0; e < d.nelem; ++e) {
        nodes.assign(d.conn.begin() + d.offset[e],
                     d.conn.begin() + d.offset[e + 1]);
        const int nnode = static_cast<int>(nodes.size());

        // History only needs the strain-driven psi^+; the per-material cache
        // supplies the constitutive matrices. MatParams (Gc, l0, k) are not
        // used here -- they appear only in the residual / tangent assembly.
        const MatCache& mc = d.mat_caches[d.matno[e]];

        const fem::RefElement& ref = d.ref_elems.get(nnode);

        // Nodal coordinates and element displacement.
        fem::DimNodeMat elcod(ndime, nnode);
        Eigen::VectorXd u_elem(2 * nnode);
        for (int i = 0; i < nnode; ++i) {
            const int node = nodes[i];
            for (int dim = 0; dim < ndime; ++dim)
                elcod(dim, i) = d.coord(node, dim);
            u_elem(2 * i)     = u(2 * node);
            u_elem(2 * i + 1) = u(2 * node + 1);
        }

        if (static_cast<int>(history[e].size()) != ref.ngp)
            throw std::runtime_error(
                "updateHistory: history[e] size != number of Gauss points");

        fem::DimNodeMat cartd;
        fem::BuMat      Bu;
        double          djacb = 0.0;

        for (int g = 0; g < ref.ngp; ++g) {
            const std::size_t gi = static_cast<std::size_t>(g);

            fem::jacob2Fast(elcod, ref.dNr[gi], nnode, ndime, cartd, djacb);
            fem::buildBu(cartd, nnode, Bu);

            const Eigen::Vector3d eps = Bu * u_elem;
            const EnergySplit     sp  = energySplit(eps, mc, d.split, d.hybrid);

            // Irreversibility: H can only grow.
            history[e][g] = std::max(history[e][g], sp.psi_plus);
        }
    }
}

}  // namespace pfm
