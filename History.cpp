#include "History.h"

#include "ConstitutiveModel.h"
#include "ElementUtils.h"
#include "Jacob2.h"
#include "Profiling.h"
#include "ShapeFunc.h"
#include "StiffnessPFM.h"      // for pfm::MatParams

#include <algorithm>
#include <array>
#include <cmath>
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

namespace {

// Distance from point (px, py) to the segment {x0, y0, x1, y1}.
double distanceToSegment(double px, double py, const std::array<double, 4>& s)
{
    const double ax = s[0], ay = s[1];
    const double dx = s[2] - ax, dy = s[3] - ay;
    const double len2 = dx * dx + dy * dy;
    double t = 0.0;
    if (len2 > 0.0)
        t = std::clamp(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0);
    const double qx = ax + t * dx - px;
    const double qy = ay + t * dy - py;
    return std::sqrt(qx * qx + qy * qy);
}

}  // namespace

std::vector<SeedReport>
seedInitialCrackHistory(const FemInput& d, HistoryField& history)
{
    std::vector<SeedReport> reports;
    if (d.initial_cracks.empty()) return reports;
    if (static_cast<int>(history.size()) != d.nelem)
        throw std::runtime_error(
            "seedInitialCrackHistory: history size != nelem");

    for (const auto& crack : d.initial_cracks) {
        SeedReport rep;
        rep.label      = crack.label;
        rep.n_segments = static_cast<int>(crack.segments.size());
        rep.w_min      = HUGE_VAL;
        rep.w_max      = 0.0;

        for (int e = 0; e < d.nelem; ++e) {
            const MatParams mp = MatParams::from(d.props[d.matno[e]]);
            const double Hc = (d.crack_model == CrackModel::AT1)
                                  ? 3.0 * mp.Gc / (16.0 * mp.l0)
                                  : mp.Gc / (2.0 * mp.l0);
            const double w  = (crack.half_width > 0.0) ? crack.half_width
                                                       : 0.25 * mp.l0;

            const int nnode = d.nnode_of(e);
            const fem::RefElement& ref = d.ref_elems.get(nnode);
            const int* nodes = d.conn.data() + d.offset[e];

            for (int g = 0; g < ref.ngp; ++g) {
                const auto& N = ref.N[static_cast<std::size_t>(g)];
                double x = 0.0, y = 0.0;
                for (int i = 0; i < nnode; ++i) {
                    x += N(i) * d.coord(nodes[i], 0);
                    y += N(i) * d.coord(nodes[i], 1);
                }
                double dist = HUGE_VAL;
                for (const auto& sgm : crack.segments)
                    dist = std::min(dist, distanceToSegment(x, y, sgm));
                if (dist >= w) continue;

                const double H0 = crack.B * Hc * (1.0 - dist / w);
                double& H = history[e][static_cast<std::size_t>(g)];
                if (H0 > H) H = H0;
                ++rep.n_gauss;
                rep.H_max = std::max(rep.H_max, H0);
                rep.w_min = std::min(rep.w_min, w);
                rep.w_max = std::max(rep.w_max, w);
            }
        }
        if (rep.n_gauss == 0) rep.w_min = rep.w_max = 0.0;
        reports.push_back(rep);
    }
    return reports;
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
