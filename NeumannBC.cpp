#include "NeumannBC.h"

#include "Gauss.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace pfm {

namespace {

// ---------------------------------------------------------------------------
// 1D parent-element shape functions on s in [-1, 1].
//
// Line2 (2 nodes, gmsh ordering [end0, end1]):
//   N0 = 0.5 * (1 - s)            -- node at s = -1
//   N1 = 0.5 * (1 + s)            -- node at s = +1
//   dN0/ds = -0.5,  dN1/ds = 0.5
//
// Line3 (3 nodes, gmsh ordering [end0, end1, mid]):
//   N0 = 0.5 * s * (s - 1)        -- end0 at s = -1
//   N1 = 0.5 * s * (s + 1)        -- end1 at s = +1
//   N2 = 1 - s * s                -- mid  at s =  0
//   dN0/ds = s - 0.5
//   dN1/ds = s + 0.5
//   dN2/ds = -2 * s
//
// Returned via fixed-size std::array so the caller never heap-allocates per
// Gauss point. nnode is read separately and tells the caller how many of
// the three slots are populated.
// ---------------------------------------------------------------------------
struct Line1DShape {
    std::array<double, 3> N;
    std::array<double, 3> dN;
};

Line1DShape line_shape(int nnode, double s)
{
    Line1DShape sh{};
    if (nnode == 2) {
        sh.N[0]  = 0.5 * (1.0 - s);
        sh.N[1]  = 0.5 * (1.0 + s);
        sh.dN[0] = -0.5;
        sh.dN[1] =  0.5;
    } else if (nnode == 3) {
        sh.N[0]  = 0.5 * s * (s - 1.0);
        sh.N[1]  = 0.5 * s * (s + 1.0);
        sh.N[2]  = 1.0 - s * s;
        sh.dN[0] = s - 0.5;
        sh.dN[1] = s + 0.5;
        sh.dN[2] = -2.0 * s;
    } else {
        throw std::runtime_error(
            "NeumannBC: only Line2 (nnode=2) and Line3 (nnode=3) edges are supported");
    }
    return sh;
}

// Quadrature order that integrates the surface-traction term exactly for a
// straight edge: 2 points for Line2, 3 for Line3 (the |dx/ds| Jacobian on a
// straight Line3 is linear in s, so 3 points are enough; if you ever support
// curved Line3 you may want 4).
int ngauss_for(int nnode)
{
    return (nnode == 3) ? 3 : 2;
}

}  // namespace

Eigen::VectorXd assembleExternalForce(const FemInput& d)
{
    const int npoin = d.npoin;
    Eigen::VectorXd F = Eigen::VectorXd::Zero(2 * npoin);

    // Nothing to do if there are neither edge tractions nor point loads.
    if (d.neumann_edges.empty() && d.point_loads.empty())
        return F;
    if (d.ndime != 2)
        throw std::runtime_error("assembleExternalForce: only 2D is implemented");

    // -----------------------------------------------------------------------
    // 1. Neumann edge tractions: F += INT_edge N(s)^T * t * |dx/ds| ds
    // -----------------------------------------------------------------------
    if (!d.neumann_edges.empty()) {
    // Cache 1D rules so we don't rebuild them per edge.
    const quadrature::QuadGauss gp2 = quadrature::quadGauss(2);
    const quadrature::QuadGauss gp3 = quadrature::quadGauss(3);

    for (const FemInput::NeumannEdge& edge : d.neumann_edges) {
        const int nnode = static_cast<int>(edge.nodes.size());
        const int ng    = ngauss_for(nnode);
        const auto& rule = (ng == 3) ? gp3 : gp2;

        // Cache nodal coordinates of this edge: x[i], y[i].
        std::array<double, 3> ex{}, ey{};
        for (int i = 0; i < nnode; ++i) {
            const int node = edge.nodes[i];
            ex[i] = d.coord(node, 0);
            ey[i] = d.coord(node, 1);
        }

        const double tx = edge.tx;
        const double ty = edge.ty;

        // Element-local force accumulator (max 3 nodes -> 6 dofs).
        std::array<double, 6> fe{};

        for (int g = 0; g < ng; ++g) {
            const double s   = rule.xi[g];
            const double wgt = rule.w[g];

            const Line1DShape sh = line_shape(nnode, s);

            // Jacobian of the 1D mapping: |dx/ds| = sqrt((dx/ds)^2 + (dy/ds)^2).
            double dxds = 0.0, dyds = 0.0;
            for (int i = 0; i < nnode; ++i) {
                dxds += sh.dN[i] * ex[i];
                dyds += sh.dN[i] * ey[i];
            }
            const double jac = std::sqrt(dxds * dxds + dyds * dyds);
            const double dS  = jac * wgt;

            // Accumulate N_i * t * dS at every edge node.
            for (int i = 0; i < nnode; ++i) {
                fe[2 * i    ] += sh.N[i] * tx * dS;
                fe[2 * i + 1] += sh.N[i] * ty * dS;
            }
        }

        // Scatter into the global u-block.
        for (int i = 0; i < nnode; ++i) {
            const int node = edge.nodes[i];
            F(2 * node    ) += fe[2 * i    ];
            F(2 * node + 1) += fe[2 * i + 1];
        }
    }
    }  // if (!d.neumann_edges.empty())

    // -----------------------------------------------------------------------
    // 2. Concentrated nodal point loads: add the force straight into F.
    //    No integration -- a point load IS the nodal force.
    // -----------------------------------------------------------------------
    for (const FemInput::PointLoad& pl : d.point_loads) {
        F(2 * pl.node    ) += pl.fx;
        F(2 * pl.node + 1) += pl.fy;
    }

    return F;
}

}  // namespace pfm
