#include "ElementUtils.h"

#include "Gauss.h"
#include "ShapeFunc.h"

#include <stdexcept>
#include <string>

namespace fem {

// ===========================================================================
// Original API
// ===========================================================================

std::vector<GaussPoint> gaussPoints(int nnode, int ngaus_per_dir)
{
    std::vector<GaussPoint> pts;

    if (nnode == 3) {
        const auto tri = quadrature::triGauss(3);
        pts.reserve(static_cast<std::size_t>(tri.ngaus()));
        for (int g = 0; g < tri.ngaus(); ++g)
            pts.push_back({ tri.xi[g], tri.eta[g], tri.w[g] });
        return pts;
    }

    if (nnode == 4 || nnode == 8) {
        const int n1 = (ngaus_per_dir > 0) ? ngaus_per_dir
                                           : (nnode == 4 ? 2 : 3);

        const auto q = quadrature::quadGauss(n1);
        pts.reserve(static_cast<std::size_t>(n1) * n1);
        for (int i = 0; i < n1; ++i)
            for (int j = 0; j < n1; ++j)
                pts.push_back({ q.xi[i], q.xi[j], q.w[i] * q.w[j] });
        return pts;
    }

    throw std::runtime_error(
        "fem::gaussPoints: unsupported nnode = " + std::to_string(nnode));
}

Eigen::MatrixXd buildBu(const std::vector<std::vector<double>>& cartd, int nnode)
{
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 2 * nnode);
    for (int i = 0; i < nnode; ++i) {
        const double dNdx = cartd[0][i];
        const double dNdy = cartd[1][i];
        B(0, 2 * i)     = dNdx;
        B(1, 2 * i + 1) = dNdy;
        B(2, 2 * i)     = dNdy;
        B(2, 2 * i + 1) = dNdx;
    }
    return B;
}

Eigen::MatrixXd buildBphi(const std::vector<std::vector<double>>& cartd,
                          int ndime, int nnode)
{
    Eigen::MatrixXd B(ndime, nnode);
    for (int d = 0; d < ndime; ++d)
        for (int i = 0; i < nnode; ++i)
            B(d, i) = cartd[d][i];
    return B;
}

// ===========================================================================
// Fast API
// ===========================================================================

RefElement buildRefElement(int nnode, int ngaus_per_dir, int ndime)
{
    if (nnode > kMaxNodesPerElement)
        throw std::runtime_error(
            "fem::buildRefElement: nnode = " + std::to_string(nnode) +
            " exceeds kMaxNodesPerElement = " +
            std::to_string(kMaxNodesPerElement) +
            "; raise that constant in ElementUtils.h");
    if (ndime > kMaxDime)
        throw std::runtime_error(
            "fem::buildRefElement: ndime = " + std::to_string(ndime) +
            " exceeds kMaxDime = " + std::to_string(kMaxDime));
    if (nnode <= 0 || ndime <= 0)
        throw std::runtime_error("fem::buildRefElement: nnode and ndime must be > 0");

    // Evaluate the EXISTING routines once per Gauss point. Doing it this way
    // (rather than re-deriving the formulas here) guarantees the tabulated
    // values match what the old code computed, bit for bit.
    const std::vector<GaussPoint> gps = gaussPoints(nnode, ngaus_per_dir);

    RefElement ref;
    ref.nnode = nnode;
    ref.ndime = ndime;
    ref.ngp   = static_cast<int>(gps.size());

    ref.w  .resize(static_cast<std::size_t>(ref.ngp));
    ref.N  .resize(static_cast<std::size_t>(ref.ngp));
    ref.dNr.resize(static_cast<std::size_t>(ref.ngp));

    for (int g = 0; g < ref.ngp; ++g) {
        const std::size_t gi = static_cast<std::size_t>(g);
        const ShapeData sh = shapeFunc(gps[gi].xi, gps[gi].eta, nnode);

        if (static_cast<int>(sh.Shape.size()) != nnode)
            throw std::runtime_error(
                "fem::buildRefElement: shapeFunc returned the wrong node count");
        // shapeFunc() currently hardcodes 2 derivative rows. Fail loudly rather
        // than index past the end if someone raises ndime without updating it.
        if (static_cast<int>(sh.Deriv.size()) < ndime)
            throw std::runtime_error(
                "fem::buildRefElement: shapeFunc supplied " +
                std::to_string(sh.Deriv.size()) + " derivative rows but ndime = " +
                std::to_string(ndime) + "; ShapeFunc.cpp is 2D-only");

        ref.w[gi] = gps[gi].w;

        ref.N[gi].resize(nnode);
        for (int i = 0; i < nnode; ++i)
            ref.N[gi](i) = sh.Shape[static_cast<std::size_t>(i)];

        ref.dNr[gi].resize(ndime, nnode);
        for (int dd = 0; dd < ndime; ++dd)
            for (int i = 0; i < nnode; ++i)
                ref.dNr[gi](dd, i) =
                    sh.Deriv[static_cast<std::size_t>(dd)][static_cast<std::size_t>(i)];
    }

    return ref;
}

const RefElement& RefElementTable::get(int nnode) const
{
    if (!has(nnode))
        throw std::runtime_error(
            "fem::RefElementTable::get: no reference element registered for "
            "nnode = " + std::to_string(nnode) +
            " (was it present in the mesh when the table was built?)");
    return by_nnode[static_cast<std::size_t>(nnode)];
}

RefElementTable buildRefElementTable(const std::vector<int>& distinct_nnode,
                                     int ngaus_per_dir,
                                     int ndime)
{
    RefElementTable table;
    for (const int nn : distinct_nnode) {
        if (nn <= 0) continue;
        if (nn > kMaxNodesPerElement)
            throw std::runtime_error(
                "fem::buildRefElementTable: element with nnode = " +
                std::to_string(nn) + " exceeds kMaxNodesPerElement = " +
                std::to_string(kMaxNodesPerElement) +
                "; raise that constant in ElementUtils.h");
        table.by_nnode[static_cast<std::size_t>(nn)] =
            buildRefElement(nn, ngaus_per_dir, ndime);
    }
    return table;
}

void buildBu(const DimNodeMat& cartd, int nnode, BuMat& B)
{
    B.setZero(3, 2 * nnode);
    for (int i = 0; i < nnode; ++i) {
        const double dNdx = cartd(0, i);
        const double dNdy = cartd(1, i);
        B(0, 2 * i)     = dNdx;
        B(1, 2 * i + 1) = dNdy;
        B(2, 2 * i)     = dNdy;
        B(2, 2 * i + 1) = dNdx;
    }
}

}  // namespace fem
