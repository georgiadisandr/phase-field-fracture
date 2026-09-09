#include "Jacob2.h"
#include <vector>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fem {

Jacobian jacob2(const std::vector<std::vector<double>>& elcod,
                const ShapeData& sh_f,
                int nnode, int ndime)
{
    // const int ndime=2;
    Jacobian J;
    J.cartd.assign(ndime, std::vector<double>(nnode, 0.0));// cartesian Derivatives [∂N/∂x ∂N/∂y]
    J.gcpod.assign(ndime, 0.0); //Gauss point global coordinates Σ Ν_i*x_i

    // Global coordinates at this Gauss point
    for (int i = 0; i < ndime; ++i) {
        double acc = 0.0;
        for (int n = 0; n < nnode; ++n) {
            acc += elcod[i][n] * sh_f.Shape[n]; 
        }
        J.gcpod[i] = acc;
    }

    // Jacobian matrix (2D)
    double xjacm[2][2] = {{0.0, 0.0}, {0.0, 0.0}};//initialize J = [0]
    for (int i = 0; i < ndime; ++i) {
        for (int j = 0; j < ndime; ++j) {
            double acc = 0.0;
            for (int n = 0; n < nnode; ++n) {
                acc += sh_f.Deriv[i][n] * elcod[j][n]; //Σ Νd_i*x_i
            }
            xjacm[i][j] = acc;
        }
    }

    // Determinant det(J)
    J.djacb = xjacm[0][0] * xjacm[1][1] - xjacm[0][1] * xjacm[1][0];
    if (J.djacb <= 0.0) {
        std::cerr << "Zero or negative area for element " << '\n';
        return J;            // bail out before dividing by it
    }

    // Inverse — one reciprocal, four multiplies
    const double invDet = 1.0 / J.djacb;
    const double xjaci[2][2] = {
        {  xjacm[1][1] * invDet, -xjacm[0][1] * invDet },
        { -xjacm[1][0] * invDet,  xjacm[0][0] * invDet }
    };

    // Cartesian derivatives — write straight into J.cartd
    for (int i = 0; i < ndime; ++i) {
        for (int n = 0; n < nnode; ++n) {
            double acc = 0.0;
            for (int j = 0; j < ndime; ++j) {
                acc += xjaci[i][j] * sh_f.Deriv[j][n];
            }
            J.cartd[i][n] = acc;
        }
    }

    return J;
}

// ---------------------------------------------------------------------------
// Degenerate / inverted element reporting.
//
// A non-positive det(J) means the element is degenerate (zero area) or
// inverted (negative area) -- a MESH problem, not a solver problem. The
// routine below bails out before dividing by it, which leaves cartd zero, so
// the element silently contributes ZERO stiffness and ZERO residual. The run
// then completes and produces plausible-looking but wrong results.
//
// jacob2Fast() is not told which element it is working on (elementSystem never
// receives an element index), but it does have the nodal coordinates -- which
// are more directly useful anyway, since they let you find the element in Gmsh.
//
// Reporting is throttled: full detail for the first few occurrences, then a
// running count at intervals, so a persistent problem stays visible without
// flooding the log with one message per Gauss point per iteration.
// ---------------------------------------------------------------------------
namespace {

long long g_degenerate_jacobian_count = 0;

void reportDegenerateJacobian(const DimNodeMat& elcod,
                              int nnode, int ndime, double djacb)
{
    ++g_degenerate_jacobian_count;

    constexpr long long kDetailLimit  = 10;    // full report for the first N
    constexpr long long kSummaryEvery = 1000;  // then a count every M

    if (g_degenerate_jacobian_count > kDetailLimit) {
        if (g_degenerate_jacobian_count % kSummaryEvery == 0)
            std::cerr << "[mesh] WARNING: " << g_degenerate_jacobian_count
                      << " non-positive Jacobian evaluations so far.\n";
        return;
    }

    // Centroid and bounding box of the offending element, to locate it.
    double cx = 0.0, cy = 0.0;
    double xmin = elcod(0, 0), xmax = elcod(0, 0);
    double ymin = (ndime > 1) ? elcod(1, 0) : 0.0;
    double ymax = ymin;
    for (int i = 0; i < nnode; ++i) {
        const double x = elcod(0, i);
        const double y = (ndime > 1) ? elcod(1, i) : 0.0;
        cx += x;  cy += y;
        if (x < xmin) xmin = x;   if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;   if (y > ymax) ymax = y;
    }
    cx /= static_cast<double>(nnode);
    cy /= static_cast<double>(nnode);

    const std::ios_base::fmtflags saved = std::cerr.flags();
    const std::streamsize         prec  = std::cerr.precision();

    std::cerr << std::scientific << std::setprecision(6)
        << "[mesh] WARNING: non-positive Jacobian determinant.\n"
        << "       det(J)   = " << djacb << "  (must be > 0)\n"
        << "       element  = " << nnode << "-node\n"
        << "       centroid = (" << cx << ", " << cy << ")\n"
        << "       bbox     = x [" << xmin << ", " << xmax << "]"
                        << "  y [" << ymin << ", " << ymax << "]\n"
        << "       This element contributes ZERO stiffness and ZERO residual,\n"
        << "       so the results WILL be wrong even though the run completes.\n"
        << "       Inspect the mesh at the coordinates above (a zero-area or\n"
        << "       inverted element, usually from a bad refinement transition).\n"
        << "       Occurrence " << g_degenerate_jacobian_count;
    if (g_degenerate_jacobian_count == kDetailLimit)
        std::cerr << " -- further occurrences reported every "
                  << kSummaryEvery << " only";
    std::cerr << ".\n";

    std::cerr.flags(saved);
    std::cerr.precision(prec);
}

}  // namespace

// ---------------------------------------------------------------------------
// Fast path. The arithmetic below is deliberately written as the same explicit
// loops, in the same order, as jacob2() above -- not as Eigen expressions --
// so the floating-point result is identical and this change cannot perturb the
// solution. The savings come entirely from not allocating and from reading the
// reference derivatives out of a precomputed table.
// ---------------------------------------------------------------------------
bool jacob2Fast(const DimNodeMat& elcod,
                const DimNodeMat& dNr,
                int nnode, int ndime,
                DimNodeMat& cartd,
                double&     djacb)
{
    // This routine is 2D by construction (xjacm below is 2x2, and the inverse
    // is the closed-form 2x2 one). Guard rather than overrun the buffer.
    if (ndime != 2)
        throw std::runtime_error(
            "fem::jacob2Fast: ndime must be 2 (got " + std::to_string(ndime) +
            "); the Jacobian inverse here is the closed-form 2x2 one");

    cartd.setZero(ndime, nnode);
    djacb = 0.0;

    // Jacobian matrix (2D):  xjacm[i][j] = sum_n dN_n/dxi_i * x_j,n
    double xjacm[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int i = 0; i < ndime; ++i) {
        for (int j = 0; j < ndime; ++j) {
            double acc = 0.0;
            for (int n = 0; n < nnode; ++n) {
                acc += dNr(i, n) * elcod(j, n);
            }
            xjacm[i][j] = acc;
        }
    }

    djacb = xjacm[0][0] * xjacm[1][1] - xjacm[0][1] * xjacm[1][0];
    if (djacb <= 0.0) {
        reportDegenerateJacobian(elcod, nnode, ndime, djacb);
        return false;                 // cartd stays zero, as in jacob2()
    }

    const double invDet = 1.0 / djacb;
    const double xjaci[2][2] = {
        {  xjacm[1][1] * invDet, -xjacm[0][1] * invDet },
        { -xjacm[1][0] * invDet,  xjacm[0][0] * invDet }
    };

    // Cartesian derivatives:  dN_n/dx_i = sum_j J^-1[i][j] * dN_n/dxi_j
    for (int i = 0; i < ndime; ++i) {
        for (int n = 0; n < nnode; ++n) {
            double acc = 0.0;
            for (int j = 0; j < ndime; ++j) {
                acc += xjaci[i][j] * dNr(j, n);
            }
            cartd(i, n) = acc;
        }
    }

    return true;
}

}  // namespace fem