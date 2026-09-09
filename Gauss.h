
#ifndef GAUSS_H
#define GAUSS_H

#include <vector>

namespace quadrature {

// ---------------------------------------------------------------------------
// Triangle quadrature
// ---------------------------------------------------------------------------

struct TriGauss {
    std::vector<double> xi;
    std::vector<double> eta;
    std::vector<double> w;
    int ngaus() const { return static_cast<int>(w.size()); }// return number of gauss poitns
};

// ---------------------------------------------------------------------------
// Quadrilateral quadrature
// ---------------------------------------------------------------------------
// 1D Gauss-Legendre rule on the reference interval [-1, 1].
// For 2D quads, take the tensor product:
struct QuadGauss {
    std::vector<double> xi;
    std::vector<double> w;
    int ngaus1d() const { return static_cast<int>(w.size()); }// return number of gauss points
};

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------
// Triangle rule. Supported ngaus:
//   1  -> degree 1 (centroid)
//   3  -> degree 2 (edge midpoints)
//   7  -> degree 5 (Dunavant: 1 centroid + 6 interior points)
TriGauss triGauss(int ngaus);

// 1D Gauss-Legendre rule (use tensor-product for 2D quads).
// Supported ngaus: 1, 2, 3, 4, 5, 6.
QuadGauss quadGauss(int ngaus);

} // namespace quadrature

#endif // GAUSS_H
