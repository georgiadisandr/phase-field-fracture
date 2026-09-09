
#ifndef JACOB2_H
#define JACOB2_H
#include <vector>
#include "ElementUtils.h"
#include "ShapeFunc.h"

namespace fem {

// ---------------------------------------------------------------------------
// Original API. Allocates cartd / gcpod on every call -- fine at setup time,
// far too expensive inside a Gauss-point loop. Kept for reference and for any
// caller that still wants the gcpod field.
// ---------------------------------------------------------------------------
struct Jacobian {
    std::vector<std::vector<double>> cartd; //Cartesian derivatives
    std::vector<double> gcpod; //Gauss point global coordinates
    double djacb =0.0; //det of Jacobian matrix
};

Jacobian jacob2(
    const std::vector<std::vector<double>>& elcod, //element coordinates
    const ShapeData& sh_f,//shape functions in natural coordinates
    int nnode,int ndime);//number of node per element

// ---------------------------------------------------------------------------
// Fast API (2D).
//
// Same arithmetic as jacob2(), in the same order, but:
//   * takes the PRECOMPUTED reference derivatives dN_i/dxi from RefElement
//     instead of recomputing the shape functions,
//   * writes into caller-owned fixed-size storage instead of allocating,
//   * skips gcpod entirely -- the Gauss point's global coordinates are
//     computed by the old routine on every call and never read by anything.
//
//   elcod : ndime x nnode, nodal coordinates
//   dNr   : ndime x nnode, reference derivatives at this Gauss point
//   cartd : OUT, ndime x nnode, Cartesian derivatives dN_i/dx_d
//   djacb : OUT, det(J)
//
// Returns false and leaves cartd zeroed when det(J) <= 0 (degenerate or
// inverted element) -- matching the original's behaviour of bailing out
// before the division.
// ---------------------------------------------------------------------------
bool jacob2Fast(const DimNodeMat& elcod,
                const DimNodeMat& dNr,
                int nnode, int ndime,
                DimNodeMat& cartd,
                double&     djacb);

}  // namespace fem

#endif //JACOB2_H
