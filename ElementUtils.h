#pragma once
//
// Small element-level helpers shared by every assembly routine (stiffness,
// residual, post-processing). They live in namespace fem because they only
// depend on the mesh / reference element, not on the physics.
//
// Quick map of who builds what:
//
//   Gauss.{h,cpp}         raw quadrature rules on the reference element
//   ShapeFunc.{h,cpp}     reference shape functions and their (xi,eta) derivs
//   Jacob2.{h,cpp}        reference -> physical mapping (cartd, djacb)
//   ElementUtils.{h,cpp}  glue that the physics calls: a Gauss-point list
//                         tailored to the element family, and the standard
//                         strain-displacement / phase-field B matrices.
//
// ---------------------------------------------------------------------------
// TWO APIs LIVE HERE
//
//   * The ORIGINAL one (gaussPoints, buildBu, buildBphi returning MatrixXd).
//     Kept for setup-time use and for reference; it allocates on every call.
//
//   * The FAST one (RefElement / RefElementTable + the in-place builders).
//     Used by the Gauss-point loops. Two ideas:
//
//       1. The reference shape functions N_i(xi_g, eta_g) and their reference
//          derivatives dN_i/dxi depend ONLY on the element TYPE and the Gauss
//          point -- never on which element. For a quad4 mesh with ngaus = 2
//          there are exactly 4 distinct results in the whole program, yet the
//          old code recomputed them once per Gauss point per element per
//          assembly. RefElement stores them once, at setup.
//
//       2. ndime <= 3 and nnode <= kMaxNodesPerElement, so every per-element
//          working array fits in a fixed-size Eigen object whose storage lives
//          on the stack. That removes ~11 heap allocations per Gauss point and
//          makes the data contiguous (vector<vector<double>> was neither).
//
//     The reference tables are BUILT BY CALLING the original shapeFunc() and
//     gaussPoints(), so the numbers are identical by construction.
// ---------------------------------------------------------------------------
//

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace fem {

// ---------------------------------------------------------------------------
// Size bounds for the stack-allocated element types.
//
// Raise kMaxNodesPerElement if you add a richer element (Quad9 = 9 is already
// covered; Tri10 / Hex20 would need more). It is the ONLY place to change --
// buildRefElement() checks it and throws a clear error rather than letting
// Eigen overrun a fixed buffer (Eigen's own assertion is compiled out in
// Release by EIGEN_NO_DEBUG).
// ---------------------------------------------------------------------------
inline constexpr int kMaxNodesPerElement = 9;
inline constexpr int kMaxUDofsPerElement = 2 * kMaxNodesPerElement;
inline constexpr int kMaxDime            = 3;

// Fixed-MAXIMUM-size Eigen types. The actual size is set at runtime, but the
// storage is a plain array sized at compile time, so these never touch the
// heap and never need an allocator call.
//
//   NodeVec    : one scalar per element node          (length nnode)
//   UDofVec    : one scalar per element displacement dof (length 2*nnode)
//   DimNodeMat : ndime x nnode -- used for elcod, cartd, reference derivs
//                and B_phi (which IS cartd; see buildBphi note below)
//   BuMat      : 3 x 2*nnode strain-displacement matrix (2D Voigt)
using NodeVec    = Eigen::Matrix<double, Eigen::Dynamic, 1,
                                 0, kMaxNodesPerElement, 1>;
using DimVec     = Eigen::Matrix<double, Eigen::Dynamic, 1,
                                 0, kMaxDime, 1>;
using UDofVec    = Eigen::Matrix<double, Eigen::Dynamic, 1,
                                 0, kMaxUDofsPerElement, 1>;
using DimNodeMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 0, kMaxDime, kMaxNodesPerElement>;
using BuMat      = Eigen::Matrix<double, 3, Eigen::Dynamic,
                                 0, 3, kMaxUDofsPerElement>;

// ---------------------------------------------------------------------------
// Original API (setup-time / reference)
// ---------------------------------------------------------------------------

// One Gauss point on the reference element. eta is unused for 1D quads / lines
// but kept zero by callers for those (we don't use 1D right now).
struct GaussPoint {
    double xi  = 0.0;
    double eta = 0.0;
    double w   = 0.0;
};

// Gauss-point list for the chosen element family.
//   nnode = 3        : 3-point Dunavant rule on the reference triangle
//                       (ngaus_per_dir is ignored)
//   nnode = 4 or 8   : tensor-product Gauss-Legendre on [-1, 1]^2,
//                       n1d = ngaus_per_dir (defaults: 2 for Q4, 3 for Q8)
std::vector<GaussPoint> gaussPoints(int nnode, int ngaus_per_dir);

// Strain-displacement matrix B_u (3 x 2*nnode) in Voigt, engineering-shear.
// Column block for node i:
//        | dN_i/dx     0       |
//        |   0       dN_i/dy   |
//        | dN_i/dy   dN_i/dx   |
//
// cartd[idime][inode] is the Cartesian-derivative table produced by jacob2.
Eigen::MatrixXd buildBu(const std::vector<std::vector<double>>& cartd,
                        int nnode);

// Phase-field gradient matrix B_phi (ndime x nnode). Column i is grad N_i.
// Thin wrapper around cartd, kept here for symmetry with buildBu so call
// sites read uniformly.
Eigen::MatrixXd buildBphi(const std::vector<std::vector<double>>& cartd,
                          int ndime,
                          int nnode);

// ---------------------------------------------------------------------------
// Fast API: precomputed reference element
// ---------------------------------------------------------------------------

// Everything about one element TYPE that is independent of which element it
// is: the quadrature weights, the shape functions and the reference-coordinate
// derivatives, all evaluated at each Gauss point.
//
//   w[g]      quadrature weight at Gauss point g
//   N[g]      shape functions      N_i(xi_g, eta_g)      (length nnode)
//   dNr[g]    reference derivatives dN_i/dxi_d           (ndime x nnode)
//
// The physical Cartesian derivatives dN_i/dx are NOT here -- those depend on
// the element's nodal coordinates through the Jacobian and are still computed
// per element, by jacob2Fast().
struct RefElement {
    int nnode = 0;              // 0 marks an unpopulated slot
    int ngp   = 0;
    int ndime = 0;

    std::vector<double>     w;
    std::vector<NodeVec>    N;
    std::vector<DimNodeMat> dNr;
};

// Build one RefElement by evaluating the existing gaussPoints() / shapeFunc()
// once per Gauss point. Throws if nnode exceeds kMaxNodesPerElement.
RefElement buildRefElement(int nnode, int ngaus_per_dir, int ndime);

// Lookup table indexed directly by nnode, so the assembly loop resolves an
// element's reference data with a single array read.
struct RefElementTable {
    std::array<RefElement, kMaxNodesPerElement + 1> by_nnode{};

    // Throws if the element type was not registered at setup.
    const RefElement& get(int nnode) const;

    bool has(int nnode) const {
        return nnode >= 0 && nnode <= kMaxNodesPerElement
            && by_nnode[static_cast<std::size_t>(nnode)].nnode != 0;
    }
};

// Build the table for every distinct node count present in the mesh.
RefElementTable buildRefElementTable(const std::vector<int>& distinct_nnode,
                                     int ngaus_per_dir,
                                     int ndime);

// In-place B_u builder: writes into caller-owned storage, no allocation and no
// return-by-value temporary. B is resized to 3 x 2*nnode (a no-op after the
// first call, since the storage is fixed-size).
void buildBu(const DimNodeMat& cartd, int nnode, BuMat& B);

// NOTE on B_phi: the old buildBphi() copied cartd element by element into a
// fresh matrix -- B_phi IS cartd. The fast path therefore uses cartd directly
// and no B_phi builder exists here, saving a copy per Gauss point.

}  // namespace fem
