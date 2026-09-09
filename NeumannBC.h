#pragma once
//
// External-force vector for the monolithic phase-field model.
//
// Sister module to BoundaryConditions.cpp:
//   * BoundaryConditions.cpp  applies Dirichlet constraints on K and R.
//   * NeumannBC.cpp           builds the load_factor-independent external
//                             force vector F_ext.
//
// F_ext collects two kinds of applied loading:
//
//   1. Neumann edge tractions. Each NeumannEdge in FemInput is one 1D
//      boundary element (Line2 or Line3) extracted by GmshReader from a
//      [[neumann]] physical group. For each edge the standard surface-
//      traction term is integrated with 1D Gauss-Legendre quadrature:
//
//          f_e_i = INT_edge  N_i(s) * t  *  |dx/ds| ds
//
//   2. Concentrated nodal point loads. Each PointLoad in FemInput is a
//      force applied directly at one mesh node (from a [[point_load]]
//      0D physical group). No integration -- the force IS the nodal load.
//
// The function is called ONCE per load step (or even once per run, if the
// mesh is not changing). The caller scales the result by load_factor and
// subtracts it from the assembled internal residual:
//
//     R_full = R_internal - load_factor * F_ext
//
// before applyDirichlet rewrites K, R for the constrained DOFs.
//
// Layout of the returned vector: length 2*npoin (displacement DOFs only,
// node-major: [u0x, u0y, u1x, u1y, ...]). The phase-field block has no
// external forces, so it is not exposed here.
//

#include "FEMINPUT.h"

#include <Eigen/Dense>

namespace pfm {

// Build the full-load external force vector F_ext from every NeumannEdge and
// every PointLoad in the FemInput. Output length is 2 * d.npoin. Returns the
// zero vector when neither is present.
[[nodiscard]] Eigen::VectorXd assembleExternalForce(const FemInput& d);

}  // namespace pfm
