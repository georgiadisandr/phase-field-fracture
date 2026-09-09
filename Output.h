#pragma once
//
// VTK output for visualisation of the displacement and phase-field state.
//
// Writes a legacy VTK file containing the current mesh plus the current
// (u, phi) state. The resulting .vtk file can be opened directly in ParaView.
//
// File contents:
//   * POINTS         : nodal coordinates (x, y, 0)
//   * CELLS          : element connectivity (Tri3, Quad4, Quad8 supported)
//   * CELL_TYPES     : VTK cell-type ids matched to nnode
//   * POINT_DATA:
//       SCALARS phi          : nodal phase-field value
//       VECTORS displacement : nodal displacement (ux, uy, 0)
//   * CELL_DATA (one value per element):
//       SCALARS sigma_xx     : effective Cauchy stress, xx component
//       SCALARS sigma_yy     : effective Cauchy stress, yy component
//       SCALARS sigma_xy     : effective Cauchy shear stress
//       SCALARS von_mises    : von Mises scalar from the effective stress
//     Stresses are sigma_eff = g(phi) sigma^+ + sigma^- averaged over the
//     element's Gauss points (plane-stress von Mises formula).
//
// For load stepping, call writeVTK repeatedly with filenames like
// "step_000.vtk", "step_001.vtk", ... ParaView treats a numbered sequence
// as an animation automatically.
//

#include "FEMINPUT.h"

#include <Eigen/Dense>

#include <string>

namespace pfm {
namespace io {

// One-shot snapshot of (u, phi) on the current mesh.
//
//   filename : output path (".vtk" convention but any extension works)
//   d        : FemInput (mesh + element connectivity)
//   u        : displacement, length 2 * npoin, node-major [u0x, u0y, ...]
//   phi      : phase field, length npoin
//
// Throws if vector sizes don't match the mesh.
void writeVTK(const std::string&     filename,
              const FemInput&        d,
              const Eigen::VectorXd& u,
              const Eigen::VectorXd& phi);

}  // namespace io
}  // namespace pfm
