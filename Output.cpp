#include "Output.h"

#include "Profiling.h"
#include "StiffnessPFM.h"     // for pfm::computeElementStresses

#include <fstream>
#include <stdexcept>
#include <string>

namespace pfm {
namespace io {

namespace {

// Legacy-VTK cell-type ids for the element families we support.
//   Tri3  -> VTK_TRIANGLE       = 5
//   Quad4 -> VTK_QUAD           = 9
//   Quad8 -> VTK_QUADRATIC_QUAD = 23
int vtkCellType(int nnode)
{
    switch (nnode) {
        case 3:  return 5;
        case 4:  return 9;
        case 8:  return 23;
        default:
            throw std::runtime_error(
                "writeVTK: unsupported nnode = " + std::to_string(nnode));
    }
}

// Permutation from our INTERNAL Q8 node order (interleaved) back to VTK's Q8
// order (corners first, then midpoints). Inverse of GmshReader's kQ8Perm.
constexpr int kInternalToVtkQ8[8] = { 0, 2, 4, 6,    // corners
                                       1, 3, 5, 7 }; // midpoints

}  // namespace

void writeVTK(const std::string&     filename,
              const FemInput&        d,
              const Eigen::VectorXd& u,
              const Eigen::VectorXd& phi)
{
    if (u.size()   != 2 * d.npoin)
        throw std::runtime_error("writeVTK: u size != 2*npoin");
    if (phi.size() != d.npoin)
        throw std::runtime_error("writeVTK: phi size != npoin");

    prof::Scope _t(prof::Output);

    std::ofstream f(filename);
    if (!f)
        throw std::runtime_error("writeVTK: cannot open " + filename);

    // 1. Header
    f << "# vtk DataFile Version 3.0\n";
    f << "PFM solution\n";
    f << "ASCII\n";
    f << "DATASET UNSTRUCTURED_GRID\n";

    // 2. Nodal coordinates (z = 0 for 2D meshes)
    f << "POINTS " << d.npoin << " float\n";
    for (int n = 0; n < d.npoin; ++n) {
        f << d.coord(n, 0) << ' ' << d.coord(n, 1) << " 0\n";
    }

    // 3. Cell connectivity. Each row: <nnode> <node0> <node1> ...
    int total_cell_data = 0;
    for (int e = 0; e < d.nelem; ++e)
        total_cell_data += 1 + d.nnode_of(e);

    f << "CELLS " << d.nelem << ' ' << total_cell_data << '\n';
    for (int e = 0; e < d.nelem; ++e) {
        const int nnode = d.nnode_of(e);
        f << nnode;
        for (int i = 0; i < nnode; ++i) {
            const int src = (nnode == 8) ? kInternalToVtkQ8[i] : i;
            f << ' ' << d.conn[d.offset[e] + src];
        }
        f << '\n';
    }

    // 4. Cell types (one id per element)
    f << "CELL_TYPES " << d.nelem << '\n';
    for (int e = 0; e < d.nelem; ++e)
        f << vtkCellType(d.nnode_of(e)) << '\n';

    // 5. Point data: phi (scalar) and displacement (3-vector with z=0).
    f << "POINT_DATA " << d.npoin << '\n';

    f << "SCALARS phi float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int n = 0; n < d.npoin; ++n)
        f << phi(n) << '\n';

    f << "VECTORS displacement float\n";
    for (int n = 0; n < d.npoin; ++n)
        f << u(2 * n) << ' ' << u(2 * n + 1) << " 0\n";

    // 6. Cell data: per-element effective stress and von Mises.
    //    sigma_eff = g(phi) * sigma^+ + sigma^-, averaged over the element's
    //    Gauss points. Written as CELL_DATA -- in ParaView each cell shows a
    //    single (flat) value, which matches how the quantity is integrated.
    const PerElementStresses se = computeElementStresses(d, u, phi);

    f << "CELL_DATA " << d.nelem << '\n';

    f << "SCALARS sigma_xx float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int e = 0; e < d.nelem; ++e) f << se.sigma_xx[e] << '\n';

    f << "SCALARS sigma_yy float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int e = 0; e < d.nelem; ++e) f << se.sigma_yy[e] << '\n';

    f << "SCALARS sigma_xy float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int e = 0; e < d.nelem; ++e) f << se.sigma_xy[e] << '\n';

    f << "SCALARS von_mises float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int e = 0; e < d.nelem; ++e) f << se.von_mises[e] << '\n';
}

}  // namespace io
}  // namespace pfm
