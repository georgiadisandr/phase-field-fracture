#include "GmshReader.h"

#include <gmsh.h>
extern "C" {
#include <gmshc.h>   // C API: ABI-stable across MinGW<->MSVC, used everywhere
}                    // we'd otherwise pass std::vector across the DLL boundary.

#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fem {

namespace {

constexpr int kTri3  = 2;   // gmsh element type id, 3-node triangle
constexpr int kQuad4 = 3;   // gmsh element type id, 4-node quadrilateral
constexpr int kQuad8 = 16;  // gmsh element type id, 8-node serendipity quad

constexpr int kLine2 = 1;   // gmsh element type id, 2-node line
constexpr int kLine3 = 8;   // gmsh element type id, 3-node line (quadratic)

// Number of nodes for a supported 1D (line) gmsh element type, or 0 otherwise.
int nnode_for_line_type(int gtype) {
    switch (gtype) {
        case kLine2: return 2;
        case kLine3: return 3;
        default:     return 0;
    }
}

// Gmsh's Quad8 node order is [c0, c1, c2, c3, m01, m12, m23, m30]
// (four corners CCW, then four edge midpoints CCW from edge 0-1).
// ShapeFunc::Q8 expects the interleaved order
// [c0, m01, c1, m12, c2, m23, c3, m30].
// This permutation maps fem-slot -> gmsh-slot for Q8 connectivity.
constexpr int kQ8Perm[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };

// Number of nodes for a supported gmsh element type, or 0 if unsupported.
int nnode_for_gmsh_type(int gtype) {
    switch (gtype) {
        case kTri3:  return 3;
        case kQuad4: return 4;
        case kQuad8: return 8;
        default:     return 0;
    }
}

void check_ierr(int ierr, const char* where) {
    if (ierr) throw std::runtime_error(
        std::string("gmsh C API error in ") + where +
        " (ierr=" + std::to_string(ierr) + ")");
}

// Find the integer tag of a physical group by (dim, name).
// Returns -1 if no group with that name exists at that dimension.
int find_physical_tag(int dim, const std::string& name)
{
    int ierr = 0;
    int* dimTags = nullptr;
    std::size_t dimTags_n = 0;
    gmshModelGetPhysicalGroups(&dimTags, &dimTags_n, dim, &ierr);
    check_ierr(ierr, "gmshModelGetPhysicalGroups");

    int found = -1;
    // dimTags is a flat array of (dim, tag) pairs.
    for (std::size_t i = 0; i + 1 < dimTags_n; i += 2) {
        const int d = dimTags[i];
        const int t = dimTags[i + 1];

        char* name_c = nullptr;
        gmshModelGetPhysicalName(d, t, &name_c, &ierr);
        check_ierr(ierr, "gmshModelGetPhysicalName");
        if (name_c && name == name_c) {
            found = t;
            gmshFree(name_c);
            break;
        }
        gmshFree(name_c);
    }
    gmshFree(dimTags);
    return found;
}

// Resolve a physical group by name across several dimensions, in priority
// order. Writes the dimension it matched into found_dim, or -1 if no match.
//
// Used by the Dirichlet reader so a BC can name either an EDGE (1D) or a
// POINT (0D). Order matters: 1D is tried first, so if a mesh happens to
// carry both an edge and a vertex under the same name, the edge wins and
// existing configs behave exactly as before.
int find_physical_tag_multi(const std::vector<int>& dims,
                            const std::string& name,
                            int& found_dim)
{
    for (const int dim : dims) {
        const int t = find_physical_tag(dim, name);
        if (t >= 0) { found_dim = dim; return t; }
    }
    found_dim = -1;
    return -1;
}

}  // namespace

FemInput readFemInputFromGmsh(const FemSpec& spec)
{
    FemInput d;
    int ierr = 0;

    // ----- Header counts that come from the spec --------------------------
    d.ntype = spec.ntype;
    d.ndofn = spec.ndofn;
    d.ngaus = spec.ngaus;
    d.nstre = spec.nstre;
    d.ndime = 2;
    d.split  = spec.split;
    d.hybrid = spec.hybrid;
    d.nmats = static_cast<int>(spec.materials.size());
    d.nprop = (d.nmats > 0) ? static_cast<int>(spec.materials[0].props.size()) : 0;
    d.nelem = 0;

    // ----- 1. Nodes & coordinates -----------------------------------------
    std::size_t* nodeTags  = nullptr; std::size_t nodeTags_n  = 0;
    double*      coord     = nullptr; std::size_t coord_n     = 0;
    double*      paramCoor = nullptr; std::size_t paramCoor_n = 0;
    gmshModelMeshGetNodes(&nodeTags, &nodeTags_n,
                          &coord,    &coord_n,
                          &paramCoor,&paramCoor_n,
                          /*dim=*/-1, /*tag=*/-1,
                          /*includeBoundary=*/0, /*returnParam=*/0,
                          &ierr);
    check_ierr(ierr, "gmshModelMeshGetNodes");

    // gmsh node tags are arbitrary positive ints (and may have gaps). We
    // compress them into 0..npoin-1 contiguous fem ids and remember the map.
    std::unordered_map<std::size_t, int> tag_to_id;
    tag_to_id.reserve(nodeTags_n);

    d.npoin = static_cast<int>(nodeTags_n);
    d.coord = Eigen::MatrixXd::Zero(d.npoin, d.ndime);

    for (std::size_t i = 0; i < nodeTags_n; ++i) {
        const int fem_id = static_cast<int>(i);
        tag_to_id[nodeTags[i]] = fem_id;
        d.coord(fem_id, 0) = coord[3 * i + 0];   // x
        d.coord(fem_id, 1) = coord[3 * i + 1];   // y
        // z is ignored for 2D
    }

    gmshFree(nodeTags); gmshFree(coord); gmshFree(paramCoor);

    // ----- 2. Elements (2D) -----------------------------------------------
    int*          elemTypes      = nullptr; std::size_t elemTypes_n      = 0;
    std::size_t** elemTags       = nullptr; std::size_t* elemTags_nn     = nullptr;
                                            std::size_t elemTags_n       = 0;
    std::size_t** elemNodeTags   = nullptr; std::size_t* elemNodeTags_nn = nullptr;
                                            std::size_t elemNodeTags_n   = 0;
    gmshModelMeshGetElements(&elemTypes, &elemTypes_n,
                             &elemTags, &elemTags_nn, &elemTags_n,
                             &elemNodeTags, &elemNodeTags_nn, &elemNodeTags_n,
                             /*dim=*/2, /*tag=*/-1, &ierr);
    check_ierr(ierr, "gmshModelMeshGetElements");

    // First pass: count supported elements and compute total connectivity size.
    // Unsupported gmsh types are warned about once and skipped.
    std::size_t total_elems = 0;
    std::size_t total_conn  = 0;
    for (std::size_t i = 0; i < elemTypes_n; ++i) {
        const int n = nnode_for_gmsh_type(elemTypes[i]);
        if (n > 0) {
            total_elems += elemTags_nn[i];
            total_conn  += elemTags_nn[i] * static_cast<std::size_t>(n);
        } else {
            std::cerr
                << "Warning: ignoring " << elemTags_nn[i]
                << " elements of gmsh type " << elemTypes[i]
                << " (supported: Tri3, Quad4, Quad8)\n";
        }
    }

    d.nelem = static_cast<int>(total_elems);
    d.conn  .assign(total_conn, 0);
    d.offset.assign(d.nelem + 1, 0);
    d.matno .assign(d.nelem, 0);             // default material 0

    // gmsh element tag -> fem element id (0-based). Single map for all types,
    // because fem element ids are now globally unique.
    std::unordered_map<std::size_t, int> elem_tag_to_id;
    elem_tag_to_id.reserve(total_elems);

    int fem_eid    = 0;
    int conn_cursor = 0;
    for (std::size_t ti = 0; ti < elemTypes_n; ++ti) {
        const int gtype = elemTypes[ti];
        const int nnode = nnode_for_gmsh_type(gtype);
        if (nnode == 0) continue;

        const std::size_t  ne   = elemTags_nn[ti];
        const std::size_t* tags = elemTags[ti];
        const std::size_t* gconn = elemNodeTags[ti];

        for (std::size_t e = 0; e < ne; ++e) {
            elem_tag_to_id[tags[e]] = fem_eid;
            d.offset[fem_eid] = conn_cursor;

            for (int k = 0; k < nnode; ++k) {
                // For Quad8 we permute gmsh's (corners,then midpoints) order
                // into ShapeFunc's interleaved order.
                const int src = (gtype == kQuad8) ? kQ8Perm[k] : k;
                const std::size_t nodeTag = gconn[e * nnode + src];

                const auto it = tag_to_id.find(nodeTag);
                if (it == tag_to_id.end())
                    throw std::runtime_error(
                        "Element references unknown node tag (gmsh type "
                        + std::to_string(gtype) + ")");

                d.conn[conn_cursor++] = it->second;
            }
            ++fem_eid;
        }
    }
    d.offset[d.nelem] = conn_cursor;                  // sentinel

    gmshFree(elemTypes);
    for (std::size_t i = 0; i < elemTags_n;     ++i) gmshFree(elemTags[i]);
    gmshFree(elemTags); gmshFree(elemTags_nn);
    for (std::size_t i = 0; i < elemNodeTags_n; ++i) gmshFree(elemNodeTags[i]);
    gmshFree(elemNodeTags); gmshFree(elemNodeTags_nn);

    // ----- 3. Materials ----------------------------------------------------
    d.props.assign(d.nmats, std::vector<double>(d.nprop, 0.0));
    for (int m = 0; m < d.nmats; ++m) {
        if (static_cast<int>(spec.materials[m].props.size()) != d.nprop)
            throw std::runtime_error("All materials must have the same nprop");
        for (int p = 0; p < d.nprop; ++p)
            d.props[m][p] = spec.materials[m].props[p];
    }

    // Precompute the per-material elasticity cache (D_d, full C, and the
    // constant C_plus / C_minus pairs for the chosen split). With this in
    // hand the Gauss-point loop in elementSystem never rebuilds those 3x3
    // matrices -- it only does strain-dependent scalar / mat-vec work.
    d.mat_caches.clear();
    d.mat_caches.reserve(d.nmats);
    for (int m = 0; m < d.nmats; ++m) {
        // props convention: { E, nu, Gc, l0, k }; only E and nu are used here.
        const double E_m  = d.props[m][0];
        const double nu_m = d.props[m][1];
        d.mat_caches.push_back(
            pfm::MatCache::build(E_m, nu_m, d.ntype, d.split));
    }

    // Apply material_for_group: every element belonging to a 2D physical
    // group is tagged with the requested material id.
    for (const auto& pair : spec.material_for_group) {
        const std::string& name    = pair.first;
        const int          mat_idx = pair.second;
        if (mat_idx < 0 || mat_idx >= d.nmats)
            throw std::runtime_error("material index out of range for group " + name);

        const int phys_tag = find_physical_tag(2, name);
        if (phys_tag < 0)
            throw std::runtime_error("2D physical group not found: " + name);

        // Get all CAD entities making up this physical group.
        int*        entTags   = nullptr; //pointer of physical group
        std::size_t entTags_n = 0; //size
        gmshModelGetEntitiesForPhysicalGroup(2, phys_tag, &entTags, &entTags_n, &ierr);
        check_ierr(ierr, "gmshModelGetEntitiesForPhysicalGroup");

        for (std::size_t e = 0; e < entTags_n; ++e) {
            int*          eTypes  = nullptr; std::size_t eTypes_n  = 0;
            std::size_t** eTags   = nullptr; std::size_t* eTags_nn = nullptr;
                                             std::size_t eTags_n   = 0;
            std::size_t** eNT     = nullptr; std::size_t* eNT_nn   = nullptr;
                                             std::size_t eNT_n     = 0;
            //Get element information of physical group
            gmshModelMeshGetElements(&eTypes, &eTypes_n,
                                     &eTags, &eTags_nn, &eTags_n,
                                     &eNT,   &eNT_nn,   &eNT_n,
                                     /*dim=*/2, entTags[e], &ierr);
            check_ierr(ierr, "gmshModelMeshGetElements (per entity)");

            for (std::size_t ti = 0; ti < eTypes_n; ++ti) {
                const std::size_t  ne   = eTags_nn[ti];
                const std::size_t* tags = eTags[ti];
                for (std::size_t k = 0; k < ne; ++k) {
                    const auto it = elem_tag_to_id.find(tags[k]);
                    if (it != elem_tag_to_id.end())
                    //assaign to physical group elements the corresponding material
                        d.matno[it->second] = mat_idx;
                    // tags absent from the map are unsupported element types
                    // already warned about; ignore.
                }
            }

            gmshFree(eTypes);
            for (std::size_t ti = 0; ti < eTags_n; ++ti) gmshFree(eTags[ti]);
            gmshFree(eTags); gmshFree(eTags_nn);
            for (std::size_t ti = 0; ti < eNT_n;   ++ti) gmshFree(eNT[ti]);
            gmshFree(eNT); gmshFree(eNT_nn);
        }
        gmshFree(entTags);
    }

    // ----- 4. Boundary conditions -----------------------------------------
    // Many BCSpecs can touch the same node (corners, shared edges). Collect
    // per-node flags/values in a map first, then flatten to FemInput's arrays.
    struct NodeBC { std::vector<int> flags; std::vector<double> values; };
    std::unordered_map<int, NodeBC> node_bc;

    for (const auto& bc : spec.bcs) {
        if (static_cast<int>(bc.flags.size())  != d.ndofn ||
            static_cast<int>(bc.values.size()) != d.ndofn)
            throw std::runtime_error("BCSpec '" + bc.physical_name +
                                     "' flags/values must have size ndofn");

        // A Dirichlet BC may name an EDGE (1D) or a POINT (0D). 1D is tried
        // first so a mesh carrying both under one name keeps its old meaning.
        int bc_dim = -1;
        const int phys_tag =
            find_physical_tag_multi({1, 0}, bc.physical_name, bc_dim);
        if (phys_tag < 0)
            throw std::runtime_error(
                "physical group not found for [[bcs]] '" + bc.physical_name +
                "' (searched dimension 1 (edges) then 0 (points)). Check the "
                "name matches the mesh, and that the group is not 2D.");

        if (bc_dim == 0) {
            // Pinning a single node is a legitimate idealisation -- it is what
            // the textbook three-point-bend figure draws -- but it concentrates
            // the entire reaction on one node. The resulting stress
            // singularity sits exactly where the reaction is measured, and in
            // a phase-field model it can nucleate damage under the support
            // rather than at the notch. That failure looks like a result, so
            // say something rather than let it pass silently.
            std::cout << "[bc] '" << bc.physical_name
                      << "' resolved to a 0D group (point restraint). The "
                         "reaction is carried by a single node; if damage "
                         "appears there rather than at the notch, use a short "
                         "edge (\"pad\") instead.\n";
        }

        std::size_t* nTags   = nullptr; std::size_t nTags_n   = 0;
        double*      nCoord  = nullptr; std::size_t nCoord_n  = 0;
        gmshModelMeshGetNodesForPhysicalGroup(bc_dim, phys_tag,
                                              &nTags, &nTags_n,
                                              &nCoord, &nCoord_n,
                                              &ierr);
        check_ierr(ierr, "gmshModelMeshGetNodesForPhysicalGroup");

        if (nTags_n == 0)
            std::cerr << "[bc] WARNING: group '" << bc.physical_name
                      << "' exists but contains no nodes; this BC does nothing.\n";

        for (std::size_t i = 0; i < nTags_n; ++i) {
            const auto it = tag_to_id.find(nTags[i]);
            if (it == tag_to_id.end()) continue;        // shouldn't happen
            NodeBC& nbc = node_bc[it->second];
            if (nbc.flags.empty()) { //if flag empty node is free
                nbc.flags .assign(d.ndofn, 0);
                nbc.values.assign(d.ndofn, 0.0);
            }
            for (int k = 0; k < d.ndofn; ++k) {
                if (bc.flags[k]) {
                    nbc.flags [k] = 1;
                    nbc.values[k] = bc.values[k];   // last writer wins
                }
            }
        }

        gmshFree(nTags); gmshFree(nCoord);
    }

    d.nvfix = static_cast<int>(node_bc.size());  // number of constrained (Dirichlet) nodes
    d.nofix.assign(d.nvfix, 0);  // nofix[i] = node id of the i-th constrained node, size nvfix
    d.iffix.assign(d.nvfix, std::vector<int>   (d.ndofn, 0)); // flags of fixed nodes
    d.fixed.assign(d.nvfix, std::vector<double>(d.ndofn, 0.0)); //values of fixed nodes

    int row = 0;
    for (const auto& kv : node_bc) {
        d.nofix[row] = kv.first;
        d.iffix[row] = kv.second.flags;
        d.fixed[row] = kv.second.values;
        ++row;
    }

    // ----- 5. Neumann boundary conditions ---------------------------------
    // For each NeumannSpec we walk every CAD entity in the named 1D physical
    // group, ask gmsh for the line elements on that entity, translate gmsh
    // node tags to fem ids, and emit one NeumannEdge per line element. The
    // traction vector [tx, ty] is the same for every edge in the group.
    for (const auto& nb : spec.neumann) {
        if (static_cast<int>(nb.traction.size()) != d.ndofn)
            throw std::runtime_error("NeumannSpec '" + nb.physical_name +
                                     "' traction must have size ndofn");

        const int phys_tag = find_physical_tag(1, nb.physical_name);
        if (phys_tag < 0)
            throw std::runtime_error("1D physical group not found (neumann): "
                                     + nb.physical_name);

        int*        entTags   = nullptr;
        std::size_t entTags_n = 0;
        gmshModelGetEntitiesForPhysicalGroup(1, phys_tag, &entTags, &entTags_n, &ierr);
        check_ierr(ierr, "gmshModelGetEntitiesForPhysicalGroup (neumann)");

        for (std::size_t e = 0; e < entTags_n; ++e) {
            int*          eTypes  = nullptr; std::size_t eTypes_n  = 0;
            std::size_t** eTags   = nullptr; std::size_t* eTags_nn = nullptr;
                                             std::size_t eTags_n   = 0;
            std::size_t** eNT     = nullptr; std::size_t* eNT_nn   = nullptr;
                                             std::size_t eNT_n     = 0;

            gmshModelMeshGetElements(&eTypes, &eTypes_n,
                                     &eTags, &eTags_nn, &eTags_n,
                                     &eNT,   &eNT_nn,   &eNT_n,
                                     /*dim=*/1, entTags[e], &ierr);
            check_ierr(ierr, "gmshModelMeshGetElements (neumann, per entity)");

            for (std::size_t ti = 0; ti < eTypes_n; ++ti) {
                const int gtype = eTypes[ti];
                const int nnode = nnode_for_line_type(gtype);
                if (nnode == 0) {
                    // Higher-order or unsupported line; skip with a notice.
                    std::cerr
                        << "Warning: ignoring " << eTags_nn[ti]
                        << " line elements of gmsh type " << gtype
                        << " on Neumann group \"" << nb.physical_name
                        << "\" (supported: Line2, Line3)\n";
                    continue;
                }

                const std::size_t  ne   = eTags_nn[ti];
                const std::size_t* gconn = eNT[ti];

                for (std::size_t k = 0; k < ne; ++k) {
                    FemInput::NeumannEdge edge;
                    edge.nodes.resize(nnode);
                    for (int j = 0; j < nnode; ++j) {
                        const std::size_t nodeTag = gconn[k * nnode + j];
                        const auto it = tag_to_id.find(nodeTag);
                        if (it == tag_to_id.end())
                            throw std::runtime_error(
                                "Neumann edge references unknown node tag");
                        edge.nodes[j] = it->second;
                    }
                    edge.tx = nb.traction[0];
                    edge.ty = nb.traction[1];
                    d.neumann_edges.push_back(std::move(edge));
                }
            }

            gmshFree(eTypes);
            for (std::size_t ti = 0; ti < eTags_n; ++ti) gmshFree(eTags[ti]);
            gmshFree(eTags); gmshFree(eTags_nn);
            for (std::size_t ti = 0; ti < eNT_n;   ++ti) gmshFree(eNT[ti]);
            gmshFree(eNT); gmshFree(eNT_nn);
        }
        gmshFree(entTags);
    }

    // ----- 6. Concentrated nodal point loads ------------------------------
    // Each PointLoadSpec binds a Cartesian force to a 0D physical group.
    // Unlike the Neumann edges there is no integration: the force is recorded
    // directly against every mesh node of the group.
    for (const auto& pl : spec.point_loads) {
        if (static_cast<int>(pl.force.size()) != d.ndofn)
            throw std::runtime_error("PointLoadSpec '" + pl.physical_name +
                                     "' force must have size ndofn");

        const int phys_tag = find_physical_tag(0, pl.physical_name);
        if (phys_tag < 0)
            throw std::runtime_error("0D physical group not found (point_load): "
                                     + pl.physical_name);

        std::size_t* nTags  = nullptr; std::size_t nTags_n  = 0;
        double*      nCoord = nullptr; std::size_t nCoord_n = 0;
        gmshModelMeshGetNodesForPhysicalGroup(0, phys_tag,
                                              &nTags, &nTags_n,
                                              &nCoord, &nCoord_n,
                                              &ierr);
        check_ierr(ierr, "gmshModelMeshGetNodesForPhysicalGroup (point_load)");

        for (std::size_t i = 0; i < nTags_n; ++i) {
            const auto it = tag_to_id.find(nTags[i]);
            if (it == tag_to_id.end()) continue;        // shouldn't happen
            FemInput::PointLoad load;
            load.node = it->second;
            load.fx   = pl.force[0];
            load.fy   = pl.force[1];
            d.point_loads.push_back(load);
        }

        gmshFree(nTags); gmshFree(nCoord);
    }

    // ----- 7. Initial phi overrides ---------------------------------------
    // Each InitialPhiSpec binds a scalar phi value to every mesh node of a
    // physical group. The group can be 0D (a single point, e.g. a crack
    // tip), 1D (a line, e.g. a notch face) or 2D (a region). We probe
    // dimensions in increasing order and use the first match -- mirroring
    // how a user thinks about "phi = 1 on the crack" without forcing them
    // to declare the dimension.
    for (const auto& ip : spec.initial_phi) {
        int phys_tag = -1;
        int phys_dim = -1;
        for (int dim = 0; dim <= 2; ++dim) {
            const int t = find_physical_tag(dim, ip.physical_name);
            if (t >= 0) { phys_tag = t; phys_dim = dim; break; }
        }
        if (phys_tag < 0)
            throw std::runtime_error(
                "physical group not found (initial_phi): " + ip.physical_name);

        std::size_t* nTags  = nullptr; std::size_t nTags_n  = 0;
        double*      nCoord = nullptr; std::size_t nCoord_n = 0;
        gmshModelMeshGetNodesForPhysicalGroup(phys_dim, phys_tag,
                                              &nTags, &nTags_n,
                                              &nCoord, &nCoord_n,
                                              &ierr);
        check_ierr(ierr,
            "gmshModelMeshGetNodesForPhysicalGroup (initial_phi)");

        for (std::size_t i = 0; i < nTags_n; ++i) {
            const auto it = tag_to_id.find(nTags[i]);
            if (it == tag_to_id.end()) continue;
            FemInput::InitialPhiNode entry;
            entry.node  = it->second;
            entry.value = ip.value;
            d.initial_phi_nodes.push_back(entry);
        }

        gmshFree(nTags); gmshFree(nCoord);
    }

    // Precompute the reference-element tables, now that the connectivity (and
    // therefore the set of element types present) is final. One entry per
    // distinct nnode; the Gauss-point loops read shape functions, reference
    // derivatives and quadrature weights straight out of these instead of
    // recomputing them per element. Throws with a clear message if the mesh
    // contains an element type shapeFunc()/gaussPoints() do not support, or
    // one larger than fem::kMaxNodesPerElement.
    d.ref_elems = fem::buildRefElementTable(d.distinct_nnode(),
                                            d.ngaus, d.ndime);

    return d;
}

}  // namespace fem
