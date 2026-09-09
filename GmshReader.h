#pragma once

// Build a FemInput directly from the gmsh model that's currently in memory.
//
// Preconditions: gmsh::initialize() has been called, mesh::generate(cfg) has
// produced a 2D mesh, and physical groups have been added. The function reads
// nodes, elements, materials-by-physical-group, Dirichlet BCs, Neumann edge
// tractions and concentrated point loads, and packs them into the FemInput
// struct used by the rest of the FEM code.

#include "FEMINPUT.h"
#include <string>
#include <utility>
#include <vector>

namespace fem {

// Material with an arbitrary list of properties (e.g. {E, nu, Gc, l0, k}).
struct Material {
    std::vector<double> props;
};

// One Dirichlet BC, applied to every node of a physical group.
//
// The group may be 1D (an edge) or 0D (a point). The reader tries dimension 1
// first, then 0, so a name that exists at both keeps its edge meaning.
//
// A 0D group gives a POINT RESTRAINT -- the idealised knife-edge support and
// load point of a textbook three-point-bend figure. It is supported, and it
// prints a note when used, because concentrating the whole reaction on one
// node creates a stress singularity exactly where the reaction is measured;
// in a phase-field model that can nucleate damage under the support instead
// of at the notch. Prefer a short edge ("pad") when that matters.
//
//   physical_name : name of a gmsh physical group of dimension 1 or 0
//   flags         : size = ndofn. flags[k]==1 means dof k is fixed.
//   values        : size = ndofn. values[k] is the prescribed value if fixed.
struct BCSpec {
    std::string         physical_name;
    std::vector<int>    flags;
    std::vector<double> values;
};

// One Neumann BC: a uniform Cartesian surface traction applied to every
// boundary line element belonging to a 1D physical group.
//
//   physical_name : name of a gmsh physical group of dimension 1.
//   traction      : length = ndofn = 2. Cartesian components [tx, ty]
//                   in units of force per unit edge length (MPa*mm).
//                   The solver multiplies by the current load_factor.
struct NeumannSpec {
    std::string         physical_name;
    std::vector<double> traction;
};

// One concentrated nodal point load: a Cartesian force applied directly at
// every node of a 0D physical group (no integration involved).
//
//   physical_name : name of a gmsh physical group of dimension 0 (a point).
//   force         : length = ndofn = 2. Cartesian components [Fx, Fy] in
//                   units of force (N). The solver multiplies by the current
//                   load_factor. If the group contains several points each
//                   one receives the full force.
struct PointLoadSpec {
    std::string         physical_name;
    std::vector<double> force;
};

// One initial phase-field assignment: every mesh node of the given physical
// group has its phi value set to 'value' at the start of the run (step 0).
// The solver then evolves phi freely from that initial state.
//
//   physical_name : name of a gmsh physical group of any dimension (0D, 1D,
//                   or 2D). The GMSH reader probes dimensions in increasing
//                   order and uses the first match.
//   value         : initial phi value to assign (typically 1.0 for a fully
//                   damaged region, 0.0 for intact -- the default for nodes
//                   not covered by any spec).
//
// Note: this only sets the INITIAL phi. Nothing is pinned: a region with
// phi=1 but H=0 may relax back if surrounding energies do not yet hold it.
struct InitialPhiSpec {
    std::string physical_name;
    double      value = 0.0;
};

// Everything the FEM solver needs that gmsh does NOT know about.
struct FemSpec {
    int ntype = 1;     // 1 plane-stress, 2 plane-strain
    int ndofn = 2;     // dofs per node (2 for 2D elasticity: ux, uy)
    int ngaus = 2;     // Gauss points per direction
    int nstre = 3;     // sxx, syy, txy

    // Strain-energy split model (TOML key fem.energy_split). Default = Amor.
    pfm::SplitModel split = pfm::SplitModel::Amor;

    // Hybrid formulation (Ambati et al. 2015), TOML key fem.hybrid.
    // ORTHOGONAL to `split`: it selects which EQUATION sees the split, not
    // which split is used. When true the momentum balance is isotropic
    // (sigma = g(phi) C eps, linear in u) while the phase-field equation is
    // still driven by psi^+ from `split`. Default false = fully anisotropic.
    bool hybrid = false;

    // Material library. props[0] is property 0 (e.g. E), props[1] (e.g. nu)...
    std::vector<Material> materials;

    // Map a 2D physical-group name to an index into 'materials'.
    // Every element in that physical group is tagged with that material id.
    // Elements not covered by any entry get material 0 by default.
    std::vector<std::pair<std::string, int>> material_for_group;

    // Dirichlet BCs, applied per 1D physical group. Multiple BCSpecs can hit
    // the same node (e.g. a corner shared by "Bottom" and "LeftLower"); the
    // results are merged dof-by-dof, with later entries overriding earlier
    // ones for any dof they fix.
    std::vector<BCSpec> bcs;

    // Neumann BCs (surface tractions). Optional; an empty vector means the
    // run is pure-Dirichlet (the legacy behaviour).
    std::vector<NeumannSpec> neumann;

    // Concentrated nodal point loads, applied per 0D physical group.
    // Optional; an empty vector means no point loads.
    std::vector<PointLoadSpec> point_loads;

    // Initial phi overrides applied at step 0, per physical group (any dim).
    // Optional; an empty vector means phi starts at zero everywhere.
    std::vector<InitialPhiSpec> initial_phi;
};

// Build the FemInput from the live gmsh model + the user-supplied spec.
[[nodiscard]] FemInput readFemInputFromGmsh(const FemSpec& spec);

}  // namespace fem
