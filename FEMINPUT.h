#pragma once
//
// FemInput: the data the FEM solver consumes after pre-processing.
//
// Populated by the GMSH reader (GmshReader.cpp) from the live gmsh model
// plus a user-supplied fem::FemSpec, and then read by the element routines
// (Jacob2, StiffnessPFM, ...).
//

#include "ConstitutiveModel.h"
#include "ElementUtils.h"

#include <Eigen/Dense>

#include <vector>

struct FemInput {
    // -----------------------------------------------------------------------
    // Problem-level counts
    // -----------------------------------------------------------------------
    int ntype = 1; // 1 plane-stress, 2 plane-strain
    int ndofn = 2; // dofs per node (2 for 2D elasticity: ux, uy)
    int ngaus = 2; // Gauss points per direction
    int nstre = 3; // sxx, syy, txy
    int ndime = 2; // dimension of Geometry

    pfm::SplitModel split = pfm::SplitModel::Amor;

    // Hybrid formulation (Ambati et al. 2015). Orthogonal to `split`: the
    // momentum balance uses the FULL energy (sigma = g(phi) C eps, linear in
    // u) while the phase field is still driven by psi^+ from `split`.
    bool hybrid = false;

    int npoin = 0; //number of total nodes
    int nelem = 0; //number of elements

    int nmats = 0; //number of materials
    int nprop = 0; //number of material properties

    int nvfix = 0; //number of fixed nodes

    // -----------------------------------------------------------------------
    // Geometry
    // -----------------------------------------------------------------------
    Eigen::MatrixXd coord;            // npoin x ndime

    // -----------------------------------------------------------------------
    // Mixed-element connectivity in CSR form.
    // -----------------------------------------------------------------------
    std::vector<int> conn; // vector with all nodes of all elements 
    std::vector<int> offset; // offset from wihich index starts every element  
    std::vector<int> matno; //material id

    //helper function to tell how many nodes has the nth element
    int nnode_of(int e) const { return offset[e + 1] - offset[e]; } 

    //counts how many different types of elemetns there is
    int count_by_nnode(int n) const {
        int c = 0;
        for (int e = 0; e < nelem; ++e)
            if (nnode_of(e) == n) ++c;
        return c;
    }

    // Sorted, de-duplicated list of the node counts actually present in the
    // mesh -- the set of element types the reference tables must cover.
    std::vector<int> distinct_nnode() const {
        std::vector<int> present;
        for (int e = 0; e < nelem; ++e) {
            const int n = nnode_of(e);
            bool seen = false;
            for (const int p : present) if (p == n) { seen = true; break; }
            if (!seen) present.push_back(n);
        }
        return present;
    }

    // -----------------------------------------------------------------------
    // Precomputed reference-element data
    // -----------------------------------------------------------------------
    // Shape functions N_i(xi_g), reference derivatives dN_i/dxi and quadrature
    // weights for every element TYPE in the mesh, evaluated once at setup and
    // indexed by nnode. These depend only on the reference element, never on a
    // specific element's coordinates, so the Gauss-point loops read them from
    // here instead of calling shapeFunc()/gaussPoints() ~400k times per pass.
    //
    // The element-specific part of the mapping (Jacobian, Cartesian
    // derivatives, det J) is still computed per element, by fem::jacob2Fast().
    fem::RefElementTable ref_elems;

    // -----------------------------------------------------------------------
    // Material library
    // -----------------------------------------------------------------------
    // props[m] = { E, nu, Gc, l0, k }
    std::vector<std::vector<double>> props;

    // Per-material precomputed elasticity (D_d, full C, and the constant
    // C_plus / C_minus pairs for the chosen split). Built once when the
    // FemInput is constructed; indexed by matno in elementSystem so the
    // Gauss-point loop never rebuilds those 3x3 matrices.
    std::vector<pfm::MatCache> mat_caches;

    // -----------------------------------------------------------------------
    // Dirichlet boundary conditions
    // -----------------------------------------------------------------------
    std::vector<int>                 nofix;// nodes id fixed
    std::vector<std::vector<int>>    iffix;// flag of directions 1-> fixed 0->free
    std::vector<std::vector<double>> fixed;// value of fixed node direction ux or uy

    // -----------------------------------------------------------------------
    // Neumann boundary conditions
    // -----------------------------------------------------------------------
    struct NeumannEdge {
        std::vector<int> nodes;
        double tx = 0.0;
        double ty = 0.0;
    };
    std::vector<NeumannEdge> neumann_edges;

    // -----------------------------------------------------------------------
    // Concentrated nodal point loads
    // -----------------------------------------------------------------------
    struct PointLoad {
        int    node = 0;
        double fx   = 0.0;
        double fy   = 0.0;
    };
    std::vector<PointLoad> point_loads;

    // -----------------------------------------------------------------------
    // Initial phase-field overrides (resolved from [[initial_phi]] specs)
    // -----------------------------------------------------------------------
    // Each entry says "set phi(node) = value at the start of the run". The
    // driver (main.cpp) applies these once at step 0; the solver then evolves
    // phi freely -- nothing is pinned. If multiple entries hit the same node
    // later entries override earlier ones in the order produced by GmshReader.
    struct InitialPhiNode {
        int    node  = 0;
        double value = 0.0;
    };
    std::vector<InitialPhiNode> initial_phi_nodes;
};
