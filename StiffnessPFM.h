#pragma once
//
// Monolithic element + global assembly for the phase-field model of brittle
// fracture. The Newton system at a load step is
//
//        K(u, phi) * delta = - R(u, phi)
//
// Element block layout (size 3*nnode), with g(phi) = (1-phi)^2 + k and the
// energy split (C^+, C^-, sigma^+, psi^+) from ConstitutiveModel.h, and
// H = max(H_stored, psi^+) the crack-driving history (History.h):
//
//   R^u      = INT_V B_u^T ( g(phi) sigma^+ + sigma^- ) dV
//   R^phi    = INT_V [ N ( Gc/l0 phi - 2(1-phi) H )
//                     + Gc l0 B_phi^T grad(phi) ] dV
//
//   K_uu     = INT_V B_u^T ( g(phi) C^+ + C^- ) B_u dV
//   K_uphi   = INT_V -2(1-phi) B_u^T sigma^+ N^T dV
//   K_phiu   = (K_uphi)^T
//   K_phiphi = INT_V [ Gc l0 B_phi^T B_phi + (Gc/l0 + 2 H) N N^T ] dV
//
// DOF ordering inside the (3*nnode) element block:
//   [0 .. 2*nnode)        -> displacement dofs, node-major [u0x,u0y,u1x,...]
//   [2*nnode .. 3*nnode)  -> phase-field dofs [phi0, phi1, ...]
// Global ordering (size 3*npoin): first 2*npoin displacement, last npoin phi.
//
// Material property convention (per material id):  props[m] = {E,nu,Gc,l0,k}.
//
// The element residual and tangent are produced together by elementSystem(),
// in a single Gauss-point loop -- the shape functions, Jacobian, B-matrices
// and the energy split are evaluated exactly once and shared.
//

#include "ConstitutiveModel.h"
#include "FEMINPUT.h"
#include "History.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

namespace pfm {

// Phase-field + elasticity parameters for one material id.
struct MatParams {
    double E  = 0.0;   // Young's modulus
    double nu = 0.0;   // Poisson's ratio
    double Gc = 0.0;   // critical energy release rate
    double l0 = 0.0;   // phase-field length scale
    double k  = 0.0;   // residual stiffness (small, e.g. 1e-9)

    // Unpack from FemInput::props[m] = { E, nu, Gc, l0, k }.
    static MatParams from(const std::vector<double>& props);
};

// Assembled global Newton system, block layout [u | phi], size 3*npoin.
struct GlobalSystem {
    Eigen::SparseMatrix<double> K;   // tangent
    Eigen::VectorXd             R;   // internal-force residual
};

// Element-level physics for ONE element, evaluated in a single Gauss-point
// loop. The internal-force residual 're' (length 3*nnode) is always filled.
// The tangent 'Ke' (3*nnode x 3*nnode) is filled only when want_K is true;
// when want_K is false the (more expensive) tangent block products are
// skipped and Ke is left untouched.
//
//   nodes    : connectivity row (0-based fem node ids), length nnode
//   matno    : index into d.props for this element's material
//   u_elem   : length 2*nnode, [u0x, u0y, u1x, u1y, ...]
//   phi_elem : length nnode
//   H_elem   : length = number of Gauss points; stored history per Gauss point
void elementSystem(const std::vector<int>&    nodes,
                   int                        matno,
                   const FemInput&            d,
                   const Eigen::VectorXd&     u_elem,
                   const Eigen::VectorXd&     phi_elem,
                   const std::vector<double>& H_elem,
                   bool                       want_K,
                   Eigen::MatrixXd&           Ke,
                   Eigen::VectorXd&           re);

// Assemble the global tangent K and residual R together (one pass over the
// elements). The K sparsity pattern is structurally fixed -- every element
// entry is emitted, including numerically-zero ones -- so the solver can
// reuse a single symbolic factorization across iterations and load steps.
//
//   u_global   : length 2*npoin
//   phi_global : length npoin
//   history    : per-element, per-Gauss-point crack-driving history
GlobalSystem
assembleGlobalSystem(const FemInput&        d,
                     const Eigen::VectorXd& u_global,
                     const Eigen::VectorXd& phi_global,
                     const HistoryField&    history);

// ===========================================================================
// Staggered-scheme specialised assemblers.
//
// The staggered scheme alternates two sub-problems, each of which only needs
// ONE block of the coupled tangent. The routines below assemble exactly that
// block at a fraction of the full-coupled cost:
//
//   * U sub-system (phi frozen):
//       size = 2*npoin, contains K_uu only;
//       skips K_uphi, K_phiu, K_phiphi (and R_phi).
//
//   * Phi sub-system (u frozen):
//       size = npoin, contains K_phiphi only;
//       skips K_uu, K_uphi, K_phiu (and R_u) -- the largest savings, because
//       the elasticity triple product B_u^T C B_u is the most expensive per-
//       Gauss-point work and we no longer do it here.
//
// The monolithic scheme keeps using assembleGlobalSystem above; these are
// only consumed by subsolveU / subsolvePhi in Solver.cpp.
// ===========================================================================

// Assembled global system for the displacement sub-problem (size 2*npoin).
struct USystem {
    Eigen::SparseMatrix<double> K;   // (2*npoin) x (2*npoin)
    Eigen::VectorXd             R;   // length 2*npoin
};

// Assembled global system for the phase-field sub-problem (size npoin).
struct PhiSystem {
    Eigen::SparseMatrix<double> K;   // npoin x npoin
    Eigen::VectorXd             R;   // length npoin
};

// Element-level U sub-problem (phi enters as a parameter via g(phi)).
//   Ke_uu : (2*nnode) x (2*nnode)
//   re_u  : length 2*nnode
void elementUSystem(const std::vector<int>&    nodes,
                    int                        matno,
                    const FemInput&            d,
                    const Eigen::VectorXd&     u_elem,
                    const Eigen::VectorXd&     phi_elem,
                    const std::vector<double>& H_elem,
                    bool                       want_K,
                    Eigen::MatrixXd&           Ke_uu,
                    Eigen::VectorXd&           re_u);

// Element-level Phi sub-problem (u enters as a parameter via H = max(H, psi+)).
//   Ke_phiphi : nnode x nnode
//   re_phi    : length nnode
void elementPhiSystem(const std::vector<int>&    nodes,
                      int                        matno,
                      const FemInput&            d,
                      const Eigen::VectorXd&     u_elem,
                      const Eigen::VectorXd&     phi_elem,
                      const std::vector<double>& H_elem,
                      bool                       want_K,
                      Eigen::MatrixXd&           Ke_phiphi,
                      Eigen::VectorXd&           re_phi);

// Global U and Phi sub-system assemblers.
USystem
assembleU_System  (const FemInput&        d,
                   const Eigen::VectorXd& u_global,
                   const Eigen::VectorXd& phi_global,
                   const HistoryField&    history);

PhiSystem
assemblePhi_System(const FemInput&        d,
                   const Eigen::VectorXd& u_global,
                   const Eigen::VectorXd& phi_global,
                   const HistoryField&    history);

// ===========================================================================
// Per-element stress field for post-processing (VTK output).
//
// Effective Cauchy stress at each element, averaged across the element's
// Gauss points:
//     sigma_eff = g(phi) * sigma^+ + sigma^-
// plus the von Mises scalar derived from that effective stress. Stress is
// piecewise constant per element in this output (no nodal projection); the
// driver writes it as CELL_DATA in the VTK file.
// ===========================================================================
struct PerElementStresses {
    std::vector<double> sigma_xx;
    std::vector<double> sigma_yy;
    std::vector<double> sigma_xy;
    std::vector<double> von_mises;
};

PerElementStresses
computeElementStresses(const FemInput&        d,
                       const Eigen::VectorXd& u,
                       const Eigen::VectorXd& phi);

// ===========================================================================
// Total internal energy functional E(u, phi)  (Ambati et al. 2015, Eq. 11;
// Miehe/AT2 normalization, matching the assembled residual in this file):
//
//   E_elastic  = INT_V [ g(phi) psi^+ + psi^- ] dV
//   E_fracture = INT_V  Gc ( phi^2/(2 l0) + (l0/2)|grad phi|^2 ) dV
//   E_total    = E_elastic + E_fracture
//
// This is the scalar whose stationarity gives R^u / R^phi, and the natural
// quantity to monitor staggered-cycle convergence: {E^k} decreases
// monotonically toward the minimizer (see Ambati Sect. 3.4). The ACTUAL
// elastic psi^+(eps) is used here, NOT the history H -- E is the genuine
// stored energy, while H is only the irreversibility device in the residual.
// There is no external-work term (displacement control enters via the
// Dirichlet BC on u, as in the paper's E(u,d)).
// ===========================================================================
struct EnergyParts {
    double elastic  = 0.0;   // INT g(phi) psi^+ + psi^- dV
    double fracture = 0.0;   // INT Gc ( phi^2/(2 l0) + (l0/2)|grad phi|^2 ) dV
    double total() const { return elastic + fracture; }
};

EnergyParts
computeEnergy(const FemInput&        d,
              const Eigen::VectorXd& u,
              const Eigen::VectorXd& phi);

}  // namespace pfm
