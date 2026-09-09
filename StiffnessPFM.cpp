#include "StiffnessPFM.h"

#include "ElementUtils.h"
#include "Jacob2.h"
#include "Profiling.h"
#include "ShapeFunc.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfm {

// ---------------------------------------------------------------------------
// MatParams
// ---------------------------------------------------------------------------
MatParams MatParams::from(const std::vector<double>& props)
{
    if (props.size() < 5)
        throw std::runtime_error(
            "pfm::MatParams::from: expected props = {E, nu, Gc, l0, k} "
            "(at least 5 entries); got " + std::to_string(props.size()));

    MatParams m;
    m.E  = props[0];
    m.nu = props[1];
    m.Gc = props[2];
    m.l0 = props[3];
    m.k  = props[4];
    return m;
}

// ---------------------------------------------------------------------------
// Combined element residual + tangent
//
// One pass over the Gauss points evaluates the shape functions, Jacobian,
// B-matrices and the energy split exactly once, then accumulates:
//   * the internal-force residual r_e   (always)
//   * the monolithic tangent Ke         (only when want_K)
// This replaces the old separate elementStiffness / elementResidual, which
// each repeated the whole Gauss-point pipeline at the same (u, phi).
// ---------------------------------------------------------------------------
void elementSystem(const std::vector<int>&    nodes,
                   int                        matno,
                   const FemInput&            d,
                   const Eigen::VectorXd&     u_elem,
                   const Eigen::VectorXd&     phi_elem,
                   const std::vector<double>& H_elem,
                   bool                       want_K,
                   Eigen::MatrixXd&           Ke,
                   Eigen::VectorXd&           re)
{
    const int nnode = static_cast<int>(nodes.size()); //number of element nodes
    const int ndime = d.ndime; // dimension of the field
    const int ndofu = 2 * nnode; // displacement dofs [u]
    const int ndof  = 3 * nnode; // [u | phi]

    if (u_elem.size()   != ndofu)
        throw std::runtime_error("elementSystem: u_elem size != 2*nnode");
    if (phi_elem.size() != nnode)
        throw std::runtime_error("elementSystem: phi_elem size != nnode");

    // Material parameters for this element. The split model itself is
    // carried by d.split; everything constitutive that doesn't depend on the
    // current strain (D_d, full C, and the constant C_plus / C_minus pairs)
    // comes from the per-material cache d.mat_caches[matno], built once.
    const MatParams      mp = MatParams::from(d.props[matno]);
    const MatCache&      mc = d.mat_caches[matno];

    // Precomputed reference-element data for this element type: quadrature
    // weights, shape functions and reference derivatives. Built once at setup
    // (see FemInput::ref_elems); nothing here depends on WHICH element this is.
    const fem::RefElement& ref = d.ref_elems.get(nnode);

    // Nodal coordinates as an ndime x nnode block on the stack.
    fem::DimNodeMat elcod(ndime, nnode);
    for (int i = 0; i < nnode; ++i) {
        const int node = nodes[i];
        for (int dim = 0; dim < ndime; ++dim)
            elcod(dim, i) = d.coord(node, dim);
    }

    // Residual block accumulators (always).
    fem::UDofVec R_u   = fem::UDofVec::Zero(ndofu);
    fem::NodeVec R_phi = fem::NodeVec::Zero(nnode);

    // Tangent block accumulators (want_K only).
    Eigen::MatrixXd K_uu, K_uphi, K_phiu, K_phiphi;
    if (want_K) {
        K_uu     = Eigen::MatrixXd::Zero(ndofu, ndofu);
        K_uphi   = Eigen::MatrixXd::Zero(ndofu, nnode);
        K_phiu   = Eigen::MatrixXd::Zero(nnode, ndofu);
        K_phiphi = Eigen::MatrixXd::Zero(nnode, nnode);
    }

    if (static_cast<int>(H_elem.size()) != ref.ngp)
        throw std::runtime_error("elementSystem: H_elem size != #Gauss points");

    // Gauss-point scratch, declared once and reused. Fixed-size storage, so
    // the resizes inside jacob2Fast / buildBu never hit the heap.
    fem::DimNodeMat cartd;
    fem::BuMat      Bu;
    fem::DimVec     grad_phi;
    double          djacb = 0.0;

    for (int g = 0; g < ref.ngp; ++g) {
        const std::size_t gi = static_cast<std::size_t>(g);

        // Element-specific mapping: Jacobian, det, Cartesian derivatives.
        // The reference-side quantities (N, dN/dxi, w) come from the table.
        fem::jacob2Fast(elcod, ref.dNr[gi], nnode, ndime, cartd, djacb);

        const double dV = djacb * ref.w[gi];

        const fem::NodeVec&    N    = ref.N[gi];
        fem::buildBu(cartd, nnode, Bu);
        // B_phi IS cartd -- the old buildBphi() only copied it element by
        // element, so we bind a reference and skip the copy.
        const fem::DimNodeMat& Bphi = cartd;

        // Phase-field value / gradient and degradation factor.
        const double phi_g = N.dot(phi_elem);
        const double g_phi = degradation(phi_g, mp.k);
        grad_phi.noalias() = Bphi * phi_elem;                    // size ndime

        // Strain at the Gauss point + energy split -- the shared,
        // once-per-Gauss-point physics.
        const Eigen::Vector3d eps = Bu * u_elem;
        const EnergySplit     sp  = energySplit(eps, mc, d.split, d.hybrid);

        // Crack-driving history: H = max(H_stored, psi^+) (irreversibility).
        const double H_eff = std::max(H_elem[g], sp.psi_plus);

        // ---- residual ----
        // R_u : sigma^+ is degraded by g(phi); sigma^- survives at full value.
        const Eigen::Vector3d sigma_eff = g_phi * sp.sigma_plus + sp.sigma_minus;
        R_u += (Bu.transpose() * sigma_eff) * dV;

        // R_phi : N (Gc/l0 phi - 2(1-phi) H) + Gc l0 Bphi^T grad(phi).
        const double n_scalar = mp.Gc / mp.l0 * phi_g
                              - 2.0 * (1.0 - phi_g) * H_eff;
        R_phi += (n_scalar * N) * dV;
        R_phi += (mp.Gc * mp.l0) * (Bphi.transpose() * grad_phi) * dV;

        // ---- tangent (only if requested) ----
        if (want_K) {
            // K_uu : effective tangent C_eff = g(phi) C^+ + C^-.
            const Eigen::Matrix3d C_eff = g_phi * sp.C_plus + sp.C_minus;
            K_uu += (Bu.transpose() * C_eff * Bu) * dV;

            // Coupling blocks. The two are transposes of each other ONLY in
            // the fully anisotropic model; they differentiate different
            // things and hybrid makes that visible:
            //
            //   K_uphi = dR_u/dphi   = g'(phi) Bu^T sigma^+        (degraded
            //                          stress -> the u-block sigma_plus)
            //   K_phiu = dR_phi/du   = g'(phi) N (dH/du)^T, and H comes from
            //                          psi^+, so this needs the SPLIT stress
            //                          sigma_plus_split.
            //
            // Without hybrid the two coincide and K_phiu == K_uphi^T exactly.
            const fem::UDofVec BtSp  = Bu.transpose() * sp.sigma_plus;
            const fem::UDofVec BtSps = Bu.transpose() * sp.sigma_plus_split;
            K_uphi += (-2.0 * (1.0 - phi_g)) * (BtSp  * N.transpose()) * dV;
            K_phiu += (-2.0 * (1.0 - phi_g)) * (N * BtSps.transpose()) * dV;

            // K_phiphi : Gc l0 Bphi^T Bphi + (Gc/l0 + 2 H) N N^T.
            K_phiphi += (mp.Gc * mp.l0) * (Bphi.transpose() * Bphi) * dV;
            K_phiphi += (mp.Gc / mp.l0 + 2.0 * H_eff)
                        * (N * N.transpose()) * dV;
        }
    }

    // Pack the residual blocks into the [u | phi] element vector.
    re.resize(ndof);
    re.head(ndofu) = R_u;
    re.tail(nnode) = R_phi;

    // Pack the tangent blocks into the [u | phi] element matrix.
    if (want_K) {
        Ke = Eigen::MatrixXd::Zero(ndof, ndof);
        Ke.block(0,     0,     ndofu, ndofu) = K_uu;
        Ke.block(0,     ndofu, ndofu, nnode) = K_uphi;
        Ke.block(ndofu, 0,     nnode, ndofu) = K_phiu;
        Ke.block(ndofu, ndofu, nnode, nnode) = K_phiphi;
    }
}

// ---------------------------------------------------------------------------
// Global assembly
// ---------------------------------------------------------------------------
namespace {

// Pull this element's nodal u/phi values and the global dof map.
//   gdof[0 .. 2*nnode)        -> global displacement dofs
//   gdof[2*nnode .. 3*nnode)  -> global phase-field dofs
void extractElement(const std::vector<int>& nodes,
                    const FemInput&         d,
                    const Eigen::VectorXd&  u_global,
                    const Eigen::VectorXd&  phi_global,
                    Eigen::VectorXd&        u_elem,
                    Eigen::VectorXd&        phi_elem,
                    std::vector<int>&       gdof)
{
    const int nnode = static_cast<int>(nodes.size());
    const int ndofu = 2 * nnode;
    const int npoin = d.npoin;

    u_elem.resize(ndofu);
    phi_elem.resize(nnode);
    gdof.resize(3 * nnode);

    for (int i = 0; i < nnode; ++i) {
        const int node = nodes[i];

        u_elem(2 * i)     = u_global(2 * node);
        u_elem(2 * i + 1) = u_global(2 * node + 1);
        gdof[2 * i]       = 2 * node;
        gdof[2 * i + 1]   = 2 * node + 1;

        phi_elem(i)       = phi_global(node);
        gdof[ndofu + i]   = 2 * npoin + node;
    }
}

}  // namespace

GlobalSystem
assembleGlobalSystem(const FemInput&        d,
                     const Eigen::VectorXd& u_global,
                     const Eigen::VectorXd& phi_global,
                     const HistoryField&    history)
{
    const int npoin = d.npoin;
    const int total = 3 * npoin;            // 2*npoin + npoin

    if (u_global.size()   != 2 * npoin)
        throw std::runtime_error("assembleGlobalSystem: u_global size != 2*npoin");
    if (phi_global.size() != npoin)
        throw std::runtime_error("assembleGlobalSystem: phi_global size != npoin");
    if (static_cast<int>(history.size()) != d.nelem)
        throw std::runtime_error("assembleGlobalSystem: history size != nelem");

    GlobalSystem sys;
    sys.R = Eigen::VectorXd::Zero(total);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(d.nelem) * 12 * 12);

    std::vector<int> nodes;
    Eigen::VectorXd  u_elem, phi_elem, re;
    Eigen::MatrixXd  Ke;
    std::vector<int> gdof;

    {
        prof::Scope _t(prof::ElemAssembly);
        for (int e = 0; e < d.nelem; ++e) {
            nodes.assign(d.conn.begin() + d.offset[e],
                         d.conn.begin() + d.offset[e + 1]);
            extractElement(nodes, d, u_global, phi_global, u_elem, phi_elem, gdof);

            elementSystem(nodes, d.matno[e], d, u_elem, phi_elem,
                          history[e], /*want_K=*/true, Ke, re);

            const int eDof = static_cast<int>(gdof.size());
            for (int a = 0; a < eDof; ++a) {
                sys.R(gdof[a]) += re(a);
                for (int b = 0; b < eDof; ++b) {
                    // No "if (v != 0.0)" guard: every structural entry is emitted,
                    // including numerically-zero ones. This keeps the global
                    // sparsity pattern fixed for the whole run, so the solver can
                    // analyzePattern() once and only re-factorize() per iteration.
                    trips.emplace_back(gdof[a], gdof[b], Ke(a, b));
                }
            }
        }
    }

    {
        prof::Scope _t(prof::SparseBuild);
        sys.K.resize(total, total);
        sys.K.setFromTriplets(trips.begin(), trips.end());
        sys.K.makeCompressed();
    }
    return sys;
}

// ===========================================================================
// Staggered scheme: per-sub-block element + global assemblers
// ===========================================================================

// ---------------------------------------------------------------------------
// Element U sub-system (phi frozen). Skips K_uphi, K_phiu, K_phiphi, R_phi.
// ---------------------------------------------------------------------------
void elementUSystem(const std::vector<int>&    nodes,
                    int                        matno,
                    const FemInput&            d,
                    const Eigen::VectorXd&     u_elem,
                    const Eigen::VectorXd&     phi_elem,
                    const std::vector<double>& H_elem,
                    bool                       want_K,
                    Eigen::MatrixXd&           Ke_uu,
                    Eigen::VectorXd&           re_u)
{
    const int nnode = static_cast<int>(nodes.size());
    const int ndime = d.ndime;
    const int ndofu = 2 * nnode;

    if (u_elem.size()   != ndofu)
        throw std::runtime_error("elementUSystem: u_elem size != 2*nnode");
    if (phi_elem.size() != nnode)
        throw std::runtime_error("elementUSystem: phi_elem size != nnode");

    const MatParams mp = MatParams::from(d.props[matno]);
    const MatCache& mc = d.mat_caches[matno];

    const fem::RefElement& ref = d.ref_elems.get(nnode);

    fem::DimNodeMat elcod(ndime, nnode);
    for (int i = 0; i < nnode; ++i) {
        const int node = nodes[i];
        for (int dim = 0; dim < ndime; ++dim)
            elcod(dim, i) = d.coord(node, dim);
    }

    fem::UDofVec    R_u = fem::UDofVec::Zero(ndofu);
    Eigen::MatrixXd K_uu;
    if (want_K) K_uu = Eigen::MatrixXd::Zero(ndofu, ndofu);

    if (static_cast<int>(H_elem.size()) != ref.ngp)
        throw std::runtime_error("elementUSystem: H_elem size != #Gauss points");

    fem::DimNodeMat cartd;
    fem::BuMat      Bu;
    double          djacb = 0.0;

    for (int g = 0; g < ref.ngp; ++g) {
        const std::size_t gi = static_cast<std::size_t>(g);

        fem::jacob2Fast(elcod, ref.dNr[gi], nnode, ndime, cartd, djacb);
        const double dV = djacb * ref.w[gi];

        const fem::NodeVec& N = ref.N[gi];
        fem::buildBu(cartd, nnode, Bu);

        const double phi_g = N.dot(phi_elem);
        const double g_phi = degradation(phi_g, mp.k);

        const Eigen::Vector3d eps = Bu * u_elem;
        const EnergySplit     sp  = energySplit(eps, mc, d.split, d.hybrid);

        // R_u with the same g(phi) sigma^+ + sigma^- as the monolithic path.
        const Eigen::Vector3d sigma_eff = g_phi * sp.sigma_plus + sp.sigma_minus;
        R_u += (Bu.transpose() * sigma_eff) * dV;

        if (want_K) {
            const Eigen::Matrix3d C_eff = g_phi * sp.C_plus + sp.C_minus;
            K_uu += (Bu.transpose() * C_eff * Bu) * dV;
        }
    }

    re_u = R_u;
    if (want_K) Ke_uu = K_uu;
}

// ---------------------------------------------------------------------------
// Element Phi sub-system (u frozen). Skips K_uu, K_uphi, K_phiu, R_u. The
// elasticity triple product B_u^T C B_u is the dominant per-Gauss-point cost
// in the coupled path -- not paying it here is the main saving over the
// monolithic assembly. The strain (and therefore psi^+ for the history) is
// still required.
// ---------------------------------------------------------------------------
void elementPhiSystem(const std::vector<int>&    nodes,
                      int                        matno,
                      const FemInput&            d,
                      const Eigen::VectorXd&     u_elem,
                      const Eigen::VectorXd&     phi_elem,
                      const std::vector<double>& H_elem,
                      bool                       want_K,
                      Eigen::MatrixXd&           Ke_phiphi,
                      Eigen::VectorXd&           re_phi)
{
    const int nnode = static_cast<int>(nodes.size());
    const int ndime = d.ndime;
    const int ndofu = 2 * nnode;

    if (u_elem.size()   != ndofu)
        throw std::runtime_error("elementPhiSystem: u_elem size != 2*nnode");
    if (phi_elem.size() != nnode)
        throw std::runtime_error("elementPhiSystem: phi_elem size != nnode");

    const MatParams mp = MatParams::from(d.props[matno]);
    const MatCache& mc = d.mat_caches[matno];

    const fem::RefElement& ref = d.ref_elems.get(nnode);

    fem::DimNodeMat elcod(ndime, nnode);
    for (int i = 0; i < nnode; ++i) {
        const int node = nodes[i];
        for (int dim = 0; dim < ndime; ++dim)
            elcod(dim, i) = d.coord(node, dim);
    }

    fem::NodeVec    R_phi = fem::NodeVec::Zero(nnode);
    Eigen::MatrixXd K_phiphi;
    if (want_K) K_phiphi = Eigen::MatrixXd::Zero(nnode, nnode);

    if (static_cast<int>(H_elem.size()) != ref.ngp)
        throw std::runtime_error("elementPhiSystem: H_elem size != #Gauss points");

    fem::DimNodeMat cartd;
    fem::BuMat      Bu;
    fem::DimVec     grad_phi;
    double          djacb = 0.0;

    for (int g = 0; g < ref.ngp; ++g) {
        const std::size_t gi = static_cast<std::size_t>(g);

        fem::jacob2Fast(elcod, ref.dNr[gi], nnode, ndime, cartd, djacb);
        const double dV = djacb * ref.w[gi];

        const fem::NodeVec& N = ref.N[gi];
        fem::buildBu(cartd, nnode, Bu);
        const fem::DimNodeMat& Bphi = cartd;   // B_phi IS cartd

        const double phi_g = N.dot(phi_elem);
        grad_phi.noalias() = Bphi * phi_elem;

        const Eigen::Vector3d eps = Bu * u_elem;
        const EnergySplit     sp  = energySplit(eps, mc, d.split, d.hybrid);

        const double H_eff = std::max(H_elem[g], sp.psi_plus);

        const double n_scalar = mp.Gc / mp.l0 * phi_g
                              - 2.0 * (1.0 - phi_g) * H_eff;
        R_phi += (n_scalar * N) * dV;
        R_phi += (mp.Gc * mp.l0) * (Bphi.transpose() * grad_phi) * dV;

        if (want_K) {
            K_phiphi += (mp.Gc * mp.l0) * (Bphi.transpose() * Bphi) * dV;
            K_phiphi += (mp.Gc / mp.l0 + 2.0 * H_eff)
                        * (N * N.transpose()) * dV;
        }
    }

    re_phi = R_phi;
    if (want_K) Ke_phiphi = K_phiphi;
}

// ---------------------------------------------------------------------------
// Global U sub-system assembly (size 2*npoin). Dirichlet BCs are NOT applied
// here -- subsolveU calls applyDirichlet after this and after any Neumann
// F_ext correction.
// ---------------------------------------------------------------------------
USystem
assembleU_System(const FemInput&        d,
                 const Eigen::VectorXd& u_global,
                 const Eigen::VectorXd& phi_global,
                 const HistoryField&    history)
{
    const int npoin = d.npoin;
    const int total = 2 * npoin;

    if (u_global.size()   != 2 * npoin)
        throw std::runtime_error("assembleU_System: u_global size != 2*npoin");
    if (phi_global.size() != npoin)
        throw std::runtime_error("assembleU_System: phi_global size != npoin");
    if (static_cast<int>(history.size()) != d.nelem)
        throw std::runtime_error("assembleU_System: history size != nelem");

    USystem sys;
    sys.R = Eigen::VectorXd::Zero(total);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(d.nelem) * 8 * 8);

    std::vector<int> nodes;
    Eigen::VectorXd  u_elem, phi_elem, re_u;
    Eigen::MatrixXd  Ke_uu;
    std::vector<int> gdof;          // extractElement fills [u | phi]; we only use the u part.

    {
        prof::Scope _t(prof::ElemAssembly);
        for (int e = 0; e < d.nelem; ++e) {
            nodes.assign(d.conn.begin() + d.offset[e],
                         d.conn.begin() + d.offset[e + 1]);
            extractElement(nodes, d, u_global, phi_global, u_elem, phi_elem, gdof);

            elementUSystem(nodes, d.matno[e], d, u_elem, phi_elem,
                           history[e], /*want_K=*/true, Ke_uu, re_u);

            const int ndofu = 2 * static_cast<int>(nodes.size());
            for (int a = 0; a < ndofu; ++a) {
                sys.R(gdof[a]) += re_u(a);
                for (int b = 0; b < ndofu; ++b) {
                    // Emit every structural entry (as in assembleGlobalSystem)
                    // so the LDLT symbolic analysis can be cached across iters.
                    trips.emplace_back(gdof[a], gdof[b], Ke_uu(a, b));
                }
            }
        }
    }

    {
        prof::Scope _t(prof::SparseBuild);
        sys.K.resize(total, total);
        sys.K.setFromTriplets(trips.begin(), trips.end());
        sys.K.makeCompressed();
    }
    return sys;
}

// ---------------------------------------------------------------------------
// Global Phi sub-system assembly (size npoin). The phase field has no
// Dirichlet constraints, so subsolvePhi consumes (K, R) directly.
// ---------------------------------------------------------------------------
PhiSystem
assemblePhi_System(const FemInput&        d,
                   const Eigen::VectorXd& u_global,
                   const Eigen::VectorXd& phi_global,
                   const HistoryField&    history)
{
    const int npoin = d.npoin;

    if (u_global.size()   != 2 * npoin)
        throw std::runtime_error("assemblePhi_System: u_global size != 2*npoin");
    if (phi_global.size() != npoin)
        throw std::runtime_error("assemblePhi_System: phi_global size != npoin");
    if (static_cast<int>(history.size()) != d.nelem)
        throw std::runtime_error("assemblePhi_System: history size != nelem");

    PhiSystem sys;
    sys.R = Eigen::VectorXd::Zero(npoin);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<std::size_t>(d.nelem) * 4 * 4);

    std::vector<int> nodes;
    Eigen::VectorXd  u_elem, phi_elem, re_phi;
    Eigen::MatrixXd  Ke_phiphi;

    {
        prof::Scope _t(prof::ElemAssembly);
        for (int e = 0; e < d.nelem; ++e) {
            nodes.assign(d.conn.begin() + d.offset[e],
                         d.conn.begin() + d.offset[e + 1]);

            // The phi sub-system is indexed directly by node id, so we build the
            // element u_elem / phi_elem inline rather than reusing extractElement
            // (which constructs a 3*nnode [u | phi] gdof map we don't need here).
            const int nnode = static_cast<int>(nodes.size());
            const int ndofu = 2 * nnode;

            u_elem.resize(ndofu);
            phi_elem.resize(nnode);
            for (int i = 0; i < nnode; ++i) {
                const int node    = nodes[i];
                u_elem(2 * i)     = u_global(2 * node);
                u_elem(2 * i + 1) = u_global(2 * node + 1);
                phi_elem(i)       = phi_global(node);
            }

            elementPhiSystem(nodes, d.matno[e], d, u_elem, phi_elem,
                             history[e], /*want_K=*/true, Ke_phiphi, re_phi);

            for (int a = 0; a < nnode; ++a) {
                sys.R(nodes[a]) += re_phi(a);
                for (int b = 0; b < nnode; ++b) {
                    trips.emplace_back(nodes[a], nodes[b], Ke_phiphi(a, b));
                }
            }
        }
    }

    {
        prof::Scope _t(prof::SparseBuild);
        sys.K.resize(npoin, npoin);
        sys.K.setFromTriplets(trips.begin(), trips.end());
        sys.K.makeCompressed();
    }
    return sys;
}


// ===========================================================================
// Per-element effective stress for post-processing.
//
// At each element, evaluates sigma_eff = g(phi) * sigma^+ + sigma^- at every
// Gauss point and averages the result. The same evaluation also produces the
// von Mises scalar (plane-stress formula -- the standard 2D quick-look; for
// plane strain it underestimates the out-of-plane sigma_z contribution but
// the qualitative picture is unchanged).
//
// The output is one value per element; the driver writes it as CELL_DATA in
// the VTK file. This isn't a hot path -- called once per accepted load step
// from writeVTK -- so we don't bother with the MatCache / B-matrix reuse
// tricks of the assembly loop.
// ===========================================================================
PerElementStresses
computeElementStresses(const FemInput&        d,
                       const Eigen::VectorXd& u,
                       const Eigen::VectorXd& phi)
{
    if (u.size()   != 2 * d.npoin)
        throw std::runtime_error(
            "computeElementStresses: u size != 2*npoin");
    if (phi.size() != d.npoin)
        throw std::runtime_error(
            "computeElementStresses: phi size != npoin");

    PerElementStresses out;
    out.sigma_xx .assign(d.nelem, 0.0);
    out.sigma_yy .assign(d.nelem, 0.0);
    out.sigma_xy .assign(d.nelem, 0.0);
    out.von_mises.assign(d.nelem, 0.0);

    std::vector<int> nodes;
    Eigen::VectorXd  u_elem, phi_elem;

    for (int e = 0; e < d.nelem; ++e) {
        nodes.assign(d.conn.begin() + d.offset[e],
                     d.conn.begin() + d.offset[e + 1]);
        const int nnode = static_cast<int>(nodes.size());
        const int ndime = d.ndime;

        const MatParams mp = MatParams::from(d.props[d.matno[e]]);
        const MatCache& mc = d.mat_caches[d.matno[e]];

        const fem::RefElement& ref = d.ref_elems.get(nnode);

        fem::DimNodeMat elcod(ndime, nnode);
        u_elem  .resize(2 * nnode);
        phi_elem.resize(nnode);

        for (int i = 0; i < nnode; ++i) {
            const int node = nodes[i];
            for (int dim = 0; dim < ndime; ++dim)
                elcod(dim, i) = d.coord(node, dim);
            u_elem(2 * i)     = u(2 * node);
            u_elem(2 * i + 1) = u(2 * node + 1);
            phi_elem(i)       = phi(node);
        }

        const int ngp = ref.ngp;

        fem::DimNodeMat cartd;
        fem::BuMat      Bu;
        double          djacb = 0.0;

        Eigen::Vector3d sigma_avg = Eigen::Vector3d::Zero();
        for (int g = 0; g < ngp; ++g) {
            const std::size_t gi = static_cast<std::size_t>(g);

            fem::jacob2Fast(elcod, ref.dNr[gi], nnode, ndime, cartd, djacb);

            const fem::NodeVec& N = ref.N[gi];
            fem::buildBu(cartd, nnode, Bu);

            const double          phi_g = N.dot(phi_elem);
            const double          g_phi = degradation(phi_g, mp.k);
            const Eigen::Vector3d eps   = Bu * u_elem;
            const EnergySplit     sp    = energySplit(eps, mc, d.split, d.hybrid);

            sigma_avg += g_phi * sp.sigma_plus + sp.sigma_minus;
        }
        sigma_avg /= static_cast<double>(ngp);

        const double sxx = sigma_avg(0);
        const double syy = sigma_avg(1);
        const double sxy = sigma_avg(2);
        const double svM = std::sqrt(sxx * sxx + syy * syy
                                      - sxx * syy + 3.0 * sxy * sxy);

        out.sigma_xx [e] = sxx;
        out.sigma_yy [e] = syy;
        out.sigma_xy [e] = sxy;
        out.von_mises[e] = svM;
    }
    return out;
}

// ===========================================================================
// Total internal energy E(u, phi). Integrates the elastic + fracture density
// over the mesh by Gauss quadrature -- same per-Gauss-point pipeline as the
// assembly, but accumulating energy scalars instead of residual/tangent.
// Not a hot path (called for diagnostics, e.g. once per staggered sweep), so
// the MatCache / B-matrix reuse tricks of the assembly loop aren't needed.
// ===========================================================================
EnergyParts
computeEnergy(const FemInput&        d,
              const Eigen::VectorXd& u,
              const Eigen::VectorXd& phi)
{
    if (u.size()   != 2 * d.npoin)
        throw std::runtime_error("computeEnergy: u size != 2*npoin");
    if (phi.size() != d.npoin)
        throw std::runtime_error("computeEnergy: phi size != npoin");

    prof::Scope _t(prof::Energy);

    EnergyParts E;

    std::vector<int> nodes;
    Eigen::VectorXd  u_elem, phi_elem;

    for (int e = 0; e < d.nelem; ++e) {
        nodes.assign(d.conn.begin() + d.offset[e],
                     d.conn.begin() + d.offset[e + 1]);
        const int nnode = static_cast<int>(nodes.size());
        const int ndime = d.ndime;

        const MatParams mp = MatParams::from(d.props[d.matno[e]]);
        const MatCache& mc = d.mat_caches[d.matno[e]];

        const fem::RefElement& ref = d.ref_elems.get(nnode);

        fem::DimNodeMat elcod(ndime, nnode);
        u_elem  .resize(2 * nnode);
        phi_elem.resize(nnode);

        for (int i = 0; i < nnode; ++i) {
            const int node = nodes[i];
            for (int dim = 0; dim < ndime; ++dim)
                elcod(dim, i) = d.coord(node, dim);
            u_elem(2 * i)     = u(2 * node);
            u_elem(2 * i + 1) = u(2 * node + 1);
            phi_elem(i)       = phi(node);
        }

        fem::DimNodeMat cartd;
        fem::BuMat      Bu;
        fem::DimVec     grad_phi;
        double          djacb = 0.0;

        for (int g = 0; g < ref.ngp; ++g) {
            const std::size_t gi = static_cast<std::size_t>(g);

            fem::jacob2Fast(elcod, ref.dNr[gi], nnode, ndime, cartd, djacb);
            const double dV = djacb * ref.w[gi];

            const fem::NodeVec&    N    = ref.N[gi];
            fem::buildBu(cartd, nnode, Bu);
            const fem::DimNodeMat& Bphi = cartd;   // B_phi IS cartd

            const double phi_g = N.dot(phi_elem);
            const double g_phi = degradation(phi_g, mp.k);
            grad_phi.noalias() = Bphi * phi_elem;

            const Eigen::Vector3d eps = Bu * u_elem;
            const EnergySplit     sp  = energySplit(eps, mc, d.split, d.hybrid);

            // Total elastic density psi = 1/2 eps^T C eps; the split is
            // additive (psi = psi^+ + psi^-), so psi^- = psi - psi^+.
            const double psi_tot   = 0.5 * eps.dot(mc.C * eps);
            const double psi_plus  = sp.psi_plus;
            const double psi_minus = psi_tot - psi_plus;

            // Elastic energy actually stored by the momentum balance.
            //
            // Anisotropic model: only psi^+ is degraded  ->  g*psi^+ + psi^-.
            // Hybrid: the momentum balance degrades the FULL energy, so the
            // functional the u-sub-step minimises is g*psi_tot. Reporting the
            // anisotropic expression under hybrid would monitor a quantity
            // that NEITHER sub-step minimises, and the energy-gamma stopping
            // criterion would lose its monotonicity.
            E.elastic  += (d.hybrid ? g_phi * psi_tot
                                    : g_phi * psi_plus + psi_minus) * dV;
            E.fracture += mp.Gc * ( phi_g * phi_g / (2.0 * mp.l0)
                                  + 0.5 * mp.l0 * grad_phi.squaredNorm() ) * dV;
        }
    }
    return E;
}

}  // namespace pfm
