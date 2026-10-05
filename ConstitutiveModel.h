#pragma once
//
// Constitutive helpers shared by the PFM element routines (stiffness, residual,
// history). Keep ALL knowledge of the constitutive law in this one module.
//
// Conventions (isotropic linear elasticity, 2D):
//   * Strain Voigt order:  eps   = { eps_xx, eps_yy, gamma_xy }
//     (engineering shear, so gamma_xy = du/dy + dv/dx).
//   * Stress Voigt order:  sigma = { sigma_xx, sigma_yy, tau_xy }.
//   * Degradation       :  g(phi) = (1 - phi)^2 + k,   k a small residual.
//
// The phase field degrades only the "positive" (crack-driving) part sigma^+
// of the stress; sigma^- is protected. Which part is which is set by the
// SplitModel (see below). The element routines call energySplit() to obtain
// sigma^+, sigma^-, their tangents C^+, C^-, and the crack-driving energy
// psi^+ (which feeds the irreversible history field).
//

#include <Eigen/Dense>

namespace pfm {

// Phase-field degradation function:  g(phi) = (1 - phi)^2 + k.
double degradation(double phi, double k);

// ---------------------------------------------------------------------------
// Crack-surface density model (TOML key fem.crack_model). Both models use the
// same degradation g(phi) above; they differ in the local term w(phi) of the
// regularised crack energy  Gamma = Gc/c_w INT ( w(phi)/l0 + l0 |grad phi|^2 ):
//
//   AT2 (default): w = phi^2, c_w = 2    -> Gc ( phi^2/(2 l0) + l0/2 |grad phi|^2 )
//                  Damage starts at any load (no elastic phase); phi stays in
//                  [0,1] on its own.
//   AT1          : w = phi,   c_w = 8/3  -> 3Gc/8 ( phi/l0 + l0 |grad phi|^2 )
//                  Linear-elastic until psi+ reaches 3 Gc/(16 l0); needs the
//                  bound phi >= 0 (enforced by the phi active set).
// ---------------------------------------------------------------------------
enum class CrackModel { AT2, AT1 };

// Gauss-point coefficients of the crack-energy part of the phi equation and
// of the crack energy itself, for the current model:
//   residual :  r_phi += local_R * N + grad * Bphi^T grad(phi)
//   tangent  :  K_phiphi += local_K * N N^T + grad * Bphi^T Bphi
//   energy   :  gamma = energy_local + energy_grad * |grad phi|^2
// (The driving term -2(1-phi) H and its tangent 2H N N^T are added by the
//  caller -- they are the same for both models.)
struct CrackTerms {
    double local_R;
    double local_K;
    double grad;
    double energy_local;
    double energy_grad;
};

inline CrackTerms crackTerms(CrackModel m, double Gc, double l0, double phi)
{
    if (m == CrackModel::AT1)
        return { 3.0 * Gc / (8.0 * l0),           // d/dphi of 3Gc/(8 l0) phi
                 0.0,
                 0.75 * Gc * l0,                  // 2 * 3Gc l0 / 8
                 3.0 * Gc * phi / (8.0 * l0),
                 0.375 * Gc * l0 };
    return { Gc / l0 * phi,                       // AT2
             Gc / l0,
             Gc * l0,
             Gc * phi * phi / (2.0 * l0),
             0.5 * Gc * l0 };
}

// ---------------------------------------------------------------------------
// Strain-energy split model. Selected per run via the TOML key
// `fem.energy_split`.
// ---------------------------------------------------------------------------
enum class SplitModel {
    None,        // 1. no split
    Lancioni,    // 2. deviatoric-volumetric (Lancioni & Royer-Carfagni)
    Amor,        // 3. Amor, Marigo, Maurini
    Spectral     // 4. spectral decomposition of the strain (Miehe)
};

// 2D elastic moduli used by every split model.
struct SplitMaterial {
    double mu  = 0.0;
    double K2D = 0.0;

    static SplitMaterial from(double E, double nu, int ntype);
};

// ---------------------------------------------------------------------------
// Per-material precomputed cache.
//
// Everything in here is a function of the material parameters (E, nu, ntype)
// and the split model -- not of the current strain or iteration. Building it
// ONCE up front and indexing by material id at the Gauss-point level removes
// the per-element-per-iteration cost of rebuilding D_d, Pi and C (and, for
// Amor / Lancioni / None, the closed-form C_plus / C_minus).
//
// The cached C_plus / C_minus pairs depend only on the sign of the volumetric
// strain theta:
//   _tens  -- value of C_plus / C_minus when theta > 0  (tensile  case)
//   _comp  -- value of C_plus / C_minus when theta <= 0 (compressive case)
//
// For Lancioni and None both pairs coincide (theta-independent).
// For Spectral the C_plus / C_minus fields are unused (the tangent is
// finite-differenced from spectralSigmaPlus and depends on the strain).
// ---------------------------------------------------------------------------
struct MatCache {
    SplitMaterial   sm;
    Eigen::Matrix3d C            = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d D_d          = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d C_plus_tens  = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d C_minus_tens = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d C_plus_comp  = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d C_minus_comp = Eigen::Matrix3d::Zero();

    static MatCache build(double E, double nu, int ntype, SplitModel split);
};

// Result of an energy split at one Gauss point.
//
// The fields divide by CONSUMER, not by mathematical consistency:
//   sigma_plus / sigma_minus / C_plus / C_minus  -> momentum balance (u-block)
//   psi_plus                                     -> history H and computeEnergy
//   sigma_plus_split                             -> d(psi_plus)/d(eps), for K_phiu
//
// In the fully anisotropic model sigma_plus_split == sigma_plus. Under the
// HYBRID formulation they deliberately differ: sigma_plus becomes the
// isotropic C*eps (so the u-equation is linear) while psi_plus -- and hence
// sigma_plus_split -- keep the split. That inconsistency IS the hybrid model.
struct EnergySplit {
    Eigen::Vector3d sigma_plus  = Eigen::Vector3d::Zero();
    Eigen::Vector3d sigma_minus = Eigen::Vector3d::Zero();
    Eigen::Matrix3d C_plus      = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d C_minus     = Eigen::Matrix3d::Zero();
    double          psi_plus    = 0.0;

    // d(psi_plus)/d(eps): always the SPLIT positive stress, even under hybrid.
    // Only K_phiu (monolithic) needs it; the staggered path never reads it.
    Eigen::Vector3d sigma_plus_split = Eigen::Vector3d::Zero();
};

// `hybrid` selects the Ambati et al. (2015) hybrid formulation. It is
// orthogonal to `model`: `model` still decides which psi^+ drives the phase
// field, `hybrid` only replaces the u-block quantities by their isotropic
// counterparts. Passing hybrid = false reproduces the previous behaviour
// bit-for-bit.
EnergySplit energySplit(const Eigen::Vector3d& eps,
                        const MatCache&        mc,
                        SplitModel             model,
                        bool                   hybrid = false);

}  // namespace pfm
