#include "ConstitutiveModel.h"

#include <cmath>
#include <stdexcept>

namespace pfm {

// degradation function g(phi)
double degradation(double phi, double k)
{
    const double one_minus_phi = 1.0 - phi;
    return one_minus_phi * one_minus_phi + k;
}

// ---------------------------------------------------------------------------
// Split material
// ---------------------------------------------------------------------------
SplitMaterial SplitMaterial::from(double E, double nu, int ntype)
{
    SplitMaterial m;
    m.mu = E / (2.0 * (1.0 + nu));
    if (ntype == 1) {                    // plane stress
        m.K2D = E / (2.0 * (1.0 - nu));
    } else if (ntype == 2) {             // plane strain
        m.K2D = E / (2.0 * (1.0 + nu) * (1.0 - 2.0 * nu));
    } else {
        throw std::runtime_error(
            "pfm::SplitMaterial::from: ntype must be 1 (plane stress) "
            "or 2 (plane strain)");
    }
    return m;
}

// ---------------------------------------------------------------------------
// Energy split
// ---------------------------------------------------------------------------
namespace {

// Closed-form positive (crack-driving) stress for the Miehe spectral split,
// as a function of the engineering-Voigt strain eps = [eps_xx, eps_yy, gamma].
// C^+ for the spectral model is obtained by finite-differencing this routine
// (see energySplit); the spectral consistent tangent has no compact closed
// form, and the finite-difference cost here is negligible.
Eigen::Vector3d spectralSigmaPlus(const Eigen::Vector3d& eps,
                                  double mu, double K2D)
{
    const double lam = K2D - mu;                  // 2D Lame's first parameter
    const double a = eps(0), b = eps(1), c = eps(2);   // c = ε_xy
    const double theta = a + b; //trace(ε)
    const double mdev  = 0.5 * (a - b); //deviatoric strain
    const double exy   = 0.5 * c; // tensor shear strain γ_xy
    const double R = std::sqrt(mdev * mdev + exy * exy); //radius in mohr circle

    //eigenvalues of strain
    const double e1 = 0.5 * theta + R;            //1st principal strain 
    const double e2 = 0.5 * theta - R;            //2nd principal strain   

    const double p1 = (e1 > 0.0) ? e1 : 0.0;      // <eps_a>_+
    const double p2 = (e2 > 0.0) ? e2 : 0.0;

    // eps^+ in tensor-Voigt [eps+_xx, eps+_yy, eps+_xy].
    Eigen::Vector3d epsp;
    if (R > 1.0e-13 * (std::abs(theta) + 1.0)) {
        const double s = p1 + p2;
        const double d = p1 - p2;
        epsp(0) = 0.5 * s + 0.5 * (d / R) * mdev;
        epsp(1) = 0.5 * s - 0.5 * (d / R) * mdev;
        epsp(2) = 0.5 * (d / R) * exy;
    } else {
        // Isotropic strain: both eigenvalues equal theta/2.
        const double ph = (0.5 * theta > 0.0) ? 0.5 * theta : 0.0;
        epsp << ph, ph, 0.0;
    }

    const double tp = (theta > 0.0) ? theta : 0.0;   // <tr eps>_+
    Eigen::Vector3d sig;
    sig(0) = lam * tp + 2.0 * mu * epsp(0);
    sig(1) = lam * tp + 2.0 * mu * epsp(1);
    sig(2) =            2.0 * mu * epsp(2);          // = mu * gamma+_xy
    return sig;
}

}  // namespace

// ---------------------------------------------------------------------------
// Per-material cache builder. Runs nmats times when FemInput is built; from
// then on energySplit only does strain-dependent work.
// ---------------------------------------------------------------------------
MatCache MatCache::build(double E, double nu, int ntype, SplitModel split)
{
    MatCache mc;
    mc.sm = SplitMaterial::from(E, nu, ntype);
    const double mu  = mc.sm.mu;
    const double K2D = mc.sm.K2D;

    // Deviatoric stress matrix D_d and full elasticity C, in Voigt.
    mc.D_d <<  mu, -mu, 0.0,
              -mu,  mu, 0.0,
               0.0, 0.0, mu;
    Eigen::Matrix3d Pi;
    Pi << 1.0, 1.0, 0.0,
          1.0, 1.0, 0.0,
          0.0, 0.0, 0.0;
    mc.C = K2D * Pi + mc.D_d;

    const Eigen::Matrix3d Z = Eigen::Matrix3d::Zero();
    switch (split) {
        case SplitModel::None:
            // Degrade the whole energy: C_plus = C, C_minus = 0.
            mc.C_plus_tens  = mc.C;       mc.C_minus_tens = Z;
            mc.C_plus_comp  = mc.C;       mc.C_minus_comp = Z;
            break;

        case SplitModel::Lancioni:
            // Only the deviatoric part is degraded; theta-independent.
            mc.C_plus_tens  = mc.D_d;     mc.C_minus_tens = mc.C - mc.D_d;
            mc.C_plus_comp  = mc.D_d;     mc.C_minus_comp = mc.C - mc.D_d;
            break;

        case SplitModel::Amor:
            // theta > 0 (tensile):  degrade dev + tensile-volumetric.
            mc.C_plus_tens  = mc.C;       mc.C_minus_tens = Z;
            // theta <= 0 (compressive): degrade dev only.
            mc.C_plus_comp  = mc.D_d;     mc.C_minus_comp = mc.C - mc.D_d;
            break;

        case SplitModel::Spectral:
            // C_plus is strain-dependent (finite-differenced at the Gauss
            // point); the cache only supplies mu, K2D, D_d and C here.
            break;
    }
    return mc;
}

// ---------------------------------------------------------------------------
// Strain-dependent part of the split. The constant matrices D_d, C and the
// cached C_plus / C_minus come from the MatCache; we only compute scalars
// and the strain-dependent sigma_plus / psi_plus here. For Spectral the
// closed-form has no constant tangent, so C_plus is finite-differenced.
// ---------------------------------------------------------------------------
EnergySplit energySplit(const Eigen::Vector3d& eps,
                        const MatCache&        mc,
                        SplitModel             model,
                        bool                   hybrid)
{
    const double mu    = mc.sm.mu;
    const double K2D   = mc.sm.K2D;
    const double theta = eps(0) + eps(1);

    Eigen::Vector3d e_voigt;
    e_voigt << 1.0, 1.0, 0.0;

    // Deviatoric strain energy  psi_dev = 1/2 eps^T D_d eps.
    const double psi_dev = 0.5 * eps.dot(mc.D_d * eps);

    EnergySplit s;

    switch (model) {
        case SplitModel::None: {
            s.C_plus     = mc.C_plus_tens;        // = mc.C
            s.C_minus    = mc.C_minus_tens;       // = 0
            s.sigma_plus = mc.C * eps;
            s.psi_plus   = 0.5 * eps.dot(s.sigma_plus);
            break;
        }
        case SplitModel::Lancioni: {
            s.C_plus     = mc.C_plus_tens;        // = mc.D_d
            s.C_minus    = mc.C_minus_tens;       // = mc.C - mc.D_d
            s.sigma_plus = mc.D_d * eps;
            s.psi_plus   = psi_dev;
            break;
        }
        case SplitModel::Amor: {
            const bool   tens  = (theta > 0.0);
            const double t_pos = tens ? theta : 0.0;
            s.C_plus     = tens ? mc.C_plus_tens  : mc.C_plus_comp;
            s.C_minus    = tens ? mc.C_minus_tens : mc.C_minus_comp;
            s.sigma_plus = (K2D * t_pos) * e_voigt + mc.D_d * eps;
            s.psi_plus   = 0.5 * K2D * t_pos * t_pos + psi_dev;
            break;
        }
        case SplitModel::Spectral: {
            // Miehe spectral split: scalar quantities from the strain
            // eigenvalues, tangent by central finite difference.
            const double lam  = K2D - mu;
            const double mdev = 0.5 * (eps(0) - eps(1));
            const double exy  = 0.5 * eps(2);
            const double R    = std::sqrt(mdev * mdev + exy * exy);
            const double e1   = 0.5 * theta + R;//1st principal strain
            const double e2   = 0.5 * theta - R;//2nd principal strain
            const double p1   = (e1 > 0.0) ? e1 : 0.0;
            const double p2   = (e2 > 0.0) ? e2 : 0.0;
            const double tp   = (theta > 0.0) ? theta : 0.0;

            s.sigma_plus = spectralSigmaPlus(eps, mu, K2D);
            s.psi_plus   = 0.5 * lam * tp * tp + mu * (p1 * p1 + p2 * p2);

            // The finite-differenced tangent is the expensive part of this
            // branch (6 extra spectralSigmaPlus evaluations). Under hybrid it
            // is overwritten by mc.C a few lines below, so skip it entirely.
            if (!hybrid) {
                const double h = 1.0e-8;
                for (int j = 0; j < 3; ++j) {
                    Eigen::Vector3d ep = eps, em = eps;
                    ep(j) += h;
                    em(j) -= h;
                    s.C_plus.col(j) =
                        (spectralSigmaPlus(ep, mu, K2D) -
                         spectralSigmaPlus(em, mu, K2D)) / (2.0 * h);
                }
                s.C_minus = mc.C - s.C_plus;      // strain-dependent
            }
            break;
        }
    }

    // psi_plus is now final for every model, so record its exact derivative
    // BEFORE the hybrid override touches sigma_plus. K_phiu needs this one.
    s.sigma_plus_split = s.sigma_plus;

    // -----------------------------------------------------------------------
    // Hybrid formulation (Ambati et al. 2015).
    //
    // The split is kept ONLY in the phase-field equation: psi_plus above is
    // left exactly as the chosen model computed it. The momentum balance
    // reverts to the isotropic form
    //     sigma = g(phi) * C * eps,
    // which makes the u-sub-problem LINEAR in u for frozen phi -- the source
    // of the scheme's speed. Setting C_plus = C and C_minus = 0 reproduces
    // that through the existing C_eff = g*C_plus + C_minus assembly, so no
    // element routine needs to know about hybrid at all.
    //
    // The resulting sigma_plus / psi_plus pair is deliberately NOT a
    // potential-derivative pair; the formulation is variationally
    // inconsistent by construction.
    // -----------------------------------------------------------------------
    if (hybrid) {
        s.sigma_plus = mc.C * eps;
        s.C_plus     = mc.C;
        s.C_minus    = Eigen::Matrix3d::Zero();
    }

    // sigma^- = C eps - sigma^+  (uniform across all splits).
    s.sigma_minus = mc.C * eps - s.sigma_plus;
    return s;
}

}  // namespace pfm
