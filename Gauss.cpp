
#include "Gauss.h"
#include <stdexcept>
#include <cmath>

namespace quadrature {

// ---------------------------------------------------------------------------
// Triangle rules
// Reference triangle area = 1/2; weights below sum to 1/2.
// ---------------------------------------------------------------------------
TriGauss triGauss(int ngaus) {
    TriGauss out;
    out.xi.assign(ngaus, 0.0);
    out.eta.assign(ngaus, 0.0);
    out.w.assign(ngaus, 0.0);

    if (ngaus == 1) {
        // Degree 1: centroid
        out.xi[0]  = 1.0 / 3.0;
        out.eta[0] = 1.0 / 3.0;
        out.w[0]   = 0.5;
        return out;
    }

    if (ngaus == 3) {
        // Degree 2: midpoints of the three edges, equal weights 1/6.
        out.xi[0]  = 0.5; out.eta[0] = 0.0;
        out.xi[1]  = 0.5; out.eta[1] = 0.5;
        out.xi[2]  = 0.0; out.eta[2] = 0.5;
        out.w[0] = out.w[1] = out.w[2] = 1.0 / 6.0;
        return out;
    }

    if (ngaus == 7) {
        // Degree 5: Dunavant rule. 1 centroid + two cyclic groups of 3 interior
        // points. Reference: D.A. Dunavant, "High Degree Efficient Symmetrical
        // Gaussian Quadrature Rules for the Triangle" (1985).
        const double s15 = std::sqrt(15.0);
        const double a1  = (6.0 - s15) / 21.0;     // ~ 0.1012865073234563
        const double a2  = (6.0 + s15) / 21.0;     // ~ 0.4701420641051151
        const double w0  = 9.0 / 80.0;             // centroid weight
        const double w1  = (155.0 - s15) / 2400.0; // weight for the a1 group
        const double w2  = (155.0 + s15) / 2400.0; // weight for the a2 group

        // Centroid
        out.xi[0]  = 1.0 / 3.0;        out.eta[0] = 1.0 / 3.0;        out.w[0] = w0;

        // a1 group: three cyclic permutations of barycentric (a1, a1, 1-2*a1)
        out.xi[1]  = a1;               out.eta[1] = a1;               out.w[1] = w1;
        out.xi[2]  = 1.0 - 2.0 * a1;   out.eta[2] = a1;               out.w[2] = w1;
        out.xi[3]  = a1;               out.eta[3] = 1.0 - 2.0 * a1;   out.w[3] = w1;

        // a2 group: three cyclic permutations of barycentric (a2, a2, 1-2*a2)
        out.xi[4]  = a2;               out.eta[4] = a2;               out.w[4] = w2;
        out.xi[5]  = 1.0 - 2.0 * a2;   out.eta[5] = a2;               out.w[5] = w2;
        out.xi[6]  = a2;               out.eta[6] = 1.0 - 2.0 * a2;   out.w[6] = w2;
        return out;
    }

    throw std::invalid_argument(
        "triGauss: unsupported ngaus. Supported values are 1, 3, 7.");
}

// ---------------------------------------------------------------------------
// 1D Gauss-Legendre rules on [-1, 1] (use tensor product for 2D quads).
// Weights below sum to 2 (the length of the reference interval).
// ---------------------------------------------------------------------------
QuadGauss quadGauss(int ngaus) {
    QuadGauss out;
    out.xi.assign(ngaus, 0.0);
    out.w.assign(ngaus, 0.0);

    switch (ngaus) {
        case 1:
            out.xi[0] = 0.0;
            out.w[0]  = 2.0;
            return out;

        case 2:
            out.xi[0] = -0.5773502691896257;
            out.xi[1] =  0.5773502691896257;
            out.w[0]  = 1.0;
            out.w[1]  = 1.0;
            return out;

        case 3:
            out.xi[0] = -0.7745966692414834;
            out.xi[1] =  0.0;
            out.xi[2] =  0.7745966692414834;
            out.w[0]  = 5.0 / 9.0;
            out.w[1]  = 8.0 / 9.0;
            out.w[2]  = 5.0 / 9.0;
            return out;

        case 4:
            out.xi[0] = -0.8611363115940526;
            out.xi[1] = -0.3399810435848563;
            out.xi[2] =  0.3399810435848563;
            out.xi[3] =  0.8611363115940526;
            out.w[0]  = 0.3478548451374538;
            out.w[1]  = 0.6521451548625461;
            out.w[2]  = 0.6521451548625461;
            out.w[3]  = 0.3478548451374538;
            return out;

        case 5:
            out.xi[0] = -0.9061798459386640;
            out.xi[1] = -0.5384693101056831;
            out.xi[2] =  0.0;
            out.xi[3] =  0.5384693101056831;
            out.xi[4] =  0.9061798459386640;
            out.w[0]  = 0.2369268850561891;
            out.w[1]  = 0.4786286704993665;
            out.w[2]  = 0.5688888888888889;
            out.w[3]  = 0.4786286704993665;
            out.w[4]  = 0.2369268850561891;
            return out;

        case 6:
            out.xi[0] = -0.9324695142031521;
            out.xi[1] = -0.6612093864662645;
            out.xi[2] = -0.2386191860831969;
            out.xi[3] =  0.2386191860831969;
            out.xi[4] =  0.6612093864662645;
            out.xi[5] =  0.9324695142031521;
            out.w[0]  = 0.1713244923791704;
            out.w[1]  = 0.3607615730481386;
            out.w[2]  = 0.4679139345726910;
            out.w[3]  = 0.4679139345726910;
            out.w[4]  = 0.3607615730481386;
            out.w[5]  = 0.1713244923791704;
            return out;

        default:
            throw std::invalid_argument(
                "quadGauss: unsupported ngaus. Supported values are 1, 2, 3, 4, 5, 6.");
    }
}

} // namespace quadrature
