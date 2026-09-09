#include "Profiling.h"

#include <iomanip>
#include <ostream>

namespace prof {

const char* const kNames[NCAT] = {
    "element assembly",
    "sparse build",
    "dirichlet",
    "factorize",
    "solve",
    "energy",
    "history",
    "output"
};

Registry& reg()
{
    static Registry g;
    return g;
}

Registry delta(const Registry& now, const Registry& before)
{
    Registry d;
    for (int i = 0; i < NCAT; ++i) {
        d.sec[i]   = now.sec[i]   - before.sec[i];
        d.calls[i] = now.calls[i] - before.calls[i];
    }
    return d;
}

void report(std::ostream& os, const Registry& r, double wall_s,
            const char* label)
{
    double accounted = 0.0;
    for (int i = 0; i < NCAT; ++i) accounted += r.sec[i];

    const double denom = (wall_s > 0.0) ? wall_s : 1.0;

    const std::ios_base::fmtflags saved = os.flags();
    const std::streamsize         prec  = os.precision();

    os << "  [profile] " << label << '\n';
    for (int i = 0; i < NCAT; ++i) {
        if (r.sec[i] <= 0.0 && r.calls[i] == 0) continue;   // skip untouched
        os << "      " << std::left << std::setw(18) << kNames[i]
           << std::right << std::fixed << std::setprecision(3)
           << std::setw(9) << r.sec[i] << " s "
           << std::setw(6) << std::setprecision(1)
           << (100.0 * r.sec[i] / denom) << "%  "
           << std::setw(8) << r.calls[i] << " calls\n";
    }
    os << "      " << std::left << std::setw(18) << "--- accounted"
       << std::right << std::fixed << std::setprecision(3)
       << std::setw(9) << accounted << " s "
       << std::setw(6) << std::setprecision(1)
       << (100.0 * accounted / denom) << "%  of "
       << std::setprecision(3) << wall_s << " s wall\n";

    os.flags(saved);
    os.precision(prec);
}

}  // namespace prof
