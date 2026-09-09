#pragma once
//
// Lightweight wall-clock profiler.
//
// Purpose: find out where a load step actually spends its time, so the
// optimization work can be ordered by measured cost instead of guesswork.
//
// Usage: drop a scoped timer at the top of a region.
//
//     void applyDirichlet(...) {
//         prof::Scope _t(prof::Dirichlet);
//         ...
//     }
//
// or around a sub-block:
//
//     { prof::Scope _t(prof::SparseBuild);
//       K.setFromTriplets(...); K.makeCompressed(); }
//
// IMPORTANT: the categories must not overlap. A Scope started inside another
// Scope's region double-counts that time, and the "accounted" line in the
// report will exceed the measured wall time. Where a function contains two
// distinct phases (element loop, then sparse build), give each its own
// non-overlapping block rather than wrapping the whole function.
//
// Cost: one steady_clock::now() pair per Scope (~40 ns). Scopes are placed
// per-assembly, never per-element, so the overhead is far below the noise.
//

#include <array>
#include <chrono>
#include <cstdint>
#include <iosfwd>

namespace prof {

enum Cat : int {
    ElemAssembly = 0,  // element kernels + scatter (assemble*_System, residual)
    SparseBuild,       // setFromTriplets + makeCompressed
    Dirichlet,         // applyDirichlet (its own matrix rebuild)
    Factorize,         // analyzePattern + numeric factorize
    Solve,             // triangular solves (solver.solve)
    Energy,            // computeEnergy
    History,           // updateHistory
    Output,            // writeVTK (incl. computeElementStresses) + CSV
    NCAT
};

extern const char* const kNames[NCAT];

struct Registry {
    std::array<double,      NCAT> sec{};
    std::array<std::int64_t, NCAT> calls{};
};

// Process-wide accumulator.
Registry& reg();

// now - before, componentwise. Used to report one step in isolation.
Registry delta(const Registry& now, const Registry& before);

// Print a breakdown. wall_s is the externally measured wall time for the same
// region, so the report can show how much time is NOT accounted for -- that
// gap is where the next unknown cost is hiding.
void report(std::ostream& os, const Registry& r, double wall_s,
            const char* label);

class Scope {
public:
    explicit Scope(Cat c)
        : c_(c), t0_(std::chrono::steady_clock::now()) {}

    ~Scope() {
        const auto t1 = std::chrono::steady_clock::now();
        Registry& g = reg();
        g.sec[c_] += std::chrono::duration<double>(t1 - t0_).count();
        ++g.calls[c_];
    }

    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Cat                                            c_;
    std::chrono::steady_clock::time_point          t0_;
};

}  // namespace prof
