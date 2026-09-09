#pragma once
//
// Config.h -- single point of truth for everything the driver needs to run.
//
// AppConfig bundles the existing problem-definition structs:
//   * mesh::Config         (specimen geometry & mesh sizing)
//   * fem::FemSpec         (element type, materials, BCs, group->material map)
//   * pfm::SolverSettings  (Newton tolerances, scheme, verbosity)
// plus a few driver-level knobs (load stepping, GUI flag).
//
// It is populated by two layers, in order:
//   1. loadConfigFromToml(path)   -- parse a TOML file into a fresh AppConfig.
//   2. applyCliOverrides(cfg, argc, argv)
//                                  -- override individual fields from the CLI.
//
// Both layers throw std::runtime_error / std::invalid_argument with a clear
// message on bad input; main() just catches std::exception.
//

#include "MeshGeneration.h"
#include "GmshReader.h"
#include "Solver.h"

#include <string>

namespace appcfg {

// Load-stepping mode.
//   Uniform  : equal increments 0 -> full load. Set run.du to give the
//              DISPLACEMENT increment directly (mm), which is what papers
//              quote and what you actually reason about -- "du = 1e-4" rather
//              than "N_steps = 1150". run.N_steps is the legacy fallback, used
//              only when du is not set, and then the increment is 1/N_steps of
//              the load factor.
//   TwoStage : displacement-controlled two-rate schedule (Ambati-style). The
//              applied displacement is advanced by du_coarse until it reaches
//              u_switch, then by du_fine to the end. The schedule is keyed on
//              the magnitude of the largest prescribed Dirichlet uy (the loaded
//              edge), converted to a load_factor increment internally. N_steps
//              is ignored in this mode. The adaptive halving safety net still
//              applies on non-convergence.
//   ThreeStage: the same, with one more rate --
//              du_coarse up to u_switch, du_fine up to u_switch2, du_final to
//              the end. Use it when the run has three distinct regimes: a
//              cheap elastic ramp, a finely resolved nucleation/peak, then a
//              long softening tail that does not need the peak's resolution.
enum class StepMode { Uniform, TwoStage, ThreeStage };

struct StepControl {
    StepMode mode      = StepMode::Uniform;
    // Uniform mode: displacement increment (mm). <= 0 means "not set", fall
    // back to N_steps. Both are kept so existing configs keep working.
    double   du        = 0.0;
    double   du_coarse = 1.0e-5;   // coarse displacement increment (mm)
    double   u_switch  = 5.0e-3;   // switch displacement (mm)
    double   du_fine   = 1.0e-6;   // fine displacement increment (mm)
    // Third stage (three_stage only): du_fine applies up to u_switch2, then
    // du_final to the end.
    double   u_switch2 = 0.0;      // second switch displacement (mm)
    double   du_final  = 0.0;      // increment after u_switch2 (mm)

    // Adaptive half-stepping depth. On a non-converged increment the driver
    // rolls (u, phi) back to the last converged state and retries with half
    // the increment, up to this many times (smallest increment reached is
    // nominal / 2^max_subdivs).
    //
    //   max_subdivs > 0  -- halving enabled, at most this many subdivisions
    //   max_subdivs == 0 -- halving DISABLED; a non-converged increment is
    //                       accepted as-is and the run continues.
    //
    // WARNING: accepting a non-converged increment is not recoverable. The
    // driver commits the history field after every accepted step, and
    // H = max(H_stored, psi^+) only ever grows -- so spurious psi^+ from an
    // unconverged displacement field is baked in permanently and will steer
    // the crack path for the remainder of the run. The force-displacement CSV
    // carries a `converged` column so the affected steps can be identified.
    int      max_subdivs = 6;
};

struct AppConfig {
    // Mesh + geometry knobs (W, H, a, y_crack, h_fine, h_far, r_fine, r_far,
    // base_name). Defaults come from mesh::Config.
    mesh::Config mesh;

    // Element type, materials library, physical-group-to-material map, and
    // Dirichlet / Neumann BCs. Defaults to an empty spec; TOML must populate it.
    // Dirichlet BCs resolve against 1D (edge) or 0D (point) physical groups;
    // Neumann tractions are 1D only, point loads 0D only.
    fem::FemSpec fem;

    // Newton solver settings (tolerances, scheme, max_iter, max_staggered).
    pfm::SolverSettings solver;

    // Number of equal load increments for load_factor: 0 -> 1.
    // Used only when step.mode == StepMode::Uniform.
    int N_steps = 15;

    // Load-stepping schedule (uniform vs. two-stage displacement control).
    StepControl step;

    // Parent directory for this run's output folder (TOML key run.output_dir,
    // CLI --outdir). Every file the run produces goes to
    //     <output_dir>/<base_name>/          logs, CSVs, .msh, mesh .vtk
    //     <output_dir>/<base_name>/vtk/      per-step snapshots
    // Default "." = beside the config file. Point it at a local path (e.g.
    // "C:/pfm-runs") to keep the hundreds of per-step writes and the CSV
    // flushes off a synced OneDrive folder.
    std::string output_dir = ".";

    // Write a VTK snapshot every `vtk_every` ACCEPTED steps (1 = every step).
    // Step 0 (initial state) and the final converged step are always written.
    int vtk_every = 1;

    // Mirror all terminal output (std::cout / std::cerr) to a per-run text
    // file named "<base_name>_run.txt".
    bool write_log = true;

    // Print the per-step [profile] cost breakdown every `profile_every`
    // ACCEPTED steps.
    //   0  -- never (default). The CUMULATIVE summary is still printed once at
    //         the end of the run, so you keep the profiling data without
    //         13,500 lines of per-step output in the log.
    //   1  -- every step (what you want when tuning something)
    //   N  -- every Nth step
    //
    // Note this only controls PRINTING. The timers themselves cost about 1 us
    // per step against a multi-second step, so there is nothing to gain by
    // compiling them out.
    int profile_every = 0;

    // Stop after the mesh and the FEM input have been built, without solving.
    // (TOML key run.mesh_only, CLI --mesh-only.)
    //
    // This is the "check my mesh" mode. It runs the full pre-processing path
    // -- mesh generation or file load, then readFemInputFromGmsh -- so every
    // physical group referenced by [[bcs]], [[neumann]], [[point_load]],
    // [[initial_phi]] and [[material_for_group]] must resolve exactly as it
    // would in a real run. It then prints the element and constraint counts
    // and exits 0.
    //
    // That makes it the verification step for an externally authored mesh:
    // if a group name is missing or has the wrong dimension you find out in
    // seconds rather than after a long solve. nvfix = 0 in the printed summary
    // means the Dirichlet groups resolved to no nodes at all.
    //
    // The Gmsh window is opened if either preview_mesh or show_gui is set, so
    // --mesh-only --no-gui is a non-interactive check suitable for scripts.
    bool mesh_only = false;

    // Launch the Gmsh GUI after the run.
    bool show_gui = true;

    // Open the Gmsh GUI AFTER meshing but BEFORE the solve, so the geometry,
    // the refinement zone and the resolved boundary groups can be inspected
    // before committing to a run. The call blocks: closing the window starts
    // the solve, Ctrl+C aborts.
    //
    // Independent of show_gui (which fires at the END of the run). Must be
    // false for unattended runs -- a blocking window in a batch job waits
    // forever.
    bool preview_mesh = false;
};

// Parse a TOML file into a fresh AppConfig. Throws on parse or validation
// errors. The TOML schema is documented at the top of the example
// `config.toml` shipped with this project.
[[nodiscard]] AppConfig loadConfigFromToml(const std::string& path);

// Apply CLI overrides on top of an already-loaded AppConfig. Recognised
// flags (all optional, all override the TOML value):
//
//   --out NAME      output base filename (no extension)
//   --mesh-file P   read the mesh from gmsh file P instead of building the
//                   built-in specimen (implies mesh.source = "file")
//   --mesh-only     build the mesh and the FEM input, then exit without solving
//   --W val         specimen width
//   --H val         specimen height (also resets y_crack = H/2 unless --ycrack
//                   is also given)
//   --ycrack val    crack y-coordinate
//   --a val         crack length
//   --hfine val     fine mesh size near the crack
//   --hfar val      coarse mesh size in the bulk
//   --steps N       number of load increments
//   --no-gui        do not open Gmsh GUI at end
//   --gui           force Gmsh GUI on at end
//
// Unknown flags produce a warning on stderr and are ignored. The TOML path
// (argv[1]) is *skipped* by this routine -- main() consumes it separately.
void applyCliOverrides(AppConfig& cfg, int argc, char** argv);

// Sanity-check a fully-populated AppConfig. Throws std::invalid_argument with
// a descriptive message if any field is out of bounds.
void validate(const AppConfig& cfg);

}  // namespace appcfg
