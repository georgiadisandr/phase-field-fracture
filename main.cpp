// Thin driver around mesh::generate + the PFM solver.
//
// Responsibilities:
//   * load run configuration from a TOML file (path is argv[1])
//   * apply optional CLI overrides on top
//   * own gmsh::initialize / gmsh::finalize
//   * call mesh::generate
//   * write .msh / .vtk
//   * read FemInput from the live gmsh model using the loaded FemSpec
//   * loop load_factor 0 -> 1 in N_steps, calling pfm::solveStep per step
//   * maintain the crack-driving history field (irreversibility)
//   * write a numbered VTK snapshot per step (ParaView reads them as a series)
//   * optionally launch the Gmsh GUI at the end
//
// Usage:
//     phasefield_1 config.toml [--W 80 --steps 30 --no-gui ...]
//
// See Config.h for the full list of recognised CLI overrides and config.toml
// for the TOML schema.

#include "Config.h"
#include "GmshReader.h"
#include "History.h"
#include "MeshGeneration.h"
#include "Output.h"
#include "Profiling.h"
#include "ResidualPFM.h"
#include "Solver.h"

#include <gmsh.h>

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_usage(const char* prog)
{
    std::cout <<
        "Usage: " << prog << " CONFIG.toml [options]\n"
        "\n"
        "  CONFIG.toml         path to TOML run configuration (required)\n"
        "\n"
        "Common CLI overrides (any TOML value can be overridden):\n"
        "  --out NAME          output base filename (no extension)\n"
        "  --mesh-file PATH    read the mesh from a gmsh file (.msh, .geo,\n"
        "                      .step, ...) instead of building the built-in\n"
        "                      specimen; implies mesh.source = \"file\"\n"
        "  --mesh-only         build the mesh and the FEM input, print the\n"
        "                      element/constraint counts, then exit without\n"
        "                      solving. Use it to check an externally authored\n"
        "                      mesh: every physical group named in the config\n"
        "                      must resolve, so a missing or misnamed group\n"
        "                      fails here in seconds instead of mid-solve.\n"
        "  --W val             specimen width\n"
        "  --H val             specimen height (also resets y_crack = H/2\n"
        "                      unless --ycrack is also given)\n"
        "  --ycrack val        crack y-coordinate\n"
        "  --a val             crack length\n"
        "  --hfine val         fine mesh size near the crack\n"
        "  --hfar val          coarse mesh size in the bulk\n"
        "  --steps N           number of load-stepping increments\n"
        "  --gui / --no-gui    force Gmsh GUI on or off after the run\n"
        "  --preview           show the mesh before solving (close the window\n"
        "                      to start the solve, Ctrl+C to abort)\n"
        "  --no-preview        skip the mesh preview\n"
        "  --help, -h          show this message and exit\n";
}

// ---------------------------------------------------------------------------
// Terminal logging: a streambuf that writes every character to TWO underlying
// buffers, so std::cout / std::cerr reach the console AND a log file at once.
// ---------------------------------------------------------------------------
class TeeBuf : public std::streambuf {
public:
    TeeBuf(std::streambuf* primary, std::streambuf* secondary)
        : primary_(primary), secondary_(secondary) {}

protected:
    int overflow(int ch) override {
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);
        const int r1 = primary_   ? primary_->sputc(static_cast<char>(ch)) : ch;
        const int r2 = secondary_ ? secondary_->sputc(static_cast<char>(ch)) : ch;
        return (r1 == traits_type::eof() || r2 == traits_type::eof())
                   ? traits_type::eof() : ch;
    }
    int sync() override {
        const int r1 = primary_   ? primary_->pubsync() : 0;
        const int r2 = secondary_ ? secondary_->pubsync() : 0;
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }

private:
    std::streambuf* primary_;
    std::streambuf* secondary_;
};

// RAII: on construction redirect std::cout / std::cerr through tee buffers that
// also write to `path`; on destruction restore the original buffers FIRST (so
// the streams are valid again before the log file is closed).
class TerminalLogger {
public:
    TerminalLogger(const std::string& path, bool enabled) : enabled_(enabled) {
        if (!enabled_) return;
        file_.open(path, std::ios::out | std::ios::trunc);
        if (!file_) {                       // couldn't open -> run without a log
            enabled_ = false;
            std::cerr << "Warning: could not open log file \"" << path
                      << "\"; continuing without a terminal log.\n";
            return;
        }
        cout_old_ = std::cout.rdbuf();
        cerr_old_ = std::cerr.rdbuf();
        tee_out_  = std::make_unique<TeeBuf>(cout_old_, file_.rdbuf());
        tee_err_  = std::make_unique<TeeBuf>(cerr_old_, file_.rdbuf());
        std::cout.rdbuf(tee_out_.get());
        std::cerr.rdbuf(tee_err_.get());
    }

    ~TerminalLogger() {
        if (!enabled_) return;
        std::cout.flush();
        std::cerr.flush();
        std::cout.rdbuf(cout_old_);
        std::cerr.rdbuf(cerr_old_);
    }

    bool active() const { return enabled_; }

private:
    bool                     enabled_;
    std::ofstream            file_;
    std::streambuf*          cout_old_ = nullptr;
    std::streambuf*          cerr_old_ = nullptr;
    std::unique_ptr<TeeBuf>  tee_out_;
    std::unique_ptr<TeeBuf>  tee_err_;
};

}  // namespace

int main(int argc, char** argv) try
{
    // ---- Make progress output actually arrive as it is produced ------------
    // When stdout is a CONSOLE the C runtime line-buffers, so the log appears
    // live. When stdout is a PIPE -- which is exactly what happens under
    // run_gui.py, or `exe > log.txt` -- it switches to block buffering and
    // holds roughly 4 KB before releasing anything. The run then looks frozen
    // and the progress log arrives in lumps, or all at once at exit if the
    // process is killed.
    //
    // std::cout prints with '\n' rather than std::endl throughout this code
    // (deliberately: endl flushes on every line and is slow), so nothing else
    // forces the issue. unitbuf flushes after each output operation, which for
    // a few lines per load step is free relative to a linear solve, and means
    // a killed or crashed run still has its log up to the last line printed.
    //
    // This must run BEFORE anything is written, hence the very top of main.
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    // ---- CLI: TOML path is mandatory ---------------------------------------
    if (argc < 2 ||
        std::string_view(argv[1]) == "--help" ||
        std::string_view(argv[1]) == "-h")
    {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    // Wall-clock timers (steady_clock so they aren't affected by NTP / DST).
    // t_start  = whole-run baseline
    // t_setup  = stamped after FEM input has been built, just before the
    //            first load step. The difference gives the setup time.
    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    // ---- Load + validate configuration -------------------------------------
    appcfg::AppConfig cfg = appcfg::loadConfigFromToml(argv[1]);
    appcfg::applyCliOverrides(cfg, argc, argv);
    appcfg::validate(cfg);

    // ---- Output directory ---------------------------------------------------
    // Everything this run produces goes under one folder:
    //
    //     <run.output_dir>/<base_name>/          log, CSVs, .msh, mesh .vtk
    //     <run.output_dir>/<base_name>/vtk/      per-step snapshots
    //
    // Created HERE, before the terminal logger is installed, because the log
    // file is the first output opened. Failure is not fatal -- fall back to the
    // working directory so a permissions or sync problem costs tidiness, not
    // the run.
    std::string run_dir = (std::filesystem::path(cfg.output_dir)
                           / cfg.mesh.base_name).string();
    std::string vtk_dir = (std::filesystem::path(run_dir) / "vtk").string();
    {
        std::error_code ec;
        std::filesystem::create_directories(vtk_dir, ec);   // creates run_dir too
        if (ec) {
            std::cerr << "[output] WARNING: could not create \"" << vtk_dir
                      << "\" (" << ec.message()
                      << "); writing everything to the working directory\n";
            run_dir.clear();
            vtk_dir.clear();
        }
    }
    // Join a filename onto a directory, or return it unchanged when the
    // directory is empty (the fallback above).
    const auto inDir = [](const std::string& dir, const std::string& name) {
        return dir.empty() ? name
                           : (std::filesystem::path(dir) / name).string();
    };

    // Per-sweep staggered energy history E^k (Ambati Sect. 3.4). Written only
    // for the staggered scheme; harmless to set otherwise (monolithic ignores
    // it). Filter the CSV by load_factor to recover one step's {E^k} sequence.
    if (cfg.solver.scheme == pfm::SolverScheme::Staggered)
        cfg.solver.energy_csv =
            inDir(run_dir, cfg.mesh.base_name + "_stagger_energy.csv");

    // Mirror all terminal output to "<base_name>_run.txt" (config: write_log).
    // Installed before the first print so the whole run is captured. Restored
    // automatically at scope exit. NOTE: Gmsh's own library messages print at
    // the C level and bypass std::cout, so they appear on the console only.
    const std::string log_path = inDir(run_dir, cfg.mesh.base_name + "_run.txt");
    TerminalLogger logger(log_path, cfg.write_log);
    if (logger.active())
        std::cout << "[log] mirroring terminal output to " << log_path << "\n";

    const bool mesh_from_file = (cfg.mesh.source == mesh::Source::File);

    std::cout << "[config] loaded \"" << argv[1] << "\"\n"
              << "         base_name = " << cfg.mesh.base_name << "\n"
              << "         output    = "
              << (run_dir.empty() ? std::string("(working directory)") : run_dir)
              << "\n";
    if (mesh_from_file) {
        // The geometry knobs are meaningless here and printing them would
        // suggest they had an effect.
        std::cout << "         mesh      from file \"" << cfg.mesh.path
                  << "\"  (geometry/sizing keys ignored)\n";
    } else {
        std::cout << "         geometry  W=" << cfg.mesh.W
                  << "  H=" << cfg.mesh.H
                  << "  a=" << cfg.mesh.a
                  << "  y_crack=" << cfg.mesh.y_crack << "\n"
                  << "         mesh size h_fine=" << cfg.mesh.h_fine
                  << "  h_far="  << cfg.mesh.h_far  << "\n";
    }
    std::cout << "         materials=" << cfg.fem.materials.size()
              << "  bcs=" << cfg.fem.bcs.size()
              << "  N_steps=" << cfg.N_steps << "\n";

    // ---- Mesh: build it, or read someone else's ----------------------------
    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 1);

    if (mesh_from_file) {
        // load_from_file() calls gmsh::open, which creates its own model --
        // so no gmsh::model::add here, or we would leave an empty model in
        // front of the one we actually want.
        mesh::load_from_file(cfg.mesh);
    } else {
        gmsh::model::add(cfg.mesh.base_name);
        mesh::generate(cfg.mesh);
    }

    const std::string msh = inDir(run_dir, cfg.mesh.base_name + ".msh");
    const std::string vtk = inDir(run_dir, cfg.mesh.base_name + ".vtk");
    gmsh::write(msh);
    gmsh::write(vtk);
    std::cout << "Wrote " << msh << " and " << vtk << "\n";

    // ---- Build FEM problem from the live gmsh model + the loaded FemSpec ---
    const FemInput d = fem::readFemInputFromGmsh(cfg.fem);

    std::cout << "[mesh] npoin = " << d.npoin
              << ",  nelem = "      << d.nelem
              << " (tri3="          << d.count_by_nnode(3)
              << ", quad4="         << d.count_by_nnode(4)
              << ", quad8="         << d.count_by_nnode(8)
              << "),  nvfix = "     << d.nvfix << "\n";

    // ---- Optional mesh preview (run.preview_mesh) --------------------------
    // Opens the Gmsh window so the geometry, the refinement transition and the
    // resolved boundary groups can be checked before committing to a run.
    // gmsh::fltk::run() BLOCKS: closing the window starts the solve, Ctrl+C
    // here aborts instead.
    //
    // Deliberately placed AFTER the FEM input is built, so the counts printed
    // above are on screen while you look -- in particular nvfix, which tells
    // you whether the Dirichlet physical groups actually resolved to nodes. A
    // mesh that looks right but reports nvfix = 0 would waste the whole run.
    if (cfg.preview_mesh) {
        std::cout << "\n[preview] run.preview_mesh = true -- opening the Gmsh "
                     "window.\n"
                  << (cfg.mesh_only
                          ? "[preview] CLOSE THE WINDOW to finish, or press "
                            "Ctrl+C here to abort.\n\n"
                          : "[preview] CLOSE THE WINDOW to start the solve, or "
                            "press Ctrl+C here to abort.\n\n");
        std::cout.flush();
        gmsh::fltk::run();
        std::cout << "[preview] window closed"
                  << (cfg.mesh_only ? ".\n" : " -- starting the solve.\n");
    }

    // ---- Mesh-only mode (run.mesh_only / --mesh-only) ----------------------
    // Everything above is the full pre-processing path: the mesh exists, the
    // .msh/.vtk are written, and readFemInputFromGmsh has resolved every
    // physical group the config names. Reaching this point at all is the
    // check -- a missing or misnamed group, or one of the wrong dimension,
    // would already have thrown with the offending name in the message.
    //
    // So this is where an externally authored mesh gets verified against the
    // contract, in seconds, without committing to a solve. Read the counts
    // printed above: nvfix = 0 means the Dirichlet groups matched no nodes,
    // which would waste an entire run.
    if (cfg.mesh_only) {
        std::cout << "\n[mesh-only] pre-processing completed successfully:\n"
                  << "[mesh-only]   every physical group named in the config "
                     "resolved\n"
                  << "[mesh-only]   npoin = " << d.npoin
                  << ", nelem = " << d.nelem
                  << ", nvfix = " << d.nvfix << "\n";
        if (d.nvfix == 0)
            std::cerr << "[mesh-only] WARNING: nvfix = 0 -- the Dirichlet "
                         "groups resolved to no nodes. A real run would have "
                         "nothing holding the specimen.\n";
        std::cout << "[mesh-only] mesh written to " << msh << "\n"
                  << "[mesh-only] exiting without solving "
                     "(run.mesh_only / --mesh-only)\n";

        // Show the mesh unless explicitly running headless. preview_mesh has
        // already opened (and closed) the window, so don't open it twice.
        if (!cfg.preview_mesh && cfg.show_gui) gmsh::fltk::run();

        gmsh::finalize();
        return 0;
    }

    Eigen::VectorXd u   = Eigen::VectorXd::Zero(2 * d.npoin);
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(d.npoin);

    // Apply any [[initial_phi]] overrides resolved by GmshReader: each
    // entry sets phi(node) = value at step 0. The solver evolves phi from
    // there; no Dirichlet pinning is applied, so phi may relax if the
    // surrounding history field cannot hold it.
    if (!d.initial_phi_nodes.empty()) {
        for (const auto& e : d.initial_phi_nodes)
            phi(e.node) = e.value;
        std::cout << "[init] applied initial phi to "
                  << d.initial_phi_nodes.size()
                  << " node(s) from [[initial_phi]]\n";
    }

    // Crack-driving history field, enforcing irreversibility. Starts at zero
    // and is raised after every converged step (see updateHistory).
    pfm::HistoryField history = pfm::makeHistoryField(d);

    // Per-step snapshots live in the nested vtk/ folder created at startup, so
    // the CSVs and the log stay visible at the top of the run directory rather
    // than buried among a couple of thousand .vtk files.
    const auto stepFilename = [&](int step) {
        std::ostringstream ss;
        ss << cfg.mesh.base_name << "_step_"
           << std::setw(3) << std::setfill('0') << step << ".vtk";
        return inDir(vtk_dir, ss.str());
    };

    // ---- Loaded direction --------------------------------------------------
    // Which DOF carries the prescribed loading is DETECTED from the Dirichlet
    // data rather than assumed to be y, so the same code drives both the
    // tension test (uy prescribed on "Top") and the shear test (ux prescribed
    // on "Top", uy pinned to zero).
    //
    // Rule: over every constrained DOF of every fixed node, take the one with
    // the largest |prescribed value|. Zero-valued constraints are supports,
    // not loading, so they never win. With a single loaded edge -- the usual
    // case -- this is exactly the edge you drive.
    int    load_dof = 1;         // 0 = x, 1 = y
    double u_full   = 0.0;       // signed prescribed value at full load
    {
        double biggest[2] = {0.0, 0.0};
        for (int i = 0; i < d.nvfix; ++i)
            for (int k = 0; k < d.ndofn && k < 2; ++k)
                if (d.iffix[i][k] != 0 &&
                    std::abs(d.fixed[i][k]) > std::abs(biggest[k]))
                    biggest[k] = d.fixed[i][k];

        load_dof = (std::abs(biggest[0]) > std::abs(biggest[1])) ? 0 : 1;
        u_full   = biggest[load_dof];

        // Mixed-mode loading is not wrong, but "the" applied displacement is
        // then ambiguous and only one component is logged. Say so loudly.
        if (biggest[0] != 0.0 && biggest[1] != 0.0)
            std::cerr << "[force-disp] WARNING: both ux and uy are prescribed "
                      << "nonzero (ux=" << biggest[0] << ", uy=" << biggest[1]
                      << "). Logging the larger component only ("
                      << (load_dof == 0 ? "ux" : "uy")
                      << "); the reported curve is not the full response.\n";
    }
    const char* dir_name = (load_dof == 0) ? "x" : "y";

    // ---- Force-displacement logging (diagnostic) ---------------------------
    // The reaction on the loaded boundary equals the internal-force residual
    // at the constrained DOFs -- exactly the quantity zeroReactions() discards
    // from the convergence norm. We sum the residual component in the loaded
    // direction over the "loaded" nodes (Dirichlet entries with a nonzero
    // prescribed displacement on that DOF) and pair it with the applied
    // displacement load_factor * u_full, written per converged step. If the
    // force rises, peaks, then drops while displacement keeps increasing (or
    // turns back), that is the snap-back signature.
    std::vector<int> loaded_dof;      // global DOF index of each loaded node
    for (int i = 0; i < d.nvfix; ++i)
        if (d.iffix[i][load_dof] != 0 && d.fixed[i][load_dof] != 0.0)
            loaded_dof.push_back(2 * d.nofix[i] + load_dof);

    std::cout << "[force-disp] loaded direction = " << dir_name
              << ";  logging reaction over " << loaded_dof.size()
              << " loaded node(s);  u_full = " << u_full << "\n";
    if (loaded_dof.empty())
        std::cerr << "[force-disp] WARNING: no loaded node found -- every "
                     "prescribed displacement is zero. The force-displacement "
                     "CSV will contain zeros.\n";

    std::ofstream fd_csv(inDir(run_dir, cfg.mesh.base_name + "_force_disp.csv"));
    // `converged` is 1 for a properly converged increment and 0 for one that
    // was force-accepted after a non-convergence (only possible when
    // run.max_subdivs = 0, or once the subdivision budget is exhausted).
    // Filter on it before trusting any part of the curve.
    //
    // Column names carry the loaded direction, so a tension run still writes
    // applied_uy/reaction_Fy exactly as before and a shear run is not silently
    // mislabelled.
    fd_csv << "step,load_factor,applied_u" << dir_name
           << ",reaction_F" << dir_name << ",max_phi,converged\n";

    // Total reaction in the loaded direction at (u, phi); append one CSV row.
    const auto logForceDisp = [&](int step, double load_factor, bool converged) {
        const Eigen::VectorXd R_full =
            pfm::assembleGlobalResidual(d, u, phi, history);
        double F = 0.0;
        for (int g : loaded_dof) F += R_full(g);
        fd_csv << step << ',' << load_factor << ','
               << (load_factor * u_full) << ',' << F << ','
               << phi.maxCoeff() << ',' << (converged ? 1 : 0) << '\n';
        fd_csv.flush();
    };

    // ---- Quasi-static load stepping ----------------------------------------
    // Walk load_factor from 0 -> 1 in N_steps equal increments. Each call to
    // solveStep finds Newton equilibrium at one load level; (u, phi) carry
    // over as warm-starts. After a converged step the history field is raised
    // so the phase field cannot heal at the next increment. On the first
    // non-converged step we halt.
    const int N_steps = cfg.N_steps;

    pfm::io::writeVTK(stepFilename(0), d, u, phi);
    logForceDisp(0, 0.0, /*converged=*/true);
    std::cout << "[step  0]  load_factor = 0  (initial state written)\n";

    // Setup complete; everything after this is the solve.
    const auto t_solve_start = clock::now();
    {
        using namespace std::chrono;
        const double setup_s = duration<double>(t_solve_start - t_start).count();
        std::cout << "[time] setup (config + mesh + FEM input) = "
                  << std::fixed << std::setprecision(2)
                  << setup_s << " s\n"
                  << std::defaultfloat;
    }

    // Cumulative solve time -- the value reported at the end excludes
    // mesh generation and any post-processing GUI work.
    double solve_seconds = 0.0;

    // ---- Adaptive load-step subdivision ------------------------------------
    // Phase-field fracture is fragile during crack-growth events: a too-large
    // load increment can let subsolveU fail, which feeds spurious psi^+ into
    // the phase field (via H = max(H_stored, psi^+)) and contaminates phi.
    // The standard remedy is to walk load_factor adaptively:
    //   * nominal increment = 1 / N_steps
    //   * on failure: roll u/phi back to the last converged state, halve the
    //                 increment, retry (up to max_subdivs times)
    //   * on success: accept, then -- after a few consecutive successes --
    //                 grow the increment back toward the nominal
    // history is only mutated by main() after an ACCEPTED step, so it never
    // needs an explicit rollback.
    //
    // Depth is configurable via run.max_subdivs. Setting it to 0 disables
    // half-stepping: a non-converged increment is then accepted as-is and the
    // run continues (see the warning in Config.h -- the history commit makes
    // that irreversible).
    const int max_subdivs = cfg.step.max_subdivs;

    if (max_subdivs == 0)
        std::cout << "[step] adaptive half-stepping DISABLED "
                     "(run.max_subdivs = 0): non-converged increments will be "
                     "accepted as-is and will contaminate the history field\n";
    else
        std::cout << "[step] adaptive half-stepping enabled: up to "
                  << max_subdivs << " subdivisions (smallest increment = "
                     "nominal / 2^" << max_subdivs << ")\n";

    // Two-stage displacement control (Ambati-style) vs. the uniform schedule.
    // In two_stage mode the applied displacement u = lf * u_ref is advanced by
    // du_coarse until it reaches u_switch, then by du_fine; the adaptive
    // halving below still applies on non-convergence.
    const bool two_stage   = (cfg.step.mode == appcfg::StepMode::TwoStage);
    const bool three_stage = (cfg.step.mode == appcfg::StepMode::ThreeStage);
    const bool staged      = two_stage || three_stage;
    // Magnitude of the full-load prescribed displacement in whichever direction
    // is being driven -- ux for the shear test, uy for tension.
    const double u_ref     = std::abs(u_full);

    // Uniform mode driven by a displacement increment rather than a step
    // count. run.du is the number the literature quotes and the one that stays
    // meaningful when u_max changes; N_steps silently rescales itself.
    const bool uniform_du = (!staged && cfg.step.du > 0.0);

    if ((staged || uniform_du) && u_ref == 0.0)
        throw std::runtime_error(
            "a displacement-based step schedule (run.du, or "
            "run.step_mode = two_stage / three_stage) needs a nonzero "
            "prescribed Dirichlet displacement (ux or uy); no loaded edge "
            "was found");

    // The staged schedule as a table of (upper bound on u, increment). Written
    // this way rather than as nested if/else so two- and three-stage share ONE
    // implementation -- including the land-exactly-on-the-boundary clamp below,
    // which is easy to get subtly wrong when duplicated per stage.
    struct Stage { double u_end; double du; };
    std::vector<Stage> stages;
    if (two_stage) {
        stages = {{cfg.step.u_switch,  cfg.step.du_coarse},
                  {HUGE_VAL,           cfg.step.du_fine}};
    } else if (three_stage) {
        stages = {{cfg.step.u_switch,  cfg.step.du_coarse},
                  {cfg.step.u_switch2, cfg.step.du_fine},
                  {HUGE_VAL,           cfg.step.du_final}};
    }

    // Stage-dependent NOMINAL load-factor increment for the current u = done*u_ref.
    const auto nominal_inc = [&](double done) -> double {
        if (uniform_du) return cfg.step.du / u_ref;
        if (!staged)    return 1.0 / N_steps;
        const double u_now = done * u_ref;
        for (const Stage& s : stages)
            if (u_now < s.u_end - 1e-15) return s.du / u_ref;
        return stages.back().du / u_ref;
    };

    if (staged) {
        std::cout << "[step] " << (three_stage ? "three" : "two")
                  << "-stage displacement schedule: du = "
                  << cfg.step.du_coarse << " up to u = " << cfg.step.u_switch;
        if (three_stage)
            std::cout << ", then " << cfg.step.du_fine << " up to u = "
                      << cfg.step.u_switch2 << ", then " << cfg.step.du_final;
        else
            std::cout << ", then " << cfg.step.du_fine;
        std::cout << "  (u_ref = " << u_ref << ", N_steps ignored)\n";
    }
    else if (uniform_du)
        std::cout << "[step] uniform schedule: du = " << cfg.step.du
                  << " (u_ref = " << u_ref << ", about "
                  << static_cast<long long>(std::ceil(u_ref / cfg.step.du))
                  << " steps; N_steps ignored)\n";
    else
        std::cout << "[step] uniform schedule: " << N_steps
                  << " equal load_factor increments  (set run.du to give the "
                     "displacement increment instead)\n";

    double lf_done        = 0.0;
    double lf_inc         = nominal_inc(0.0);
    double previous_nominal = lf_inc;
    int    step_index     = 0;       // counts ACCEPTED steps (drives VTK naming)
    int    subdiv_streak  = 0;       // subdivisions used for current failed step
    int    success_streak = 0;       // consecutive successes (drives re-growth)

    const int vtk_every     = cfg.vtk_every;  // write VTK every Nth accepted step
    int       last_vtk_step = 0;              // last step whose VTK was written (0)

    Eigen::VectorXd u_saved   = u;
    Eigen::VectorXd phi_saved = phi;

    while (lf_done < 1.0 - 1e-12) {
        // Current stage's nominal increment (changes when u crosses u_switch).
        const double nominal = nominal_inc(lf_done);
        // Start a new stage with the increment requested for that stage.  This
        // matters when three-stage loading goes from a very fine middle stage
        // to a larger final increment: clamping only downward left lf_inc
        // stuck at du_fine whenever non-converged steps prevented adaptive
        // re-growth, producing thousands of unintended extra steps.
        if (std::abs(nominal - previous_nominal)
                > 1e-14 * std::max(1.0, std::abs(previous_nominal))) {
            lf_inc = nominal;
            previous_nominal = nominal;
            success_streak = 0;
        }
        // Never exceed the stage nominal -- this shrinks lf_inc the moment the
        // schedule steps from the coarse stage into the fine stage.
        if (lf_inc > nominal) lf_inc = nominal;

        double lf_try = std::min(lf_done + lf_inc, 1.0);
        // Staged: don't overshoot a switch displacement -- land exactly on it
        // so each stage boundary is clean. Loops over every boundary, so the
        // third stage gets the same treatment as the second.
        if (staged) {
            const double u_done = lf_done * u_ref;
            const double u_try  = lf_try  * u_ref;
            for (const Stage& s : stages) {
                if (s.u_end == HUGE_VAL) continue;
                if (u_done < s.u_end - 1e-15 && u_try > s.u_end + 1e-15) {
                    lf_try = std::min(s.u_end / u_ref, 1.0);
                    break;                 // first boundary crossed wins
                }
            }
        }

        std::cout << "\n=== trying load_factor = " << lf_try
                  << "  (inc = " << lf_inc
                  << ", step_index = " << (step_index + 1)
                  << ", subdivs = " << subdiv_streak << ") ===\n";

        const auto t_step_start = clock::now();
        // Snapshot the profiler so the per-step report below shows only this
        // step's costs, not the cumulative totals.
        const prof::Registry prof_before = prof::reg();

        const pfm::SolverResult res =
            pfm::solveStep(d, u, phi, history, lf_try, cfg.solver);

        std::cout << "  result: "
                  << (res.converged ? "converged" : "DIVERGED")
                  << " in " << res.iters_used
                  << (cfg.solver.scheme == pfm::SolverScheme::Staggered
                          ? " staggered sweeps," : " Newton iters,")
                  << "  |R_final| = " << res.final_residual
                  << ",  max|u| = "   << u.cwiseAbs().maxCoeff()
                  << ",  max phi = "  << phi.maxCoeff() << "\n";

        // Decide the fate of this increment:
        //   converged                    -> accept normally
        //   not converged, budget left   -> roll back and halve, retry
        //   not converged, no budget     -> force-accept and continue
        //                                   (run.max_subdivs = 0, or the
        //                                    subdivision budget is spent)
        bool forced_accept = false;

        if (!res.converged) {
            const double lf_inc_min =
                (max_subdivs > 0)
                    ? nominal / static_cast<double>(1u << max_subdivs)
                    : nominal;
            const bool can_subdivide = (max_subdivs > 0)
                                    && (subdiv_streak < max_subdivs)
                                    && (lf_inc * 0.5 >= lf_inc_min);

            if (can_subdivide) {
                // Roll u/phi back to the last converged state and retry the
                // same lf_done with a smaller increment. history is untouched.
                u   = u_saved;
                phi = phi_saved;
                success_streak = 0;
                ++subdiv_streak;
                lf_inc *= 0.5;
                std::cerr << "  [adaptive] non-converged step rolled back; "
                          << "increment halved to " << lf_inc
                          << " and retrying.\n";
                continue;
            }

            // No subdivision available -- accept the non-converged state.
            forced_accept = true;
            std::cerr << "  [adaptive] *** WARNING: accepting NON-CONVERGED "
                      << "increment at load_factor = " << lf_try
                      << " (|R| = " << res.final_residual << ", "
                      << (max_subdivs == 0 ? "half-stepping disabled"
                                           : "subdivision budget exhausted")
                      << "). The history field will be committed from an "
                      << "unconverged displacement field and CANNOT be undone; "
                      << "every later step is affected. Rows from here on are "
                      << "flagged converged=0 in the force-displacement CSV.\n";
        }

        {
            // Accept: commit history, snapshot, advance lf_done, write VTK.
            pfm::updateHistory(d, u, history);
            u_saved   = u;
            phi_saved = phi;
            lf_done   = lf_try;
            ++step_index;
            subdiv_streak = 0;
            // A force-accepted step is not evidence the loading is behaving,
            // so it must not feed the increment re-growth trigger below.
            success_streak = forced_accept ? 0 : success_streak + 1;

            // Force-displacement curve is logged EVERY accepted step (cheap).
            logForceDisp(step_index, lf_done, !forced_accept);

            // VTK snapshot only on the stride (step 0 already written before
            // the loop; the final step is guaranteed after the loop).
            if (vtk_every <= 1 || step_index % vtk_every == 0) {
                const std::string fname = stepFilename(step_index);
                pfm::io::writeVTK(fname, d, u, phi);
                last_vtk_step = step_index;
                std::cout << "  [output] wrote " << fname << "\n";
            }

            {
                using namespace std::chrono;
                const double step_s =
                    duration<double>(clock::now() - t_step_start).count();
                solve_seconds += step_s;
                std::cout << "  [time] step " << step_index << " = "
                          << std::fixed << std::setprecision(2)
                          << step_s << " s   (cumulative solve = "
                          << solve_seconds << " s)\n"
                          << std::defaultfloat;

                // Where this step's time actually went. The "accounted" line
                // shows how much is covered by the instrumented regions --
                // a large gap means there is an unmeasured cost somewhere.
                // Printing is gated by run.profile_every (0 = never); the
                // timers themselves always run, so the cumulative summary at
                // the end of main() is available either way.
                if (cfg.profile_every > 0 &&
                    step_index % cfg.profile_every == 0)
                {
                    prof::report(std::cout,
                                 prof::delta(prof::reg(), prof_before),
                                 step_s, "this step");
                }
            }

            // Grow the increment back toward nominal after the loading is
            // clearly behaving (two consecutive successes is a conservative
            // trigger; never exceed the user's nominal step size).
            if (success_streak >= 2 && lf_inc < nominal) {
                lf_inc = std::min(lf_inc * 2.0, nominal);
                std::cout << "  [adaptive] increment grown to " << lf_inc << "\n";
            }
        }
    }

    // Always persist the final converged state, even if it fell between VTK
    // strides. u_saved / phi_saved hold the last accepted (u, phi).
    if (step_index > 0 && step_index != last_vtk_step) {
        const std::string fname = stepFilename(step_index);
        pfm::io::writeVTK(fname, d, u_saved, phi_saved);
        std::cout << "  [output] wrote final " << fname << "\n";
    }

    // Final timing summary -- printed regardless of whether the load
    // stepping completed all N_steps or bailed early on non-convergence.
    {
        using namespace std::chrono;
        const auto   t_end = clock::now();
        const double total_s = duration<double>(t_end - t_start).count();
        std::cout << "\n[time] summary:\n"
                  << std::fixed << std::setprecision(2)
                  << "         solve   = " << solve_seconds << " s\n"
                  << "         total   = " << total_s        << " s\n"
                  << std::defaultfloat;

        // Cumulative cost breakdown over every load step of the run.
        prof::report(std::cout, prof::reg(), solve_seconds,
                     "CUMULATIVE over all load steps");
    }

    if (cfg.show_gui) gmsh::fltk::run();

    gmsh::finalize();
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    try { gmsh::finalize(); } catch (...) {}
    return 1;
}
