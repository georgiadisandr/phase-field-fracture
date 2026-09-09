# Phase-Field Fracture FEM

A C++17 finite element project for two-dimensional, quasi-static phase-field fracture simulations, with Python desktop interfaces for configuring runs and viewing results. The CMake project is named `PhaseFieldFEM1` and builds the executable `phasefield_1`.

The solver couples displacement and damage fields using an AT2 fracture formulation with quadratic degradation. It includes a built-in single-edge-notched specimen and can read external Gmsh meshes, including the supplied three-point-bending example.

## Capabilities

- Plane stress and plane strain with two displacement degrees of freedom per node.
- Tri3, Quad4, and Quad8 elements.
- Energy splits: `none`, `lancioni`, `amor`, and `spectral`.
- Optional hybrid formulation with isotropically degraded momentum balance and a separately selected crack-driving energy split.
- Monolithic Newton and staggered displacement/phase-field solution schemes.
- Residual-based or energy-based staggered stopping criteria.
- Uniform or two-stage displacement stepping, with configurable increment subdivision.
- Materials, displacement constraints, tractions, point loads, and initial damage assigned through named Gmsh physical groups.
- VTK solution snapshots, force-displacement CSVs, staggered energy histories, and timing/log output.

## Requirements

The current build configuration targets Windows and uses Windows-specific Gmsh library paths.

| Component | Purpose |
| --- | --- |
| CMake 3.16 or newer | Configure and build the C++ executable |
| C++17 compiler | The existing build uses MSYS2 UCRT64 MinGW GCC |
| Eigen headers | Linear algebra; the configured local version is Eigen 5.0.0 |
| Gmsh SDK | Geometry, meshing, mesh import, and optional preview; the build links `gmsh-4.15.dll` |
| Python with Tkinter | Optional desktop interfaces; Python 3.11+ includes the required TOML reader |
| Matplotlib and NumPy | GUI plots and numerical data handling |

The TOML C++ dependency is included as `tomlplusplus.hpp`. On Python versions older than 3.11, the interfaces also require `tomli`.

## Build

Run commands from the project root in PowerShell. Ensure the compiler and build tools are available on `PATH`. Replace dependency paths below with their actual locations:

```powershell
cmake -S . -B build -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DEIGEN_DIR="C:/path/to/eigen-5.0.0" `
  -DGMSH_INCLUDE_DIR="C:/path/to/gmsh-sdk/include" `
  -DGMSH_LIB_DIR="C:/path/to/gmsh-sdk/lib"

cmake --build build --config Release --parallel
```

For this generator, the executable is `build/phasefield_1.exe`. If an existing build directory uses a different generator or toolchain, choose a fresh build directory.

`CMakeLists.txt` contains machine-specific default dependency paths, so pass the cache variables explicitly when building on another computer. The Gmsh link target is also specifically named `gmsh-4.15.dll`; using another SDK layout or platform may require updating that target.

Release builds enable optimization and, when supported, link-time optimization. GCC Release builds use `-march=native`, so rebuild for the target machine when distributing binaries. `PFM_STATIC_RUNTIME` defaults to `ON` for MinGW runtime linking. The Windows build also attempts to copy runtime DLLs beside the executable. `PFM_USE_OPENMP` is optional and defaults to `OFF`; enabling it alone does not establish parallel assembly.

## Run from the command line

Start with mesh preprocessing to check geometry and physical-group assignments without running the nonlinear solve:

```powershell
.\build\phasefield_1.exe config.toml --mesh-only --no-preview --no-gui --outdir results --out mesh_check
```

Run the built-in specimen:

```powershell
.\build\phasefield_1.exe config.toml --no-preview --no-gui --outdir results --out sent_run
```

Run the supplied external-mesh example:

```powershell
.\build\phasefield_1.exe config_ambati_3pb.toml --no-preview --no-gui --outdir results --out three_point_bend
```

Relative mesh and output paths are interpreted from the process working directory. Running from the project root lets the example find `tpb_notched.msh`. Use a distinct output name for each experiment to avoid replacing existing results.

The supplied configurations are working experiment settings, not minimal smoke tests. Review them before starting a full solve: both use two-stage stepping and set `max_subdivs = 0`; `config.toml` additionally enables Gmsh windows and contains an absolute output path. The commands above override window and output settings, but preserve the numerical settings.

### Command-line overrides

The configuration file comes first; subsequent flags override selected values.

| Flag | Effect |
| --- | --- |
| `--mesh-file PATH` | Load an external mesh or geometry file |
| `--mesh-only` | Build/read the mesh and FEM input, then stop before solving |
| `--out NAME` | Set the run name and output filename prefix |
| `--outdir PATH` | Set the parent output directory |
| `--W VALUE`, `--H VALUE` | Set built-in specimen dimensions |
| `--a VALUE`, `--ycrack VALUE` | Set built-in crack length and height |
| `--hfine VALUE`, `--hfar VALUE` | Set built-in mesh sizes |
| `--steps N` | Set the step count used by uniform stepping |
| `--preview`, `--no-preview` | Enable/disable the blocking mesh preview before solving |
| `--gui`, `--no-gui` | Enable/disable the Gmsh window after the run |

Use `phasefield_1.exe --help` to display the executable's help. For unattended runs, disable both preview and post-run GUI windows.

## Python desktop interfaces

Install the plotting dependencies and launch an interface:

```powershell
python -m pip install matplotlib numpy
python run_gui3.py config.toml
```

For Python older than 3.11, also install `tomli`. Tkinter must be available in the Python installation.

Three interface variants are retained: `run_gui.py`, `run_gui2.py`, and `run_gui3.py`. They can be launched directly for comparison; the numbered files contain different interface iterations. For example:

```powershell
python run_gui2.py config_ambati_3pb.toml
```

The interfaces expose configuration fields, launch the C++ executable as a subprocess, and display the solver log and live force-displacement data. They search common build locations for the executable and allow it to be selected manually. Generated run settings are written to `_gui_run.toml` and copied into the results directory.

## Configuration

The supplied TOML files contain detailed inline explanations. Key sections are:

| Section | Controls |
| --- | --- |
| `[mesh]` | Built-in geometry or external file, mesh sizing, output base name |
| `[fem]` | Plane stress/strain, quadrature, energy split, hybrid mode |
| `[[materials]]` | Named material definitions |
| `[[material_for_group]]` | Material assignment to physical surface groups |
| `[[bcs]]` | Prescribed displacement components on physical groups |
| `[[neumann]]` | Boundary tractions |
| `[[point_load]]` | Concentrated loads |
| `[[initial_phi]]` | Initial phase-field values on physical groups |
| `[solver]` | Newton tolerances, coupling scheme, staggered stopping settings |
| `[run]` | Load schedule, subdivision, output cadence, previews, profiling |

Material properties follow this order:

```toml
[[materials]]
name = "example"
props = [210000.0, 0.3, 2.7, 0.01, 1e-9] # E, nu, Gc, l0, k
```

Here `E` is Young's modulus, `nu` is Poisson's ratio, `Gc` is fracture toughness, `l0` is the regularization length, and `k` is residual stiffness. The supplied configurations use lengths in mm, stresses in MPa, and `Gc` in N/mm. Keep input units consistent.

`ntype = 1` selects plane stress and `ntype = 2` selects plane strain. In displacement constraints, `flags = [1, 1]` constrains both components, while `[0, 1]` constrains only the vertical component; `values` supplies the prescribed displacements.

For `step_mode = "uniform"`, `N_steps` controls the increments. For `"two_stage"`, `du_coarse`, `u_switch`, and `du_fine` control the schedule and `N_steps` is ignored. The driver determines the loading direction from the largest nonzero prescribed displacement component.

### External meshes

Set `source = "file"` and `path` under `[mesh]`, or pass `--mesh-file`. Physical-group names must match the material and boundary-condition entries exactly. Use planar 2D meshes with supported element types and appropriate physical groups for surfaces, edges, and points. Represent an explicitly open crack with separate, unmerged nodes on its opposing faces.

The supplied bending configuration references `Domain`, `SupportLeft`, `SupportRight`, and `LoadPad`. Its active energy split is `"none"`, despite nearby comments referring to a spectral split; check active values when interpreting an experiment.

## Results

Normal output layout:

```text
<output_dir>/<base_name>/
  <base_name>.msh
  <base_name>.vtk
  <base_name>_force_disp.csv
  <base_name>_stagger_energy.csv   # staggered scheme
  <base_name>_run.txt              # when write_log = true
  vtk/                           # solution snapshots
```

Solution VTK files contain nodal damage `phi`, displacement vectors, and element stress fields, including von Mises stress. The phase field uses `0` for intact material and `1` for fully damaged material. The root VTK file is the mesh export; time-step fields are in `vtk/`.

The force-displacement CSV records `step`, `load_factor`, `applied_ux`/`applied_uy`, `reaction_Fx`/`reaction_Fy`, `max_phi`, and `converged`. The component depends on the detected loading direction. Mixed loading is summarized by one component, so this curve does not capture the entire response.

`vtk_every` controls snapshot frequency, `write_log` controls terminal-log mirroring, and `profile_every` controls per-step timing reports. A cumulative timing breakdown is printed at the end.

## Source map

| Files | Responsibility |
| --- | --- |
| `main.cpp` | Run orchestration, load stepping, reactions, output scheduling |
| `Config.h/.cpp` | TOML parsing, command-line overrides, validation |
| `MeshGeneration.h/.cpp` | Built-in specimen geometry and mesh generation |
| `GmshReader.h/.cpp`, `FEMINPUT.h` | Mesh import and finite element data structures |
| `ShapeFunc`, `Gauss`, `Jacob2`, `ElementUtils` | Element interpolation, quadrature, mappings, utilities |
| `ConstitutiveModel.h/.cpp` | Elastic response and energy splits |
| `StiffnessPFM`, `ResidualPFM`, `History` | Coupled assembly, phase-field contributions, history field |
| `BoundaryConditions`, `NeumannBC` | Displacement and traction boundary conditions |
| `Solver.h/.cpp` | Monolithic and staggered nonlinear solvers |
| `Output.h/.cpp`, `Profiling.h/.cpp` | VTK export and timing instrumentation |
| `run_gui*.py` | Python desktop interfaces |
| `FLEXIBILITY_REVIEW.md` | Design review and proposed extensions; some proposals have since been implemented |

`CMakeLists.txt` is the active build definition; the `.original.txt` and `.proposed.txt` variants are retained reference copies.

## Numerical and operational notes

- Inspect the `converged` column and solver log before interpreting a result. With `max_subdivs = 0`, the driver can accept unconverged increments; their history-field updates can affect later crack evolution. Filtering CSV rows cannot repair that history.
- Initial damage assignments set the starting state; they do not impose a permanently pinned crack.
- The hybrid formulation does not implement the additional compression constraint that resets damage where negative energy exceeds positive energy. Account for this limitation in compression/contact-dominated cases.
- Check mesh resolution relative to `l0`, load-step sensitivity, and solver convergence for each problem. This README does not establish benchmark accuracy.
- If Windows exits immediately with `0xC0000135`, check the Gmsh and compiler runtime DLLs beside the executable.
- If mesh preprocessing reports missing physical groups or no constrained nodes, check the group names and dimensions against the configuration.
- If an output folder cannot be created, the driver warns and falls back to the working directory. Check the log for the actual output location.

This documentation was checked against the source and supplied configuration files. No full simulation or fresh build was run as part of writing it.
