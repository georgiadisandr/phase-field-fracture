# Flexibility review — what is already generic, what is welded to the SENT test

Scope: proposals only, no code changed. Effort tags:
**S** ≈ an afternoon · **M** ≈ 1–3 days · **L** ≈ a week+ / touches many files.

---

## 0. What is already in good shape

Worth stating, because it constrains what the rest should look like:

- The **TOML + CLI two-layer config** (`Config.h/.cpp`) with `validate()` is a solid
  backbone. Almost everything below is "add a section to this", not "rebuild it".
- **Materials, BCs, Neumann, point loads, initial φ are all keyed by gmsh
  physical-group name** — that is the right abstraction and it already means a
  different mesh with the same group names needs zero code changes.
- **Element handling is genuinely mixed-type** (CSR connectivity, `RefElementTable`
  by `nnode`, tri3/quad4/quad8). Adding element types is cheap.
- `energy_split` × `hybrid` are cleanly orthogonal and runtime-selected.
- Output layout (one folder per run, `vtk_every`, profiling, log tee) is already
  parameter-study friendly.

The rigidity is concentrated in three places: **geometry, the fracture model, and
the definition of "the load"**.

---

## 1. Geometry — the biggest single limitation  ★ highest payoff

`MeshGeneration.cpp` *is* the SENT specimen. It builds 7 points and 9 lines by
hand, and `Geometry` is a struct of named integer fields (`lCrackUp`, `p_tip`, …).
`add_physical_groups` hardcodes the ten group names. Anything that is not
"rectangle with a horizontal edge crack" requires editing C++ and recompiling.

Worse, `configure_size_field` currently has the crack-face block **commented out**
and the right-edge block live — i.e. the refinement target is being changed by
hand-editing source between runs. That is the clearest symptom of the problem.

### 1a. Accept an external mesh file — **S/M, do this first**
`[mesh] source = "file"` / `path = "specimen.msh"` (or a `.geo`). Skip
`mesh::generate` entirely and let `readFemInputFromGmsh` work on whatever is
loaded. Because everything downstream is already keyed by physical-group name,
this **immediately unlocks every geometry Gmsh can produce** — three-point bend,
L-panel, plate with hole, notched shear, your own `.geo` scripts — for maybe 60
lines of driver change. It also lets you use meshes from collaborators/papers.

### 1b. Named-entity `Geometry` + declarative physical groups — **M**
Replace the fixed-field `Geometry` struct with
`std::unordered_map<std::string, std::vector<int>>` (name → entity tags, per dim).
Then physical groups come from TOML:

```toml
[[physical_group]]
name = "Right"; dim = 1; entities = ["right_lower", "right_upper"]
```

Builders publish named entities; the config decides what becomes a group. Removes
the hardcoded group list and makes group names problem-specific.

### 1c. Declarative refinement — **S/M, kills the commented-out code**
Refinement fields should reference *names*, not `g.lRightLo`:

```toml
[[refine]]
on       = ["CrackFaceUpper", "CrackFaceLower"]   # curves or points, by name
h_min    = 2e-3
h_max    = 6e-2
dist_min = 0.5
dist_max = 0.7
```

Any number of Distance+Threshold pairs, combined with `Min`, built in a loop. You
stop recompiling to change where the mesh is fine. Also worth exposing
`Mesh.Algorithm`, `RecombinationAlgorithm`, `ElementOrder`, and the
recombine on/off flag (currently hardcoded to quad-dominant order 1) as
`[mesh.options]`.

### 1d. Geometry registry — **M/L, only if you want built-in specimens**
`mesh.type = "sent" | "shear" | "three_point_bend" | "plate_with_hole"` dispatching
to registered builder functions, each with its own `[mesh.params]` table.
Nice-to-have; **1a covers 90% of the need for 20% of the work**, so I'd only do
this for the 2–3 specimens you rerun constantly.

---

## 2. Fracture model is hardwired to AT2 + quadratic degradation — **M/L**

`Gc/l0*φ − 2(1−φ)H` and `Gc*l0 ∇φ` are written out inline in **four separate
places** in `StiffnessPFM.cpp` (residual, tangent, φ-only subsolve, energy), and
`g(φ)=(1−φ)²+k` is a free function. So:

- No **AT1** (linear dissipation, elastic limit, no damage before a threshold) —
  this is the most common reviewer request after AT2.
- No **PF-CZM / Wu** cohesive model, no alternative degradation (Borden cubic,
  rational g(φ)).
- Changing the model means editing four consistent copies of the same algebra.

**Proposal:** a small `FractureModel` interface evaluated per Gauss point —
`alpha(φ)`, `alpha'(φ)`, `alpha''(φ)`, `g(φ)`, `g'(φ)`, `g''(φ)`, `c_w`, and an
optional psi-threshold. Every residual/tangent term becomes a call into it, so all
four sites derive from one definition. Selected by `fem.at_model = "at2"|"at1"`.
Localized to `ConstitutiveModel.*` + `StiffnessPFM.cpp`; nothing else moves.

**Related, S:** `MatParams::from(props)` is positional over exactly 5 slots
`{E,nu,Gc,l0,k}`. Any model with more parameters (σ_c for AT1/PF-CZM, anisotropic
Gc) breaks it. Move to named keys in TOML:
`props = { E = 210000, nu = 0.3, Gc = 2.7, l0 = 4e-3, k = 1e-9 }` and a
`std::map<std::string,double>` lookup validated per model.

Also missing and worth flagging: **no irreversibility on φ itself** (only the
history field), and Ambati's `φ = 0 where ψ⁺ < ψ⁻` constraint is documented as
not implemented. Both are model options that belong behind the same interface.

---

## 3. "The load" is inferred by a heuristic — **S/M, high value**

`main.cpp` decides what is being driven by scanning all Dirichlet entries and
taking **the largest |prescribed value|**. Consequences:

- Mixed-mode loading only logs one component (it warns, but the curve is wrong).
- Two edges pulled in opposite directions, or a symmetric setup, picks arbitrarily.
- Force/traction-controlled runs have no "applied displacement" at all, so
  `two_stage` mode throws.
- Exactly one reaction probe is possible, on the loaded group only.

**Proposal — explicit `[load]` section (S):**

```toml
[load]
control      = "displacement"   # or "force"
driver_group = "Top"
driver_dof   = "y"
u_max        = 0.006
```

The stepper keys off this, not off `max|value|`. Removes the heuristic and the
warning entirely.

**Proposal — per-BC ramp flag (S):** each `[[bcs]]`/`[[neumann]]` gets
`ramp = true|false` (or `factor_curve = "main"`). Today *everything* is multiplied
by the same `load_factor`, so you cannot hold a confining pressure or a pre-stress
constant while ramping the crack driver. This is a genuine physical limitation,
not just ergonomics.

**Proposal — arbitrary reaction probes (S):**

```toml
[[output.reaction]]
group = "Top"; dof = "y"; label = "Fy_top"
```

N probes → N columns in the force-displacement CSV. Currently hardwired to one.

**Proposal — amplitude curves (M):** per-BC `[(t, factor), …]` table or named
profile, enabling **cyclic / unload–reload** loading. Not possible at all today
(`load_factor` is monotone 0→1 by construction).

---

## 4. Step schedule: generalize two hardcoded modes into N stages — **S**

`StepMode::{Uniform, TwoStage}` with fields `du_coarse / u_switch / du_fine` is a
two-stage schedule frozen into the type system. Replace with a list:

```toml
[[stage]]
du = 1.0e-4; until_u = 0.015
[[stage]]
du = 1.0e-5; until_u = 0.030
```

One stage reproduces `uniform`, two reproduce `two_stage`, three+ are new. The
adaptive-halving logic is unchanged and stays orthogonal. Small change, and it
also fixes the fact that the current schedule is keyed to the same `max|value|`
heuristic as §3.

**Related, S:** `max_subdivs = 0` (halving off) is currently set in your live
`config.toml`, and the code correctly warns that this permanently contaminates the
history field. Consider making 0 require an explicit `allow_unconverged = true`
so it cannot be reached by accident.

---

## 5. Dimensionality — split into "cheap hygiene" and "actual 3D"

### 5a. Hygiene — **S, worth doing regardless of 3D plans**
`ndofn` is configurable, but the code writes `2 * node + k` and `2 * d.npoin`
literally in `BoundaryConditions.cpp`, `Output.cpp`, `History.cpp` and `main.cpp`.
So setting `ndofn = 3` compiles, passes validation, and produces silent garbage.
Add one inline `gdofU(d, node, k)` helper plus `nDofU(d)` and replace every literal
`2*`. Also `d.ndime = 2` is assigned unconditionally in `GmshReader.cpp:102`
despite being a struct field. This costs nothing and removes a whole class of
silent failure.

### 5b. Real 3D — **L, only if you actually need it**
`Matrix3d`/`Vector3d` Voigt is fixed at 3 components throughout
`ConstitutiveModel`, `ShapeFunc.cpp` is 2D-only (and `ElementUtils` already throws
a clear error saying so), the VTK writer handles tri3/quad4/quad8 only, and the
spectral split's finite-differenced tangent is 2D. Doable — `Matrix<double,6,6>`,
tet4/hex8, VTK types 10/12 — but it is a real project, not a refactor. Decide
this deliberately rather than drifting toward it.

---

## 6. `main.cpp` is a ~700-line driver doing eight jobs — **M**

Config, output-dir creation, log tee, mesh, FEM input, load-direction detection,
reaction CSV, the adaptive stepping loop, VTK striding, timing, GUI. Consequences:
you cannot run two configs in one process, cannot unit-test the stepper, and every
new feature lands in the same function.

**Proposal:** extract three pieces —
`LoadStepper` (schedule + adaptive halving + rollback),
`RunOutputs` (paths, CSV writers, VTK stride, logger),
`runSimulation(const AppConfig&)` returning a result struct —
leaving `main()` at ~40 lines. This is what makes a **parameter sweep**
(loop over l0, over h_fine, over split model) a 15-line addition instead of a
shell script re-launching the exe.

**Related, S:** CLI overrides are hand-written flag-by-flag in `applyCliOverrides`
and only cover 10 of ~40 settings. A generic dotted-path form —
`--set fem.hybrid=false --set materials.0.props.l0=2e-3` — makes sweeps trivial
from one TOML. Straightforward with toml++ (`table.at_path()`).

---

## 7. Verification harness — **M, and it is what makes 1–6 safe**

There is no test suite. Every refactor above silently risks changing results, and
you would find out from a wrong crack path three days later.

**Proposal:** `tests/` with 3 fast configs and a compare script:
1. **Uniform bar, no damage** — closed-form σ = Eε, checks assembly/BCs exactly.
2. **Coarse SENT tension** (~2 min) — force-disp CSV compared to a stored reference
   within tolerance; catches any change to the constitutive/stepping path.
3. **Coarse shear** — catches the load-direction logic specifically.

Plus a golden-file check on `_stagger_energy.csv`. Once this exists, all the
proposals above become low-risk; without it they are all high-risk.

---

## 8. Housekeeping — **S**

- Three CMakeLists (`.txt`, `.original.txt`, `.proposed.txt`) in the repo — pick one.
- `README.md` (June) predates the config system; `SESSION_HANDOFF.md` and
  `PERFORMANCE_REVIEW.md` are working notes sitting next to source. Move to `docs/`.
- Linear solver is fixed at compile time (`SparseLU` / `SimplicialLDLT`). One
  `solver.linear = "lu"|"ldlt"|"cg"` key would help on large meshes — **S**.
- No flat directory structure: 20 `.cpp` at top level plus a 486 KB vendored
  `tomlplusplus.hpp`. `src/` + `external/` — cosmetic but cheap.

---

## Suggested order

| # | Item | Effort | Why now |
|---|------|--------|---------|
| 1 | §7 regression tests | M | Prerequisite for safely doing anything else |
| 2 | §1a external mesh file | S/M | Unlocks all geometries for the least work |
| 3 | §5a `ndofn` hygiene | S | Removes silent-garbage failure mode |
| 4 | §3 explicit `[load]` + reaction probes | S | Kills the max-value heuristic |
| 5 | §1c declarative refinement | S/M | Stops source-editing between runs |
| 6 | §4 N-stage schedule | S | Natural follow-on from §3 |
| 7 | §6 extract `LoadStepper` / `RunOutputs` | M | Enables parameter sweeps |
| 8 | §2 `FractureModel` interface (AT1) | M/L | Biggest physics gain, do once §7 exists |
| 9 | §1b/1d named entities, geometry registry | M | Only if built-in specimens matter |
| 10 | §5b 3D | L | Deliberate decision, separate project |

Items 2–6 are together maybe a week and cover most of what "change the case
without recompiling" means in practice.
