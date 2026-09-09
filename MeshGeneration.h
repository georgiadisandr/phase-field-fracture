#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace mesh {

// Where the mesh comes from.
//
//   Builtin -- build_geometry() below constructs the parametric SENT specimen
//              from W, H, a, y_crack and meshes it. This is the original
//              behaviour and remains the default.
//
//   File    -- an external gmsh file is opened instead and the geometry knobs
//              are ignored. Any format gmsh can read works (.msh, .geo, .step,
//              .brep, ...). If the file contains a 2D mesh it is used as-is;
//              if it only carries geometry (a .geo or CAD file) it is meshed
//              on load using whatever options the file itself sets.
//
// The solver does not care which of these produced the model: GmshReader
// resolves everything by physical-group NAME from the live gmsh session. An
// external mesh therefore only has to satisfy that contract:
//
//   * 2D, in the z = 0 plane
//   * a 2D physical group for the material region, named to match
//     [[material_for_group]] (e.g. "Domain")
//   * 1D named physical groups for every edge referenced by [[bcs]] and
//     [[neumann]], and 0D groups for [[point_load]] / [[initial_phi]] points
//   * element order 1; tri3, quad4 and quad8 are supported
//   * crack faces built from DUPLICATED, unmerged nodes (gmsh's Plugin(Crack)
//     is the reliable way to produce these)
//
// Run with --mesh-only to check a file against that contract without solving.
enum class Source {
    Builtin,
    File
};

//Specimen configuration
struct Config {
    // Mesh origin. See Source above. TOML key mesh.source, CLI --mesh-file.
    Source source = Source::Builtin;

    // Path to the external gmsh file. Used only when source == Source::File,
    // in which case it is required. Relative paths are resolved against the
    // working directory.
    std::string path;

    // All quantities are in millimetres.
    // NOTE: every geometry and mesh-size field below is IGNORED when
    // source == Source::File -- the file defines its own geometry and sizing.
    // Geometry
    double W       = 100.0;  // specimen width
    double H       = 40.0;   // specimen height
    double a       = 50.0;   // crack length
    double y_crack = 20.0;   // crack y-coordinate 

    // Mesh sizes
    double h_fine = 0.10;    // size near the crack
    double h_far  = 0.4;     // size in the bulk

    // Refinement zone radii
    double r_fine = 1.0;     // fully-fine zone around the crack faces
    double r_far  = 10.0;    // transition radius back to h_far

    // Output
    std::string base_name = "cracked_specimen_refined_line";
};

// Tags of the gmsh entities created by build_geometry.
struct Geometry {
    //Surfaces
    int upperSurf = 0; 
    int lowerSurf = 0;

    //tip of the crack   
    int p_tip    = 0; 
    int p_left1 = 0;
    int p_left2  =0;
    int p_right = 0;               
    
    //edges
    int lBottom  = 0, lTop      = 0;
    int lRightLo = 0, lRightUp  = 0;
    int lLeftUp  = 0, lLeftLo   = 0;
    int lCrackUp = 0, lCrackLo  = 0;
    int lLig     = 0; // ligament (crack tip -> right edge)                
};

// Build all gmsh entities for the cracked specimen and synchronize the CAD.
[[nodiscard]] Geometry build_geometry(const Config& cfg);

// Configure the background size field (Distance + Threshold on crack faces & tip).
void configure_size_field(const Config& cfg, const Geometry& g);

// Set recombination + meshing options for quad-dominant linear elements.
void configure_meshing(const Geometry& g);

// Add physical groups for boundaries, crack faces, ligament, tip, and the domain.
void add_physical_groups(const Geometry& g);

// Print mesh statistics (nodes, quads, tris) to stdout.
void print_mesh_stats();

// Open cfg.path into the current gmsh session (source == Source::File).
//
// Meshes the model only if the file did not already contain 2D elements, so a
// .msh is used exactly as saved and a .geo / CAD file is meshed on the spot.
// No meshing options are forced: a geometry file is expected to set its own
// (Recombine Surface, Mesh.Algorithm, element order), because overriding them
// here would silently change a mesh the author had already tuned.
//
// Throws std::runtime_error if the path is empty, the file cannot be read, or
// the result contains no 2D elements.
void load_from_file(const Config& cfg);

// Caller is responsible for gmsh::initialize / gmsh::model::add / gmsh::write
// / gmsh::finalize around this call.
Geometry generate(const Config& cfg);

}  // namespace mesh
