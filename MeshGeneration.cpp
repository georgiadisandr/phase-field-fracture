#include "MeshGeneration.h"

#include <gmsh.h>
extern "C" {
#include <gmshc.h>   // C API: ABI-stable across MinGW<->MSVC, used in print_mesh_stats
}

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unordered_map>

namespace geo = gmsh::model::geo;
namespace mf  = gmsh::model::mesh::field;

namespace {
// Gmsh element-type ids
constexpr int kTri3  = 2;
constexpr int kQuad4 = 3;
constexpr int kQuad8 = 16;

// Number of 2D elements currently in the model.
//
// Uses the C API for the same reason print_mesh_stats does: the returned
// arrays are heap-allocated by the gmsh DLL, and handing std::vector across a
// MinGW <-> MSVC boundary is not ABI-safe.
std::size_t count_surface_elements()
{
    int ierr = 0;

    int* elemTypes = nullptr;             std::size_t elemTypes_n = 0;
    std::size_t** elemTags = nullptr;     std::size_t* elemTags_nn = nullptr;
    std::size_t elemTags_n = 0;
    std::size_t** elemNodeTags = nullptr; std::size_t* elemNodeTags_nn = nullptr;
    std::size_t elemNodeTags_n = 0;

    gmshModelMeshGetElements(&elemTypes, &elemTypes_n,
                             &elemTags, &elemTags_nn, &elemTags_n,
                             &elemNodeTags, &elemNodeTags_nn, &elemNodeTags_n,
                             2, -1, &ierr);

    std::size_t total = 0;
    if (ierr == 0)
        for (std::size_t i = 0; i < elemTags_n; ++i) total += elemTags_nn[i];

    gmshFree(elemTypes);
    for (std::size_t i = 0; i < elemTags_n;     ++i) gmshFree(elemTags[i]);
    gmshFree(elemTags); gmshFree(elemTags_nn);
    for (std::size_t i = 0; i < elemNodeTags_n; ++i) gmshFree(elemNodeTags[i]);
    gmshFree(elemNodeTags); gmshFree(elemNodeTags_nn);

    return total;
}
}  // namespace

namespace mesh {

Geometry build_geometry(const Config& cfg)
{
    Geometry g;

    // Per-point sizes are ignored once a background field is active
    // (MeshSizeFromPoints = 0 below); they are kept here only as
    // documentation/fallback for users who toggle the option back on.
    const int p1  = geo::addPoint(0.0,    0.0,         0.0, cfg.h_far);
    const int p2  = geo::addPoint(cfg.W,  0.0,         0.0, cfg.h_far);
    const int p3  = geo::addPoint(cfg.W,  cfg.H,       0.0, cfg.h_far);
    const int p4  = geo::addPoint(0.0,    cfg.H,       0.0, cfg.h_far);

    // p5u and p5l are coincident at the crack mouth but kept as separate
    // entities — this is what produces a sharp crack with duplicated nodes
    // along the upper/lower faces after meshing.
    const int p5u = geo::addPoint(0.0,    cfg.y_crack, 0.0, cfg.h_fine);
    const int p5l = geo::addPoint(0.0,    cfg.y_crack, 0.0, cfg.h_fine);
    const int p6  = geo::addPoint(cfg.a,  cfg.y_crack, 0.0, cfg.h_fine);
    const int p7  = geo::addPoint(cfg.W,  cfg.y_crack, 0.0, cfg.h_far);

    g.p_tip = p6;
    g.p_left1 =p5u;
    g.p_left2 =p5l;
    g.p_right =p7;

    
    // Create lines from points 
    g.lBottom  = geo::addLine(p1,  p2);
    g.lRightLo = geo::addLine(p2,  p7);
    g.lRightUp = geo::addLine(p7,  p3);
    g.lTop     = geo::addLine(p3,  p4);
    g.lLeftUp  = geo::addLine(p4,  p5u);
    g.lLeftLo  = geo::addLine(p5l, p1);
    g.lCrackUp = geo::addLine(p5u, p6);
    g.lCrackLo = geo::addLine(p6,  p5l);
    g.lLig     = geo::addLine(p6,  p7);

    const int upperLoop = geo::addCurveLoop(
        {g.lCrackUp, g.lLig, g.lRightUp, g.lTop, g.lLeftUp});
    g.upperSurf = geo::addPlaneSurface({upperLoop});

    const int lowerLoop = geo::addCurveLoop(
        {g.lBottom, g.lRightLo, -g.lLig, g.lCrackLo, g.lLeftLo});
    g.lowerSurf = geo::addPlaneSurface({lowerLoop});

    geo::synchronize();
    return g;
}

//refinement zones
void configure_size_field(const Config& cfg, const Geometry& g)
{
    // Distance to the crack faces.
    // const int fDistFaces = mf::add("Distance");
    // mf::setNumbers(fDistFaces, "CurvesList", {
    //     static_cast<double>(g.lCrackUp),
    //     static_cast<double>(g.lCrackLo),
    //     static_cast<double>(g.lLig)
    // });

     const int fDistFaces = mf::add("Distance");
        mf::setNumbers(fDistFaces, "CurvesList", {
        static_cast<double>(g.lRightLo),
        static_cast<double>(g.lRightUp),
        // static_cast<double>(g.lLig),
  
    });

    // Distance to the crack tip (gives a rosette-like refinement there).
    const int fDistTip = mf::add("Distance");
    mf::setNumbers(fDistTip, "PointsList", {
        static_cast<double>(g.p_tip),
        // static_cast<double>(g.p_left1),
        // static_cast<double>(g.p_left2),
        // static_cast<double>(g.p_right)
    });

    const int fThFaces = mf::add("Threshold");

    mf::setNumber(fThFaces, "InField", fDistFaces);

    mf::setNumber(fThFaces, "SizeMin", cfg.h_fine);
    mf::setNumber(fThFaces, "SizeMax", cfg.h_far);

    mf::setNumber(fThFaces, "DistMin", cfg.r_fine);
    mf::setNumber(fThFaces, "DistMax", cfg.r_far);

    const int fThTip = mf::add("Threshold");

    mf::setNumber(fThTip, "InField", fDistTip);
    mf::setNumber(fThTip, "SizeMin", cfg.h_fine);
    mf::setNumber(fThTip, "SizeMax", cfg.h_far);

    mf::setNumber(fThTip, "DistMin", cfg.r_fine*0.3);
    mf::setNumber(fThTip, "DistMax", cfg.r_far*0.3);


    // Take the smallest size requested by either field at every location.
    const int fMin = mf::add("Min");
    mf::setNumbers(fMin, "FieldsList", {
        static_cast<double>(fThFaces),
        static_cast<double>(fThTip)
    });
    //keep the smallest size 
    mf::setAsBackgroundMesh(fMin);

    gmsh::option::setNumber("Mesh.MeshSizeMin",                cfg.h_fine);
    gmsh::option::setNumber("Mesh.MeshSizeMax",                cfg.h_far);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints",         0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature",      0);
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
}

void configure_meshing(const Geometry& g)
{
    gmsh::model::mesh::setRecombine(2, g.upperSurf);
    gmsh::model::mesh::setRecombine(2, g.lowerSurf);

    gmsh::option::setNumber("Mesh.Algorithm",              8); // Frontal-Delaunay for quads
    gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 1); // Blossom
    gmsh::option::setNumber("Mesh.ElementOrder",           1);
    gmsh::option::setNumber("Mesh.SaveAll",                1);
}

void add_physical_groups(const Geometry& g)
{
    // physical Groups of geometry
    gmsh::model::addPhysicalGroup(1, {g.lBottom},               1,  "Bottom");
    gmsh::model::addPhysicalGroup(1, {g.lTop},                  2,  "Top");
    gmsh::model::addPhysicalGroup(1, {g.lRightLo, g.lRightUp},  3,  "Right");
    gmsh::model::addPhysicalGroup(1, {g.lLeftUp},               4,  "LeftUpper");
    gmsh::model::addPhysicalGroup(1, {g.lLeftLo},               5,  "LeftLower");
    gmsh::model::addPhysicalGroup(1, {g.lCrackUp},              6,  "CrackFaceUpper");
    gmsh::model::addPhysicalGroup(1, {g.lCrackLo},              7,  "CrackFaceLower");
    gmsh::model::addPhysicalGroup(1, {g.lLig},                  8,  "Ligament");
    gmsh::model::addPhysicalGroup(0, {g.p_tip },                    9,  "CrackTip");
    gmsh::model::addPhysicalGroup(2, {g.upperSurf, g.lowerSurf},10, "Domain");

}

void print_mesh_stats()
{
    int ierr = 0;

    std::size_t* nodeTags = nullptr;  std::size_t nodeTags_n = 0;
    double* coord = nullptr;          std::size_t coord_n = 0;
    double* paramCoord = nullptr;     std::size_t paramCoord_n = 0;
    gmshModelMeshGetNodes(&nodeTags, &nodeTags_n,
                          &coord, &coord_n,
                          &paramCoord, &paramCoord_n,
                          -1, -1, 0, 0, &ierr);

    int* elemTypes = nullptr;             std::size_t elemTypes_n = 0;
    std::size_t** elemTags = nullptr;     std::size_t* elemTags_nn = nullptr; std::size_t elemTags_n = 0;
    std::size_t** elemNodeTags = nullptr; std::size_t* elemNodeTags_nn = nullptr; std::size_t elemNodeTags_n = 0;
    gmshModelMeshGetElements(&elemTypes, &elemTypes_n,
                             &elemTags, &elemTags_nn, &elemTags_n,
                             &elemNodeTags, &elemNodeTags_nn, &elemNodeTags_n,
                             2, -1, &ierr);


    std::size_t nQuad = 0, nTri = 0;
    for (std::size_t i = 0; i < elemTypes_n; ++i) {
            if(elemTypes[i] == kQuad4){  nQuad += elemTags_nn[i];}
            else if (elemTypes[i] == kTri3 ){nTri  += elemTags_nn[i];}            
    }

    std::cout << "Mesh stats: "
              << nodeTags_n << " nodes, "
              << nQuad << " quads, "
              << nTri  << " triangles\n";

    gmshFree(nodeTags); gmshFree(coord); gmshFree(paramCoord);
    gmshFree(elemTypes);
    for (std::size_t i = 0; i < elemTags_n;     ++i) gmshFree(elemTags[i]);
    gmshFree(elemTags); gmshFree(elemTags_nn);
    for (std::size_t i = 0; i < elemNodeTags_n; ++i) gmshFree(elemNodeTags[i]);
    gmshFree(elemNodeTags); gmshFree(elemNodeTags_nn);

}

void load_from_file(const Config& cfg)
{
    if (cfg.path.empty())
        throw std::runtime_error(
            "mesh.source = \"file\" requires mesh.path (or --mesh-file PATH)");

    // gmsh::open creates a new model from the file and throws on failure, so
    // a missing or malformed file surfaces here with gmsh's own diagnostic.
    // Config::validate() has already checked that the path exists, which
    // turns the common typo into a clearer error before gmsh is involved.
    gmsh::open(cfg.path);

    // A .msh arrives already meshed and must be used verbatim -- calling
    // generate(2) on it would discard the mesh and remesh from the underlying
    // geometry, silently replacing a mesh the author had tuned. A .geo or CAD
    // file carries geometry only and does need meshing.
    const std::size_t before = count_surface_elements();
    if (before == 0) {
        std::cout << "[mesh] \"" << cfg.path
                  << "\" contains no 2D elements -- meshing it now\n";
        gmsh::model::mesh::generate(2);
    } else {
        std::cout << "[mesh] using the mesh stored in \"" << cfg.path
                  << "\" as-is (" << before << " surface elements)\n";
    }

    if (count_surface_elements() == 0)
        throw std::runtime_error(
            "mesh file \"" + cfg.path + "\" produced no 2D elements. A 2D "
            "surface mesh is required; check that the file defines a surface "
            "and that meshing succeeded.");

    print_mesh_stats();
}

Geometry generate(const Config& cfg)
{
    const Geometry g = build_geometry(cfg);
    configure_size_field(cfg, g);
    configure_meshing(g);
    add_physical_groups(g);

    gmsh::model::mesh::generate(2);
    print_mesh_stats();
    return g;
}

}  // namespace mesh
