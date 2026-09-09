cmake_minimum_required(VERSION 3.16)
project(PhaseFieldFEM1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Release / Debug / RelWithDebInfo / MinSizeRel)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(EIGEN_DIR "C:/Users/andre/OneDrive/Documents/Phase Modeling/FEM/eigen-5.0.0")
set(GMSH_INCLUDE_DIR "C:/Program Files/Gmsh/gmsh-4.15.2-Windows64-sdk/include")
set(GMSH_LIB_DIR "C:/Program Files/Gmsh/gmsh-4.15.2-Windows64-sdk/lib")

add_executable(phasefield_1
    main.cpp
    MeshGeneration.cpp
    Gauss.cpp
    Jacob2.cpp
    ShapeFunc.cpp
    GmshReader.cpp
    ConstitutiveModel.cpp
    ElementUtils.cpp
    StiffnessPFM.cpp
    ResidualPFM.cpp
    History.cpp
    BoundaryConditions.cpp
    NeumannBC.cpp
    Solver.cpp
    Output.cpp
    Config.cpp
)

# The project root is on the include path so the vendored single-header
# tomlplusplus.hpp (sitting next to the .cpp files) is found by Config.cpp.
target_include_directories(phasefield_1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${GMSH_INCLUDE_DIR})

target_link_libraries(phasefield_1      PRIVATE "${GMSH_LIB_DIR}/gmsh-4.15.dll")

# On Windows, copy the gmsh DLL next to the executable after build so the .exe
# can find it without manual PATH juggling.
if(WIN32)
    find_file(GMSH_DLL NAMES gmsh-4.15.dll gmsh.dll
        HINTS "${GMSH_LIB_DIR}"
        PATH_SUFFIXES bin lib)
    if(GMSH_DLL)
        add_custom_command(TARGET phasefield_1 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GMSH_DLL}" "$<TARGET_FILE_DIR:phasefield_1>")
    else()
        message(WARNING "gmsh DLL not found; phasefield_1.exe will fail to start at runtime")
    endif()
endif()
