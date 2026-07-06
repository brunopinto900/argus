cmake_minimum_required(VERSION 3.16)
project(argus_mpc CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Paths ──────────────────────────────────────────────────────────────────
# CMAKE_CURRENT_SOURCE_DIR (not CMAKE_SOURCE_DIR) so these stay correct when
# this file is pulled in via add_subdirectory() from another project (e.g.
# Vista-Tracker's -DWITH_MPC=ON build), where CMAKE_SOURCE_DIR would instead
# point at the outer project's root.
set(ACADOS_ROOT "$ENV{HOME}/acados" CACHE PATH "ACADOS installation root")
set(GENERATED_CODE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")

# Fail early with a clear message if generate_mpc.py hasn't been run yet.
if(NOT EXISTS "${GENERATED_CODE_DIR}/argus_params.h")
    message(FATAL_ERROR
        "c_generated_code/argus_params.h not found.\n"
        "Run: (cd ${CMAKE_CURRENT_SOURCE_DIR} && python3 scripts/generate_mpc.py)")
endif()

# ── Generated C solver (compiled from source — no .so dependency) ──────────
# Re-run CMake configure after regenerating c_generated_code/ to pick up
# any new source files produced by scripts/generate_mpc.py.
file(GLOB SOLVER_SRCS
    "${GENERATED_CODE_DIR}/acados_solver_quadrotor.c"
    "${GENERATED_CODE_DIR}/acados_sim_solver_quadrotor.c"
    "${GENERATED_CODE_DIR}/quadrotor_model/*.c"
    "${GENERATED_CODE_DIR}/quadrotor_cost/*.c"
)

add_library(quadrotor_solver STATIC ${SOLVER_SRCS})

target_include_directories(quadrotor_solver PUBLIC
    ${GENERATED_CODE_DIR}
    ${ACADOS_ROOT}/include
    ${ACADOS_ROOT}/include/acados
    ${ACADOS_ROOT}/include/blasfeo/include
    ${ACADOS_ROOT}/include/hpipm/include
)

# Suppress warnings in auto-generated code; keep optimisation.
target_compile_options(quadrotor_solver PRIVATE -O2 -w)

# ── QuadrotorMpc library — the public, versioned API for this OCP ──────────
# Wraps the raw acados calls behind include/argus_mpc/QuadrotorMpc.hpp.
# Consumers (this repo's own demo below, or an external project like
# Vista-Tracker) only need to link this target — no acados include paths or
# link libraries to redeclare themselves; PUBLIC below propagates all of it.
add_library(quadrotor_mpc STATIC
    src/argus_mpc/QuadrotorMpc.cpp
)

target_include_directories(quadrotor_mpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_directories(quadrotor_mpc PUBLIC ${ACADOS_ROOT}/lib)

target_link_libraries(quadrotor_mpc PUBLIC
    quadrotor_solver
    acados
    hpipm
    blasfeo
    m
)

# PUBLIC + INTERFACE link options so any consumer executable (this repo's
# demo, or an external project linking quadrotor_mpc) automatically gets the
# right rpath baked into its own final link step — no per-consumer rpath
# boilerplate needed. DT_RPATH (not DT_RUNPATH) so it propagates to
# transitive dependencies (libacados.so → libhpipm.so, libblasfeo.so, ...).
target_link_options(quadrotor_mpc INTERFACE
    "-Wl,-rpath,${ACADOS_ROOT}/lib"
    "-Wl,--disable-new-dtags"
)

# ── Main executable (moving-target tracking demo) ────────────────────────────
add_executable(argus_mpc
    src/mpc_controller.cpp
    src/plant_dynamics.cpp
)

find_package(yaml-cpp REQUIRED)

target_compile_definitions(argus_mpc PRIVATE
    ARGUS_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
)

target_link_libraries(argus_mpc PRIVATE quadrotor_mpc yaml-cpp)

# ── Tests ─────────────────────────────────────────────────────────────────────
# Only when argus is configured standalone (top-level project) — when pulled
# in via add_subdirectory() from a consumer like Vista-Tracker, its own
# FetchContent(googletest) and gtest targets already cover this, and argus's
# tests are only useful for developing argus itself anyway.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(quadrotor_mpc_tests
        tests/test_quadrotor_mpc.cpp
    )
    target_link_libraries(quadrotor_mpc_tests PRIVATE
        quadrotor_mpc
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(quadrotor_mpc_tests)
endif()
