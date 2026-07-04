cmake_minimum_required(VERSION 3.16)
project(argus_mpc CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Paths ──────────────────────────────────────────────────────────────────
set(ACADOS_ROOT "$ENV{HOME}/acados" CACHE PATH "ACADOS installation root")
set(GENERATED_CODE_DIR "${CMAKE_SOURCE_DIR}/c_generated_code")

# ── Generated C solver (compiled from source — no .so dependency) ──────────
# Re-run CMake configure after regenerating c_generated_code/ to pick up
# any new source files produced by scripts/generate_mpc.py.
file(GLOB SOLVER_SRCS
    "${GENERATED_CODE_DIR}/acados_solver_quadrotor.c"
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

# ── Main executable ────────────────────────────────────────────────────────
add_executable(argus_mpc src/mpc_controller.cpp)

target_include_directories(argus_mpc PRIVATE
    ${GENERATED_CODE_DIR}
    ${ACADOS_ROOT}/include
    ${ACADOS_ROOT}/include/acados
    ${ACADOS_ROOT}/include/blasfeo/include
    ${ACADOS_ROOT}/include/hpipm/include
)

target_link_directories(argus_mpc PRIVATE ${ACADOS_ROOT}/lib)

target_link_libraries(argus_mpc PRIVATE
    quadrotor_solver
    acados
    hpipm
    blasfeo
    m
)

# Embed ACADOS lib path using DT_RPATH (not DT_RUNPATH) so the path propagates
# to transitive dependencies (libacados.so → libhpipm.so, libblasfeo.so, etc.)
set_target_properties(argus_mpc PROPERTIES
    BUILD_RPATH   "${ACADOS_ROOT}/lib"
    INSTALL_RPATH "${ACADOS_ROOT}/lib"
)
target_link_options(argus_mpc PRIVATE "-Wl,--disable-new-dtags")
