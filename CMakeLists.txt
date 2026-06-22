cmake_minimum_required(VERSION 3.18)

project(PhysicsBasedSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --extended-lambda")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

configure_file(
    ${CMAKE_SOURCE_DIR}/assets/ProjectPaths.h.in
    ${CMAKE_BINARY_DIR}/generated/ProjectPaths.h
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(extern/tetgen)
add_subdirectory(extern/glfw)

add_library(ElasticBody
    src/MeshToTet.cpp
    src/SkeletonAnimation.cpp
    src/Solver/Solver.cpp
    src/Solver/Solvergpu.cu
    src/Solver/Solvercpu.cpp
)

target_include_directories(ElasticBody
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern
    PRIVATE
        ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(ElasticBody
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        tet
)

set_target_properties(ElasticBody PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(render
    src/Render/RealtimeViewer.cpp
    src/Render/RealtimeViewerCuda.cu
)

target_include_directories(render
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/glm
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/glfw/deps
)

target_link_libraries(render
    PUBLIC
        ElasticBody
    PRIVATE
        CUDA::cudart
        glfw
        OpenGL::GL
)

set_target_properties(render PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(ElasticSimulator::ElasticBody ALIAS ElasticBody)
add_library(ElasticSimulator::render ALIAS render)

add_subdirectory(Example)
