cmake_minimum_required(VERSION 3.18)

project(ElasticSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
enable_language(CUDA)
    
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

configure_file(
    ${CMAKE_SOURCE_DIR}/assets/ProjectPaths.h.in
    ${CMAKE_BINARY_DIR}/generated/ProjectPaths.h
)

# set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# external dependencies
add_subdirectory(extern/tetgen)
# GLFW
add_subdirectory(extern/glfw)
# GLM
add_subdirectory(extern/glm)

# set source files
file(GLOB_RECURSE SRC_FILES 
    src/*.cpp 
    src/*.cu 
    include/*.h 
    include/*.hpp
)

add_executable(ElasticSimulator
    ${SRC_FILES}
    main.cpp
)

# include directories
target_include_directories(ElasticSimulator
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/glm
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/glfw/deps
        ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(ElasticSimulator
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        tet
        glfw
        OpenGL::GL
)

set_target_properties(ElasticSimulator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)