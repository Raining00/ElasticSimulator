add_executable(KComputeKCpu
    main.cpp
)

target_include_directories(KComputeKCpu
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(KComputeKCpu PROPERTIES
    FOLDER Example
)
