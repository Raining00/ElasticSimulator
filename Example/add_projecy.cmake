include_guard(GLOBAL)

function(add_elastic_example PROJECT_NAME)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ELASTIC_EXAMPLE "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT ELASTIC_EXAMPLE_SOURCES)
        message(FATAL_ERROR "add_elastic_example(${PROJECT_NAME}) requires SOURCES.")
    endif()

    add_executable(${PROJECT_NAME} ${ELASTIC_EXAMPLE_SOURCES})

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ElasticSimulator::ElasticBody
            ElasticSimulator::render
    )

    target_include_directories(${PROJECT_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/generated
    )

    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        FOLDER Example
    )
endfunction()
