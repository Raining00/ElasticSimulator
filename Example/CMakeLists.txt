include(${CMAKE_CURRENT_LIST_DIR}/add_projecy.cmake)

file(GLOB example_children RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/*)

foreach(example_child IN LISTS example_children)
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${example_child}
       AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/${example_child}/CMakeLists.txt)
        add_subdirectory(${example_child})
    endif()
endforeach()
