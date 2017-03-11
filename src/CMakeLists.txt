set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_BUILD_TYPE Release)


set(HEADER_SRC tensor.h datatype.h datatype_conversion.h tensorshape.h)
set(HEADER_PRIVATE_SRC internal/datatype_conversion_p.h)
set(SOURCES_SRC tensor.cpp tensorshape.cpp)


set(SOURCES ${HEADER_SRC} ${SOURCES_SRC} ${HEADER_PRIVATE_SRC})

add_library(tensor SHARED ${SOURCES})
#add_library(tensor_s STATIC ${SOURCES})
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_include_directories(tensor_s PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(tensor
                      PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
