if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
    set(OS_LIBRARIES "")
endif(WIN32)

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(OS_LIBRARIES pthread)
endif(UNIX)

enable_testing()
include_directories(${GTEST_INCLUDE_DIR})
include_directories(${CMAKE_MODULE_PATH}/src)

link_directories(${GTEST_LIB_DIR})

set(TEST_SOURCES main.cpp test_datatype.h common.h test_tensorshape.h test_tensor.h test_tensordata.h)
add_executable(tensor_test ${TEST_SOURCES})
target_link_libraries(tensor_test LINK_PUBLIC tensor ${GTEST_LIBRARIES} ${OS_LIBRARIES})
add_test(AllTests tensor_test)

set_target_properties(tensor_test
                      PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
