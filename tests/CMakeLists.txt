set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_MODULE_PATH}/src)

set(TEST_SOURCES main.cpp test_datatype.h common.h test_tensorshape.h)
add_executable(tensor_test ${TEST_SOURCES})
target_link_libraries(tensor_test LINK_PUBLIC tensor ${GTEST_BOTH_LIBRARIES} pthread)
add_test(AllTests tensor_test)

set_target_properties(tensor_test PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
