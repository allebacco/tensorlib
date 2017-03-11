#ifndef COMMON_H
#define COMMON_H

#include <type_traits>

#include <gtest/gtest.h>


#define EXPECT_SAME_TYPE(type,expected_type)     \
    EXPECT_EQ(typeid(type),typeid(expected_type))


#endif // COMMON_H
