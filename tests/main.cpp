
#include <gtest/gtest.h>

#include "test_datatype.h"
#include "test_tensorshape.h"
#include "test_tensor.h"



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
