#ifndef TEST_TENSOR_H
#define TEST_TENSOR_H

#include "common.h"

#include "tensor.h"

using tensor::Tensor;
using tensor::TensorShape;
using tensor::DataType;


// Test the Tensor default constructor
TEST(Tensor, DefaultConstructor)
{
    Tensor t;
    EXPECT_EQ(t.shape(), TensorShape());
    EXPECT_EQ(t.data<int8_t>(), nullptr);
    EXPECT_EQ(t.dtype(), DataType::Void);
    EXPECT_EQ(t.size(), 0);
    EXPECT_TRUE(t.is_own_data());
}



#endif // TEST_TENSOR_H
