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
    EXPECT_EQ(t.element_size(), 1);
}


// Test the Tensor constructor from size and type
TEST(Tensor, SizeAndTypeConstructor)
{
    Tensor t(100, DataType::Int32);
    EXPECT_EQ(t.shape(), TensorShape({100}));
    EXPECT_NE(t.data<int32_t>(), nullptr);
    EXPECT_EQ(t.dtype(), DataType::Int32);
    EXPECT_EQ(t.size(), 100);
    EXPECT_TRUE(t.is_own_data());
    EXPECT_EQ(t.element_size(), sizeof(int32_t));
}

// Test the Tensor constructor from size, element size and type
TEST(Tensor, SizeElementSizeAndTypeConstructor)
{
    Tensor t1(100, 10);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<uint8_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::User);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), 10);

    Tensor t2(100, 4, DataType::Float32);
    EXPECT_EQ(t2.shape(), TensorShape({100}));
    EXPECT_NE(t2.data<uint8_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Float32);
    EXPECT_EQ(t2.size(), 100);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), 4);
}


// Test the Tensor constructor from initializer list and type
TEST(Tensor, InitializerListAndTypeConstructor)
{
    Tensor t1({100}, DataType::Int32);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<int32_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::Int32);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), sizeof(int32_t));

    Tensor t2({100, 20, 30}, DataType::Int32);
    EXPECT_EQ(t2.shape(), TensorShape({100, 20, 30}));
    EXPECT_NE(t2.data<int32_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Int32);
    EXPECT_EQ(t2.size(), 100*20*30);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), sizeof(int32_t));
}


// Test the Tensor constructor from initializer list, element size and type
TEST(Tensor, InitializerListElementSizeAndTypeConstructor)
{
    Tensor t1({100}, 10);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<uint8_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::User);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), 10);

    Tensor t2({100, 20, 30}, 4, DataType::Int32);
    EXPECT_EQ(t2.shape(), TensorShape({100, 20, 30}));
    EXPECT_NE(t2.data<int32_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Int32);
    EXPECT_EQ(t2.size(), 100*20*30);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), 4);
}

#endif // TEST_TENSOR_H
