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



// Test the Tensor constructor from shape pointer and type
TEST(Tensor, ShapePtrAndTypeConstructor)
{
    std::vector<int> shape0 {100};
    Tensor t1(shape0.data(), shape0.size(), DataType::Int32);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<int32_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::Int32);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), sizeof(int32_t));

    std::vector<int> shape1 {100, 20, 30};
    Tensor t2(shape1.data(), shape1.size(), DataType::Int32);
    EXPECT_EQ(t2.shape(), TensorShape({100, 20, 30}));
    EXPECT_NE(t2.data<int32_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Int32);
    EXPECT_EQ(t2.size(), 100*20*30);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), sizeof(int32_t));
}


// Test the Tensor constructor from shape pointer, element size and type
TEST(Tensor, ShapePtrElementSizeAndTypeConstructor)
{
    std::vector<int> shape0 {100};
    Tensor t1(shape0.data(), shape0.size(), 15);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<int32_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::User);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), 15);

    std::vector<int> shape1 {100, 20, 30};
    Tensor t2(shape1.data(), shape1.size(), 27, DataType::Void);
    EXPECT_EQ(t2.shape(), TensorShape({100, 20, 30}));
    EXPECT_NE(t2.data<int32_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Void);
    EXPECT_EQ(t2.size(), 100*20*30);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), 27);
}



// Test the Tensor constructor from tensor shape and type
TEST(Tensor, TensorShapeAndTypeConstructor)
{
    Tensor t1(TensorShape({100}), DataType::Int32);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<int32_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::Int32);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), sizeof(int32_t));

    Tensor t2(TensorShape({100, 20, 30}), DataType::Int32);
    EXPECT_EQ(t2.shape(), TensorShape({100, 20, 30}));
    EXPECT_NE(t2.data<int32_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Int32);
    EXPECT_EQ(t2.size(), 100*20*30);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), sizeof(int32_t));
}


// Test the Tensor constructor from tensor shape, element size and type
TEST(Tensor, TensorShapeElementSizeAndTypeConstructor)
{
    Tensor t1(TensorShape({100}), 15);
    EXPECT_EQ(t1.shape(), TensorShape({100}));
    EXPECT_NE(t1.data<int32_t>(), nullptr);
    EXPECT_EQ(t1.dtype(), DataType::User);
    EXPECT_EQ(t1.size(), 100);
    EXPECT_TRUE(t1.is_own_data());
    EXPECT_EQ(t1.element_size(), 15);

    Tensor t2(TensorShape({100, 20, 30}), 27, DataType::Void);
    EXPECT_EQ(t2.shape(), TensorShape({100, 20, 30}));
    EXPECT_NE(t2.data<int32_t>(), nullptr);
    EXPECT_EQ(t2.dtype(), DataType::Void);
    EXPECT_EQ(t2.size(), 100*20*30);
    EXPECT_TRUE(t2.is_own_data());
    EXPECT_EQ(t2.element_size(), 27);
}


// Test the Tensor copy constructor
TEST(Tensor, CopyConstructor)
{
    Tensor other(100, DataType::Int32);

    Tensor t(other);
    EXPECT_EQ(t.shape(), other.shape());
    EXPECT_NE(t.data<int32_t>(), nullptr);
    EXPECT_NE(t.data<int32_t>(), other.data<int32_t>());
    EXPECT_EQ(t.dtype(), other.dtype());
    EXPECT_EQ(t.size(), other.size());
    EXPECT_EQ(t.is_own_data(), other.is_own_data());
    EXPECT_EQ(t.element_size(), other.element_size());
}


// Test the Tensor move constructor
TEST(Tensor, MoveConstructor)
{
    Tensor other(100, DataType::Int32);

    Tensor t(std::move(Tensor(100, DataType::Int32)));
    EXPECT_EQ(t.shape(), other.shape());
    EXPECT_NE(t.data<int32_t>(), nullptr);
    EXPECT_NE(t.data<int32_t>(), other.data<int32_t>());
    EXPECT_EQ(t.dtype(), other.dtype());
    EXPECT_EQ(t.size(), other.size());
    EXPECT_EQ(t.is_own_data(), other.is_own_data());
    EXPECT_EQ(t.element_size(), other.element_size());
}


// Test the Tensor copy operator
TEST(Tensor, CopyOperator)
{
    Tensor other(100, DataType::Int32);

    Tensor t(55, DataType::Uint8);
    t = other;
    EXPECT_EQ(t.shape(), other.shape());
    EXPECT_NE(t.data<int32_t>(), nullptr);
    EXPECT_EQ(t.data<int32_t>(), other.data<int32_t>());
    EXPECT_EQ(t.dtype(), other.dtype());
    EXPECT_EQ(t.size(), other.size());
    EXPECT_EQ(t.is_own_data(), other.is_own_data());
    EXPECT_EQ(t.element_size(), other.element_size());
}


// Test the Tensor move operator
TEST(Tensor, MoveOperator)
{
    Tensor other(100, DataType::Int32);
    Tensor ref(other);

    Tensor t(55, DataType::Uint8);
    t = std::move(ref);
    EXPECT_EQ(t.shape(), other.shape());
    EXPECT_NE(t.data<int32_t>(), nullptr);
    EXPECT_NE(t.data<int32_t>(), other.data<int32_t>());
    EXPECT_EQ(t.dtype(), other.dtype());
    EXPECT_EQ(t.size(), other.size());
    EXPECT_EQ(t.is_own_data(), other.is_own_data());
    EXPECT_EQ(t.element_size(), other.element_size());
}


#endif // TEST_TENSOR_H
