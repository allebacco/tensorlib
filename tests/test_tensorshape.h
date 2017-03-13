#ifndef TEST_TENSORSHAPE_H
#define TEST_TENSORSHAPE_H

#include "tensorshape.h"

#include "common.h"

using tensor::TensorShape;

// Test default constructor
TEST(TensorShape, DefaultConstructor)
{
    TensorShape ts;

    EXPECT_EQ(ts.size(), 0);
    EXPECT_EQ(ts.ndim(), 1);

    EXPECT_EQ(ts.shape(0), 0);

    EXPECT_EQ(ts[0], 0);
}


// Test the initializer list constructor
TEST(TensorShape, InitializerListConstructor)
{
    TensorShape ts({1, 2, 3});

    EXPECT_EQ(ts.size(), 1*2*3);
    EXPECT_EQ(ts.ndim(), 3);

    EXPECT_EQ(ts.shape(0), 1);
    EXPECT_EQ(ts.shape(1), 2);
    EXPECT_EQ(ts.shape(2), 3);

    EXPECT_EQ(ts[0], 1);
    EXPECT_EQ(ts[1], 2);
    EXPECT_EQ(ts[2], 3);
}


// Test the shape pointer constructor
TEST(TensorShape, ShapeConstructor)
{
    std::vector<int> shape {1, 2, 3};
    TensorShape ts(shape.data(), shape.size());

    EXPECT_EQ(ts.size(), 1*2*3);
    EXPECT_EQ(ts.ndim(), 3);

    EXPECT_EQ(ts.shape(0), 1);
    EXPECT_EQ(ts.shape(1), 2);
    EXPECT_EQ(ts.shape(2), 3);

    EXPECT_EQ(ts[0], 1);
    EXPECT_EQ(ts[1], 2);
    EXPECT_EQ(ts[2], 3);
}


// Test the shape pointer and stride constructor
TEST(TensorShape, ShapeStrideConstructor)
{
    std::vector<int> shape {1, 2, 3};
    TensorShape ts(shape.data(), shape.size());

    EXPECT_EQ(ts.size(), 1*2*3);
    EXPECT_EQ(ts.ndim(), 3);

    EXPECT_EQ(ts.shape(0), 1);
    EXPECT_EQ(ts.shape(1), 2);
    EXPECT_EQ(ts.shape(2), 3);

    EXPECT_EQ(ts[0], 1);
    EXPECT_EQ(ts[1], 2);
    EXPECT_EQ(ts[2], 3);
}


// Test the copy constructor
TEST(TensorShape, CopyConstructor)
{
    TensorShape other({1, 2, 3});
    TensorShape ts(other);

    EXPECT_EQ(ts.size(), other.size());
    EXPECT_EQ(ts.ndim(), other.ndim());

    for(int i=0; i<ts.ndim(); ++i)
    {
        EXPECT_EQ(ts.shape(i), other.shape(i));
        EXPECT_EQ(ts[i], other[i]);
    }
}


// Test the move constructor
TEST(TensorShape, MoveConstructor)
{
    TensorShape reference({1, 2, 3});
    TensorShape other(reference);
    TensorShape ts(std::move(other));

    EXPECT_EQ(ts.size(), reference.size());
    EXPECT_EQ(ts.ndim(), reference.ndim());

    for(int i=0; i<ts.ndim(); ++i)
    {
        EXPECT_EQ(ts.shape(i), reference.shape(i));
        EXPECT_EQ(ts[i], reference[i]);
    }
}


// Test the copy operator
TEST(TensorShape, CopyOperator)
{
    TensorShape other({1, 2, 3});
    TensorShape ts;
    ts = other;

    EXPECT_EQ(ts.size(), other.size());
    EXPECT_EQ(ts.ndim(), other.ndim());

    for(int i=0; i<ts.ndim(); ++i)
    {
        EXPECT_EQ(ts.shape(i), other.shape(i));
        EXPECT_EQ(ts[i], other[i]);
    }
}


// Test the move operator
TEST(TensorShape, MoveOperator)
{
    TensorShape reference({1, 2, 3});
    TensorShape other(reference);
    TensorShape ts;
    ts = std::move(other);

    EXPECT_EQ(ts.size(), reference.size());
    EXPECT_EQ(ts.ndim(), reference.ndim());

    for(int i=0; i<ts.ndim(); ++i)
    {
        EXPECT_EQ(ts.shape(i), reference.shape(i));
        EXPECT_EQ(ts[i], reference[i]);
    }
}


// Test the equal operator
TEST(TensorShape, EqualOperator)
{
    TensorShape other({1, 2, 3});
    TensorShape ts = other;
    EXPECT_EQ(ts, other);

    ts = TensorShape({1, 2, 3, 4});
    EXPECT_NE(ts, other);

    std::vector<int> shape {1, 2, 3};
    ts = TensorShape(shape.data(), shape.size());
    EXPECT_EQ(ts, other);
}

#endif // TEST_TENSORSHAPE_H
