#ifndef TEST_DATATYPE_H
#define TEST_DATATYPE_H

#include <gtest/gtest.h>

#include "common.h"
#include "datatype_conversion.h"

using tensor::DataType;
using tensor::type_to_dtype;
using tensor::dtype_to_type;


// Tests DataType conversion from C++ types
TEST(DataTypeEnum, ConversionFromType)
{
    // Test signed interger types
    EXPECT_EQ(type_to_dtype<int8_t>::value, DataType::Int8);
    EXPECT_EQ(type_to_dtype<int16_t>::value, DataType::Int16);
    EXPECT_EQ(type_to_dtype<int32_t>::value, DataType::Int32);
    EXPECT_EQ(type_to_dtype<int64_t>::value, DataType::Int64);

    // Test unsigned interger types
    EXPECT_EQ(type_to_dtype<uint8_t>::value, DataType::Uint8);
    EXPECT_EQ(type_to_dtype<uint16_t>::value, DataType::Uint16);
    EXPECT_EQ(type_to_dtype<uint32_t>::value, DataType::Uint32);
    EXPECT_EQ(type_to_dtype<uint64_t>::value, DataType::Uint64);

    // Test floating point types
    EXPECT_EQ(type_to_dtype<float>::value, DataType::Float);
    EXPECT_EQ(type_to_dtype<float>::value, DataType::Float32);
    EXPECT_EQ(type_to_dtype<double>::value, DataType::Double);
    EXPECT_EQ(type_to_dtype<double>::value, DataType::Float64);

    // Test void type
    EXPECT_EQ(type_to_dtype<void>::value, DataType::Void);

    // Test user type
    EXPECT_EQ(type_to_dtype<float*>::value, DataType::User);
    EXPECT_EQ(type_to_dtype<std::string>::value, DataType::User);
}


// Tests DataType conversion from C++ types
TEST(DataTypeEnum, ConversionToType)
{
    // Test signed interger types
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Int8>::type, int8_t);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Int16>::type, int16_t);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Int32>::type, int32_t);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Int64>::type, int64_t);

    // Test unsigned interger types
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Uint8>::type, uint8_t);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Uint16>::type, uint16_t);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Uint32>::type, uint32_t);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Uint64>::type, uint64_t);

    // Test floating point types
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Float>::type, float);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Float32>::type, float);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Double>::type, double);
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Float64>::type, double);

    // Test void type
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::Void>::type, void);

    // Test user type
    EXPECT_SAME_TYPE(typename dtype_to_type<DataType::User>::type, void);
}

#endif // TEST_DATATYPE_H
