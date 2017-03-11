#ifndef DATATYPE_H
#define DATATYPE_H

#include <type_traits>
#include <inttypes.h>

#include "tensor_config.h"

namespace tensor
{

enum class DataType
{
    Void,
    Int8, Int16, Int32, Int64,
    Uint8, Uint16, Uint32, Uint64,
    Float32, Float64,
    Float = Float32,
    Double = Float64,
    User = 255
};


static int64_t size_of(const DataType type)
{
    switch (type) {
    case DataType::Int8:
    case DataType::Uint8:
        return sizeof(int8_t);
    case DataType::Int16:
    case DataType::Uint16:
        return sizeof(int16_t);
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Float32:
        return sizeof(int32_t);
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Float64:
        return sizeof(int64_t);
    default:
        return sizeof(void*);
    }
}

}




#endif // DATATYPE_H
