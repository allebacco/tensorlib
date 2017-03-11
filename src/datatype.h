#ifndef DATATYPE_H
#define DATATYPE_H

#include <type_traits>

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


}




#endif // DATATYPE_H
