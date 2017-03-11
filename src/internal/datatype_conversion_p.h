#ifndef DATATYPE_CONVERSION_P_H
#define DATATYPE_CONVERSION_P_H

#include <inttypes.h>

#include "datatype.h"

namespace tensor
{

namespace internal
{


template<DataType dt>
struct datatype_constant: public std::integral_constant<DataType, dt>
{};


template<bool B, DataType t1, DataType t2>
struct selector
{
    static constexpr DataType value = t1;
};

template<DataType t1, DataType t2>
struct selector<false, t1, t2>
{
    static constexpr DataType value = t2;
};


template<typename dtype>
struct type_to_dtype_sint: public
    std::conditional_t<sizeof(dtype)==1, datatype_constant<DataType::Int8>,
        std::conditional_t<sizeof(dtype)==2, datatype_constant<DataType::Int16>,
        std::conditional_t<sizeof(dtype)==4, datatype_constant<DataType::Int32>,
        std::conditional_t<sizeof(dtype)==8, datatype_constant<DataType::Int64>,datatype_constant<DataType::User>
        > > > >
{
};


template<typename dtype>
struct type_to_dtype_uint: public
    std::conditional_t<sizeof(dtype)==1, datatype_constant<DataType::Uint8>,
        std::conditional_t<sizeof(dtype)==2, datatype_constant<DataType::Uint16>,
        std::conditional_t<sizeof(dtype)==4, datatype_constant<DataType::Uint32>,
        std::conditional_t<sizeof(dtype)==8, datatype_constant<DataType::Uint64>,datatype_constant<DataType::User>
        > > > >
{
};


template<typename dtype>
struct type_to_dtype_int: public
    std::conditional_t<std::is_signed<dtype>::value, type_to_dtype_sint<dtype>,
        std::conditional_t<std::is_unsigned<dtype>::value, type_to_dtype_uint<dtype>, datatype_constant<DataType::User>
        > >
{
};


template<typename dtype>
struct type_to_dtype_float: public
    std::conditional_t<sizeof(dtype)==4, datatype_constant<DataType::Float32>,
        std::conditional_t<sizeof(dtype)==8, datatype_constant<DataType::Float64>, datatype_constant<DataType::User>
        > >
{
};

}


}


#endif // DATATYPE_CONVERSION_P_H
