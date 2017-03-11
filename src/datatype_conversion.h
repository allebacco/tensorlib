#ifndef DATATYPE_CONVERSION_H
#define DATATYPE_CONVERSION_H

#include "datatype.h"
#include "internal/datatype_conversion_p.h"

namespace tensor
{

template<typename type>
struct type_to_dtype: public
    std::conditional_t<std::is_void<type>::value, internal::datatype_constant<DataType::Void>,
        std::conditional_t<std::is_integral<type>::value,
                           internal::type_to_dtype_int<type>,
        std::conditional_t<std::is_floating_point<type>::value,
                           internal::type_to_dtype_float<type>, internal::datatype_constant<DataType::User>
        > > >
{
};


template<DataType dtype>
struct dtype_to_type
{
    using type = std::conditional_t<dtype==DataType::Int8, int8_t,
        std::conditional_t<dtype==DataType::Int16, int16_t,
        std::conditional_t<dtype==DataType::Int32, int32_t,
        std::conditional_t<dtype==DataType::Int64, int64_t,
        std::conditional_t<dtype==DataType::Uint8, uint8_t,
        std::conditional_t<dtype==DataType::Uint16, uint16_t,
        std::conditional_t<dtype==DataType::Uint32, uint32_t,
        std::conditional_t<dtype==DataType::Uint64, uint64_t,
        std::conditional_t<dtype==DataType::Float32, float,
        std::conditional_t<dtype==DataType::Float64, double, void>
        > > > > > > > > >;
};



}


#endif // DATATYPE_CONVERSION_H
