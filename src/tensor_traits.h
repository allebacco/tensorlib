#ifndef TENSOR_TRAITS_H
#define TENSOR_TRAITS_H

#include <type_traits>

#include "tensor_config.h"

namespace tensor
{

template<class T>
using enable_if_integer_t = typename std::enable_if<std::is_integral<T>::value>::type;

template<class T>
using enable_if_float_t = typename std::enable_if<std::is_floating_point<T>::value>::type;

}

#endif // TENSOR_TRAITS_H
