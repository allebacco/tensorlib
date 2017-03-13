#ifndef TENSOR_P_H
#define TENSOR_P_H

#include "../tensor.h"

namespace tensor
{


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(size_of(type))
{
    allocate();
}


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(el_size)
{
    allocate();
}


template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
Tensor::Tensor(const Integer* size, const int64_t ndim, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size, ndim)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(size_of(type))
{
    allocate();
}


template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
Tensor::Tensor(const Integer* size, const int64_t ndim, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size, ndim)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(el_size)
{
    allocate();
}



}


#endif // TENSOR_P_H
