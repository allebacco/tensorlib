#ifndef TENSOR_P_H
#define TENSOR_P_H

#include "../tensor.h"

namespace tensor
{


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_stride(size),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();

    const int64_t num_dims = m_stride.ndim();
    for(int64_t i=0; i<num_dims; ++i)
        m_stride.m_shape[i] *= m_element_size;
}


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_stride(size),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();

    const int64_t num_dims = m_stride.ndim();
    for(int64_t i=0; i<num_dims; ++i)
        m_stride.m_shape[i] *= m_element_size;
}


template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
Tensor::Tensor(const Integer* size, const int64_t ndim, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size, ndim)),
    m_stride(size, ndim),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();

    const int64_t num_dims = m_stride.ndim();
    for(int64_t i=0; i<num_dims; ++i)
        m_stride.m_shape[i] *= m_element_size;
}


template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
Tensor::Tensor(const Integer* size, const int64_t ndim, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size, ndim)),
    m_stride(size, ndim),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();

    const int64_t num_dims = m_stride.ndim();
    for(int64_t i=0; i<num_dims; ++i)
        m_stride.m_shape[i] *= m_element_size;
}



}


#endif // TENSOR_P_H
