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
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
    m_stride = TensorStride::from_shape_and_size(m_shape, m_element_size);
}


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
    m_stride = TensorStride::from_shape_and_size(m_shape, m_element_size);
}


template<typename Integer, typename>
Tensor::Tensor(const Integer* size, const int64_t ndim, const DataType type):
    m_data(nullptr),
    m_shape(size, ndim),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
    m_stride = TensorStride::from_shape_and_size(m_shape, m_element_size);
}


template<typename Integer, typename>
Tensor::Tensor(const Integer* size, const int64_t ndim, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(size, ndim),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
    m_stride = TensorStride::from_shape_and_size(m_shape, m_element_size);
}


template<typename _Tp, typename Integer, typename>
_Tp* Tensor::ptr(const Integer* indexes)
{
    uint8_t* dptr = m_data;
    const int64_t ndims = ndim();
    const int64_t* strides = stride().m_shape.data();

    for(int64_t i=0; i<ndims; ++i)
        dptr += strides[i] * indexes[i];

    return reinterpret_cast<_Tp*>(dptr);
}


template<typename _Tp, typename Integer, typename>
const _Tp* Tensor::ptr(const Integer* indexes) const
{
    const uint8_t* dptr = m_data;
    const int64_t ndims = ndim();
    const int64_t* strides = stride().m_shape.data();

    for(int64_t i=0; i<ndims; ++i)
        dptr += strides[i] * indexes[i];

    return reinterpret_cast<const _Tp*>(dptr);
}



template<typename _Tp, typename Integer, typename>
_Tp* Tensor::ptr(std::initializer_list<Integer> indexes)
{
    uint8_t* dptr = m_data;
    const int64_t ndims = std::min<int64_t>(ndim(), indexes.size());
    const int64_t* strides = stride().m_shape.data();
    std::initializer_list<Integer>::iterator indexes_it = indexes.begin();

    for(int64_t i=0; i<ndims; ++i)
        dptr += strides[i] * indexes_it[i];

    return reinterpret_cast<_Tp*>(dptr);
}



template<typename _Tp, typename Integer, typename>
const _Tp* Tensor::ptr(std::initializer_list<Integer> indexes) const
{
    const uint8_t* dptr = m_data;
    const int64_t ndims = std::min<int64_t>(ndim(), indexes.size());
    const int64_t* strides = stride().m_shape.data();
    std::initializer_list<Integer>::iterator indexes_it = indexes.begin();

    for(int64_t i=0; i<ndims; ++i)
        dptr += strides[i] * indexes_it[i];

    return reinterpret_cast<const _Tp*>(dptr);
}




}


#endif // TENSOR_P_H
