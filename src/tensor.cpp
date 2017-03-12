#include "tensor.h"

using namespace tensor;


Tensor::Tensor():
    m_data(nullptr),
    m_shape(TensorShape()),
    m_dtype(DataType::Void),
    m_own_data(true),
    m_element_size(1)
{
}

Tensor::Tensor(const int64_t size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type),
    m_own_data(true),
    m_element_size(size_of(type))
{
    const int64_t allocation_size = size * m_element_size;
    m_data = new uint8_t[allocation_size];
}

Tensor::Tensor(const int64_t size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type),
    m_own_data(true),
    m_element_size(el_size)
{
    const int64_t allocation_size = size * m_element_size;
    m_data = new uint8_t[allocation_size];
}



Tensor::~Tensor()
{
    if(m_own_data && m_data!=nullptr)
    {
        delete[] m_data;
    }
}
