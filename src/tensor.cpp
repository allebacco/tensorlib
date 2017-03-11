#include "tensor.h"

using namespace tensor;


Tensor::Tensor():
    m_data(nullptr),
    m_shape(TensorShape()),
    m_dtype(DataType::Void),
    m_own_data(true)
{
}

Tensor::Tensor(const int64_t size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type),
    m_own_data(true)
{
    const int64_t allocation_size = size * size_of(type);
    m_data = new uint8_t[allocation_size];
}



Tensor::~Tensor()
{
    if(m_own_data && m_data!=nullptr)
    {
        delete[] m_data;
    }
}
