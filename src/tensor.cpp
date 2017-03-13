#include "tensor.h"

using namespace tensor;


Tensor::Tensor():
    m_shared_data(std::make_shared<TensorData>()),
    m_data(nullptr),
    m_shape(TensorShape()),
    m_dtype(DataType::Void),
    m_element_size(1)
{
}

Tensor::Tensor(const int64_t size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
}

Tensor::Tensor(const int64_t size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
}

Tensor::Tensor(const TensorShape& size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
}

Tensor::Tensor(const TensorShape& size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
}

Tensor::Tensor(const Tensor& other):
    m_shared_data(other.m_shared_data),
    m_data(other.m_data),
    m_shape(other.m_shape),
    m_dtype(other.m_dtype),
    m_element_size(other.m_element_size)
{
}


Tensor::Tensor(Tensor&& other):
    m_shared_data(std::move(other.m_shared_data)),
    m_data(other.m_data),
    m_shape(std::move(other.m_shape)),
    m_dtype(other.m_dtype),
    m_element_size(other.m_element_size)
{
}


Tensor::~Tensor()
{
}


Tensor& Tensor::operator=(const Tensor& other)
{
    if(this!=&other)
    {
        m_shared_data = other.m_shared_data;
        m_data = other.m_data;
        m_shape = other.m_shape;
        m_dtype = other.m_dtype;
        m_element_size = other.m_element_size;
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
{
    if(this!=&other)
    {
        m_shared_data = std::move(other.m_shared_data);
        m_data = other.m_data;
        m_shape = std::move(other.m_shape);
        m_dtype = other.m_dtype;
        m_element_size = other.m_element_size;
    }

    return *this;
}
