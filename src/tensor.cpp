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
    allocate();
}

Tensor::Tensor(const int64_t size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type),
    m_own_data(true),
    m_element_size(el_size)
{
    allocate();
}

Tensor::Tensor(const TensorShape& size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(size_of(type))
{
    allocate();
}

Tensor::Tensor(const TensorShape& size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(el_size)
{
    allocate();
}

Tensor::Tensor(const Tensor& other):
    m_data(nullptr),
    m_shape(other.m_shape),
    m_dtype(other.m_dtype),
    m_own_data(true),
    m_element_size(other.m_element_size)
{
    allocate();
}


Tensor::Tensor(Tensor&& other):
    m_data(other.m_data),
    m_shape(std::move(other.m_shape)),
    m_dtype(other.m_dtype),
    m_own_data(other.m_own_data),
    m_element_size(other.m_element_size)
{
    // release other resources
    other.m_data = nullptr;
    other.m_own_data = false;
}



Tensor::~Tensor()
{
    if(m_own_data && m_data!=nullptr)
    {
        delete[] m_data;
    }
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if(this!=&other)
    {
        deallocate();

        m_shape = other.m_shape;
        m_dtype = other.m_dtype;
        m_own_data = true;
        m_element_size = other.m_element_size;

        allocate();
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
{
    if(this!=&other)
    {
        deallocate();

        m_data = other.m_data;
        m_shape = std::move(other.m_shape);
        m_dtype = other.m_dtype;
        m_own_data = other.m_own_data;
        m_element_size = other.m_element_size;

        // release other resources
        other.m_data = nullptr;
        other.m_own_data = false;
    }

    return *this;
}

void Tensor::allocate()
{
    if(m_own_data && m_data!=nullptr)
        delete[] m_data;

    const int64_t allocation_size = m_shape.size() * m_element_size;
    m_data = new uint8_t[allocation_size];

    m_own_data = true;
}

void Tensor::deallocate()
{
    if(m_own_data && m_data!=nullptr)
        delete[] m_data;

    m_data = nullptr;
    m_shape = TensorShape();
    m_dtype = DataType::Void;
    m_own_data = true;
    m_element_size = 1;
}
