#include "tensor.h"

#include <algorithm>
#include <numeric>

using namespace tensor;


Tensor::Tensor():
    m_shared_data(std::make_shared<TensorData>()),
    m_data(nullptr),
    m_shape(TensorShape()),
    m_dtype(DataType::Void),
    m_element_size(1),
    m_stride(TensorStride())
{
}

Tensor::Tensor(const int64_t size, const DataType type):
    m_data(nullptr),
    m_shape({size}),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
    m_stride = TensorStride::from_shape_and_size(m_shape, m_element_size);
}

Tensor::Tensor(const int64_t size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape({size}),
    m_dtype(type),
    m_element_size(el_size)
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
    m_stride = TensorStride::from_shape_and_size(m_shape, m_element_size);
}

Tensor::Tensor(const TensorShape& size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_stride(TensorStride::from_shape_and_size(size, size_of(type))),
    m_dtype(type),
    m_element_size(size_of(type))
{
    m_shared_data = TensorData::make_shared(m_shape.size() * m_element_size);
    m_data = m_shared_data->data();
}

Tensor::Tensor(const TensorShape& size, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(size)),
    m_stride(TensorStride::from_shape_and_size(size, size_of(type))),
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
    m_element_size(other.m_element_size),
    m_stride(m_stride)
{
}


Tensor::Tensor(Tensor&& other):
    m_shared_data(std::move(other.m_shared_data)),
    m_data(other.m_data),
    m_shape(std::move(other.m_shape)),
    m_dtype(other.m_dtype),
    m_element_size(other.m_element_size),
    m_stride(std::move(m_stride))
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
        m_stride = other.m_stride;
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
        m_stride = std::move(other.m_stride);
    }

    return *this;
}

Tensor Tensor::view(const Slice& slice) const
{
    Tensor result(*this);

    const int64_t stop = std::min(slice.stop(), result.shape(0));
    const int64_t start = std::min(slice.start(), result.shape(0));
    if(start==stop)
    {
        TensorShape new_shape(m_shape);
        new_shape.set_shape(0, 0);
        return Tensor(new_shape, m_element_size, m_dtype);
    }

    result.m_data = result.ptr<uint8_t>({start});
    result.m_shape.m_shape[0] = stop - start;

    return result;
}


