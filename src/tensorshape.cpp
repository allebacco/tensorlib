#include "tensorshape.h"

#include <algorithm>
#include <numeric>


using namespace tensor;


TensorShape::TensorShape():
    m_internal_data(2),
    m_num_dims(1)
{
    m_internal_data[0] = 0;
    m_shape = &m_internal_data[0];
    m_strides = &m_internal_data[m_num_dims];
}

TensorShape::TensorShape(const TensorShape& other) :
    m_num_dims(other.m_num_dims),
    m_internal_data(other.m_internal_data)
{
    m_shape = &m_internal_data[0];
    m_strides = &m_internal_data[m_num_dims];
}

TensorShape::TensorShape(TensorShape&& other) :
    m_num_dims(other.m_num_dims)
{
    std::swap(m_internal_data, other.m_internal_data);
    m_shape = &m_internal_data[0];
    m_strides = &m_internal_data[m_num_dims];
}

TensorShape::~TensorShape()
{
    m_shape = nullptr;
    m_strides = nullptr;
    m_num_dims = 0;
}

TensorShape& TensorShape::operator=(const TensorShape& other)
{
    if (this != &other)
    {
        m_num_dims = other.m_num_dims;
        m_internal_data = other.m_internal_data;
        m_shape = &m_internal_data[0];
        m_strides = &m_internal_data[m_num_dims];
    }
    return *this;
}


TensorShape& TensorShape::operator=(TensorShape&& other)
{
    if (this != &other)
    {
        m_num_dims = other.m_num_dims;
        std::swap(m_internal_data, other.m_internal_data);
        m_shape = &m_internal_data[0];
        m_strides = &m_internal_data[m_num_dims];
    }
    return *this;
}


int64_t TensorShape::size() const
{
    return std::accumulate(m_shape, m_shape + m_num_dims, static_cast<int64_t>(1L), std::multiplies<int64_t>());
}

bool TensorShape::is_equal(const TensorShape& other) const
{
    return m_internal_data == other.m_internal_data;
}

TensorShape::TensorShape(const int64_t num_dims)
{
    set_ndim(num_dims);
}

void TensorShape::set_ndim(const int64_t num_dims)
{
    m_num_dims = num_dims;
    m_internal_data.resize(num_dims * 2);
    m_shape = &m_internal_data[0];
    m_strides = &m_internal_data[num_dims];
}

void TensorShape::set_shape(const int64_t index, int64_t value)
{
    m_shape[index] = value;
}

void TensorShape::set_stride(const int64_t index, int64_t value)
{
    m_strides[index] = value;
}
