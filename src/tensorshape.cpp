#include "tensorshape.h"

#include <algorithm>
#include <numeric>


using namespace tensor;


TensorShape::TensorShape():
    m_shape(1)
{
    m_shape[0] = 0;
}

TensorShape::TensorShape(const TensorShape& other) :
    m_shape(other.m_shape)
{
}

TensorShape::TensorShape(TensorShape&& other) :
    m_shape(other.m_shape)
{
}

TensorShape::~TensorShape()
{
}

TensorShape& TensorShape::operator=(const TensorShape& other)
{
    if (this != &other)
    {
        m_shape = other.m_shape;
    }
    return *this;
}


TensorShape& TensorShape::operator=(TensorShape&& other)
{
    if (this != &other)
    {
        std::swap(m_shape, other.m_shape);
    }
    return *this;
}


int64_t TensorShape::size() const
{
    return std::accumulate(m_shape.begin(), m_shape.end(), static_cast<int64_t>(1L), std::multiplies<int64_t>());
}

bool TensorShape::is_equal(const TensorShape& other) const
{
    return m_shape == other.m_shape;
}

TensorStride TensorShape::from_shape_and_size(const TensorShape& shape, const int64_t element_size)
{
    TensorStride ts(shape);
    const int num_dims = shape.ndim();
    for(int i=0; i<num_dims; ++i)
        ts.m_shape[i] *= element_size;

    return ts;
}

TensorShape::TensorShape(const int64_t num_dims)
{
    set_ndim(num_dims);
}

void TensorShape::set_ndim(const int64_t num_dims)
{
    m_shape.resize(num_dims);
}

void TensorShape::set_shape(const int64_t index, int64_t value)
{
    m_shape[index] = value;
}
