#ifndef TENSORSHAPE_H
#define TENSORSHAPE_H

#include <vector>
#include <cstdlib>

#include "tensor_config.h"

namespace tensor
{


class TensorShape;

typedef TensorShape TensorStride;


class TENSOR_API TensorShape
{
    friend class Tensor;
public:
    TensorShape();

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    TensorShape(std::initializer_list<Integer> shape_values);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    TensorShape(Integer* shape_values, const int64_t num_dims);

    TensorShape(const TensorShape& other);
    TensorShape(TensorShape&& other);

    ~TensorShape();

    TensorShape& operator=(const TensorShape& other);
    TensorShape& operator=(TensorShape&& other);

    int64_t size() const;
    int64_t ndim() const { return m_shape.size(); }
    int64_t shape(const int64_t index) const { return m_shape[index]; }
    int64_t operator[](const int64_t index) const { return shape(index); }

    bool is_equal(const TensorShape& other) const;

public:

    static TensorStride from_shape_and_size(const TensorShape& shape, const int64_t element_size);

protected:

    TensorShape(const int64_t num_dims);

    void set_shape(const int64_t index, int64_t value);

    template<class Container1>
    void assign_shape(const Container1& shape_data);

private:

    void set_ndim(const int64_t num_dims);

private:

    std::vector<int64_t> m_shape;
};


static bool operator==(const TensorShape& ts1, const TensorShape& ts2)
{
    return ts1.is_equal(ts2);
}

static bool operator!=(const TensorShape& ts1, const TensorShape& ts2)
{
    return !(ts1==ts2);
}



template<typename Integer, typename>
TensorShape::TensorShape(std::initializer_list<Integer> shape_values)
{
    set_ndim(shape_values.size());
    assign_shape(shape_values.begin());
}


template<typename Integer, typename>
TensorShape::TensorShape(Integer* shape_values, const int64_t num_dims)
{
   set_ndim(num_dims);
   assign_shape(shape_values);
}

template<class Container1>
void TensorShape::assign_shape(const Container1& shape_data)
{
    const int64_t num_dims = m_shape.size();
    for(int64_t i=0; i<num_dims; ++i)
        m_shape[i] = shape_data[i];
}


}



#endif // TENSORSHAPE_H
