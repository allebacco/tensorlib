#ifndef TENSORSHAPE_H
#define TENSORSHAPE_H

#include <vector>
#include <cstdlib>

namespace tensor
{


class TensorShape
{
public:
    TensorShape();

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    TensorShape(std::initializer_list<Integer> shape_values);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    TensorShape(Integer* shape_values, const int64_t num_dims);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    TensorShape(Integer* shape_values, Integer* stride_values, const int64_t num_dims);

    TensorShape(const TensorShape& other);
    TensorShape(TensorShape&& other);

    ~TensorShape();

    TensorShape& operator=(const TensorShape& other);
    TensorShape& operator=(TensorShape&& other);

    int64_t size() const;
    int64_t ndim() const { return m_num_dims; }
    int64_t shape(const int64_t index) const { return m_shape[index]; }
    int64_t stride(const int64_t index) const { return m_strides[index]; }
    int64_t operator[](const int64_t index) const { return shape(index); }

private:

    void init_num_dims(const int64_t num_dims);

    template<class Container1, class Container2>
    void assign_shape(const Container1& shape_data, const Container2& stride_data);

private:

    std::vector<int64_t> m_internal_data;
    int64_t* m_shape;
    int64_t* m_strides;
    int64_t m_num_dims;
};


template<typename Integer, typename>
TensorShape::TensorShape(std::initializer_list<Integer> shape_values)
{
    init_num_dims(shape_values.size());
    assign_shape(shape_values.begin(), shape_values.begin());
}


template<typename Integer, typename>
TensorShape::TensorShape(Integer* shape_values, const int64_t num_dims)
{
   init_num_dims(num_dims);
   assign_shape(shape_values, shape_values);
}

template<typename Integer, typename>
TensorShape::TensorShape(Integer* shape_values, Integer* stride_values, const int64_t num_dims)
{
    init_num_dims(num_dims);
    assign_shape(shape_values, stride_values);
}


template<class Container1, class Container2>
void TensorShape::assign_shape(const Container1& shape_data, const Container2& stride_data)
{
    for(size_t i=0; i<m_num_dims; ++i)
    {
        m_shape[i] = shape_data[i];
        m_strides[i] = stride_data[i];
    }
}


}



#endif // TENSORSHAPE_H
