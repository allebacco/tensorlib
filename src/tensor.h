#ifndef TENSOR_H
#define TENSOR_H

#include "datatype_conversion.h"
#include "tensorshape.h"

namespace tensor
{



class Tensor
{
public:
    Tensor();

    Tensor(const int64_t size, const DataType type);

    template<typename _Tp>
    Tensor(const int64_t size);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(std::initializer_list<Integer> shape_values, const DataType type);

    template<typename _Tp, typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(std::initializer_list<Integer> shape_values);

    ~Tensor();

    template<typename _Tp>
    const _Tp* data() const { return reinterpret_cast<const _Tp*>(m_data); }

    template<typename _Tp>
    _Tp* data() { return reinterpret_cast<_Tp*>(m_data); }

    const TensorShape& shape() const { return m_shape; }
    const int64_t size() const { return m_shape.size(); }
    const int64_t ndim() const { return m_shape.ndim(); }
    DataType dtype() const { return m_dtype; }
    bool is_own_data() const { return m_own_data; }

private:

    uint8_t* m_data = nullptr;
    TensorShape m_shape;
    DataType m_dtype = DataType::Void;
    bool m_own_data = true;
};


template<typename _Tp>
Tensor::Tensor(const int64_t size):
    m_data(nullptr),
    m_shape(TensorShape({size})),
    m_dtype(type_to_dtype<_Tp>::value),
    m_own_data(true)
{
    m_dtype = type_to_dtype<_Tp>::value;
    const int64_t allocation_size = size * sizeof(_Tp);
    m_data = new uint8_t[allocation_size];
}


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> shape_values, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(shape_values)),
    m_dtype(type),
    m_own_data(true)
{
    const int64_t allocation_size = m_shape.size() * size_of(type);
    m_data = new uint8_t[allocation_size];
}


template<typename _Tp, typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> shape_values):
    m_data(nullptr),
    m_shape(TensorShape(shape_values)),
    m_dtype(type_to_dtype<_Tp>::value),
    m_own_data(true)
{
    const int64_t allocation_size = m_shape.size() * sizeof(_Tp);
    m_data = new uint8_t[allocation_size];
}


}

#endif // TENSOR_H
