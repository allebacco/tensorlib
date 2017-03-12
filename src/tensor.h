#ifndef TENSOR_H
#define TENSOR_H

#include "tensor_config.h"
#include "datatype_conversion.h"
#include "tensorshape.h"

namespace tensor
{

class TENSOR_API Tensor
{
public:
    Tensor();

    Tensor(const int64_t size, const DataType type);

    Tensor(const int64_t size, const int64_t el_size, const DataType type=DataType::User);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(std::initializer_list<Integer> shape_values, const DataType type);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(std::initializer_list<Integer> shape_values, const int64_t el_size, const DataType type=DataType::User);

    ~Tensor();

    template<typename _Tp>
    const _Tp* data() const { return reinterpret_cast<const _Tp*>(m_data); }

    template<typename _Tp>
    _Tp* data() { return reinterpret_cast<_Tp*>(m_data); }

    const TensorShape& shape() const { return m_shape; }
    int64_t size() const { return m_shape.size(); }
    int64_t ndim() const { return m_shape.ndim(); }
    DataType dtype() const { return m_dtype; }
    bool is_own_data() const { return m_own_data; }
    int64_t element_size() const { return m_element_size; }

private:

    uint8_t* m_data = nullptr;
    TensorShape m_shape;
    DataType m_dtype = DataType::Void;
    bool m_own_data = true;
    int64_t m_element_size = 0;
};



template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> shape_values, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(shape_values)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(size_of(type))
{
    const int64_t allocation_size = m_shape.size() * m_element_size;
    m_data = new uint8_t[allocation_size];
}


template<typename Integer, typename>
Tensor::Tensor(std::initializer_list<Integer> shape_values, const int64_t el_size, const DataType type):
    m_data(nullptr),
    m_shape(TensorShape(shape_values)),
    m_dtype(type),
    m_own_data(true),
    m_element_size(el_size)
{
    const int64_t allocation_size = m_shape.size() * m_element_size;
    m_data = new uint8_t[allocation_size];
}


}

#endif // TENSOR_H
