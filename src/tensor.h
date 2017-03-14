#ifndef TENSOR_H
#define TENSOR_H

#include "tensor_config.h"
#include "datatype_conversion.h"
#include "tensorshape.h"
#include "tensordata.h"
#include "tensor_traits.h"
#include "slice.h"

namespace tensor
{

class TENSOR_API Tensor
{
public:
    Tensor();

    Tensor(const int64_t size, const DataType type);

    Tensor(const int64_t size, const int64_t el_size, const DataType type=DataType::User);

    template<typename Integer, typename=enable_if_integer_t<Integer>>
    Tensor(std::initializer_list<Integer> size, const DataType type);

    template<typename Integer, typename=enable_if_integer_t<Integer>>
    Tensor(std::initializer_list<Integer> size, const int64_t el_size, const DataType type=DataType::User);

    template<typename Integer, typename=enable_if_integer_t<Integer>>
    Tensor(const Integer* size, const int64_t ndim, const DataType type);

    template<typename Integer, typename=enable_if_integer_t<Integer>>
    Tensor(const Integer* size, const int64_t ndim, const int64_t el_size, const DataType type=DataType::User);

    Tensor(const TensorShape& size, const DataType type);

    Tensor(const TensorShape& size, const int64_t el_size, const DataType type=DataType::User);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other);

    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);

    template<typename _Tp>
    const _Tp* data() const { return reinterpret_cast<const _Tp*>(m_data); }

    template<typename _Tp>
    _Tp* data() { return reinterpret_cast<_Tp*>(m_data); }

    const TensorShape& shape() const { return m_shape; }
    const int64_t shape(const int64_t index) const { return m_shape[index]; }
    const TensorStride& stride() const { return m_stride; }
    const int64_t stride(const int64_t index) const { return m_stride[index]; }
    int64_t size() const { return m_shape.size(); }
    int64_t ndim() const { return m_shape.ndim(); }
    DataType dtype() const { return m_dtype; }
    bool is_own_data() const { return m_shared_data.use_count()==1; }
    int64_t element_size() const { return m_element_size; }

    //Tensor clone() const;

    Tensor view(const Slice& slice) const;

    template<typename _Tp, typename Integer, typename=enable_if_integer_t<Integer>>
    _Tp* ptr(const Integer* indexes);

    template<typename _Tp, typename Integer, typename=enable_if_integer_t<Integer>>
    const _Tp* ptr(const Integer* indexes) const;

    template<typename _Tp, typename Integer, typename=enable_if_integer_t<Integer>>
    _Tp* ptr(std::initializer_list<Integer> indexes);

    template<typename _Tp, typename Integer, typename=enable_if_integer_t<Integer>>
    const _Tp* ptr(std::initializer_list<Integer> indexes) const;

private:

    TensorSharedData m_shared_data;
    uint8_t* m_data;
    TensorShape m_shape;
    TensorStride m_stride;
    DataType m_dtype = DataType::Void;
    int64_t m_element_size = 0;
};



}

// Include template implementations
#include "internal/tensor_p.h"

#endif // TENSOR_H
