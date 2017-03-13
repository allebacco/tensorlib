#ifndef TENSOR_H
#define TENSOR_H

#include "tensor_config.h"
#include "datatype_conversion.h"
#include "tensorshape.h"
#include "tensordata.h"


namespace tensor
{

class TENSOR_API Tensor
{
public:
    Tensor();

    Tensor(const int64_t size, const DataType type);

    Tensor(const int64_t size, const int64_t el_size, const DataType type=DataType::User);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(std::initializer_list<Integer> size, const DataType type);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(std::initializer_list<Integer> size, const int64_t el_size, const DataType type=DataType::User);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
    Tensor(const Integer* size, const int64_t ndim, const DataType type);

    template<typename Integer, typename=std::enable_if_t<std::is_integral<Integer>::value>>
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
    const TensorStride& stride() const { return m_stride; }
    int64_t size() const { return m_shape.size(); }
    int64_t ndim() const { return m_shape.ndim(); }
    DataType dtype() const { return m_dtype; }
    bool is_own_data() const { return m_shared_data.use_count()==1; }
    int64_t element_size() const { return m_element_size; }

    //Tensor clone() const;

    uint8_t* ptr(const int* idx)
    {
        int64_t d = ndim();
        uchar* p = m_data;

        for(int64_t i=0; i < d; i++)
        {
            p += idx[i] * m_stride[i];
        }
        return p;
    }

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
