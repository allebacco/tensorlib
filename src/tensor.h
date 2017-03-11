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

    ~Tensor();

protected:

    uint8_t* m_data = nullptr;
    TensorShape m_shape;
    DataType m_dtype = DataType::Void;
    bool m_own_data = true;
};

}

#endif // TENSOR_H
