#include "tensor.h"

using namespace tensor;


Tensor::Tensor()
{

}

Tensor::~Tensor()
{
    if(m_own_data && m_data!=nullptr)
    {
        delete[] m_data;
    }
}
