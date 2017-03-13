#include "tensordata.h"

using namespace tensor;


TensorData::TensorData():
    m_data(nullptr),
    m_size(0)
{
}


TensorData::TensorData(const int64_t m_size):
    m_data(nullptr),
    m_size(m_size)
{
    allocate();
}


TensorData::TensorData(const uint8_t* src_data, int64_t m_size):
    m_data(nullptr),
    m_size(m_size)
{
    allocate();
    memcpy(m_data, src_data, m_size);
}


TensorData::TensorData(const TensorData& other):
    m_data(nullptr),
    m_size(m_size)
{
    allocate();
    memcpy(m_data, other.m_data, m_size);
}


TensorData::TensorData(TensorData&& other)
{
    m_size = other.m_size;
    m_data = other.m_data;
    other.m_data = nullptr;
}

TensorData::~TensorData()
{
    deallocate();
}

TensorData& TensorData::operator=(const TensorData& other)
{
    if(this!=&other)
    {
        deallocate();

        m_size = other.m_size;
        allocate();
        memcpy(m_data, other.m_data, m_size);
    }

    return *this;
}

TensorData& TensorData::operator=(TensorData&& other)
{
    if(this!=&other)
    {
        deallocate();

        m_size = other.m_size;
        m_data = other.m_data;
        other.m_data = nullptr;
    }

    return *this;
}

TensorSharedData TensorData::make_shared(const int64_t size, const uint8_t* src_data)
{
    if(src_data==nullptr)
        return std::make_shared<TensorData>(size);

    return std::make_shared<TensorData>(src_data, size);
}

void TensorData::allocate()
{
    if(m_data!=nullptr)
        deallocate();

    m_data = new uint8_t[m_size];
}

void TensorData::deallocate()
{
    if(m_data!=nullptr)
        delete[] m_data;

    m_data = nullptr;
}
