#ifndef TENSORDATA_H
#define TENSORDATA_H

#include "tensor_config.h"


#include <memory>

namespace tensor
{


class TensorData;

typedef std::shared_ptr<TensorData> TensorSharedData;

class TENSOR_API TensorData
{
public:
    TensorData();
    TensorData(const int64_t m_size);
    TensorData(const uint8_t* src_data, int64_t m_size);
    TensorData(const TensorData& other);
    TensorData(TensorData&& other);

    ~TensorData();

    TensorData& operator=(const TensorData& other);
    TensorData& operator=(TensorData&& other);

    const uint8_t* data() const { return m_data; }
    uint8_t* data() { return m_data; }

    int64_t size() const { return m_size; }

public:

    static TensorSharedData make_shared(const int64_t size, const uint8_t* src_data=nullptr);

protected:

    virtual void allocate();
    virtual void deallocate();

protected:

    uint8_t* m_data;
    int64_t m_size;
};


}

#endif // TENSORDATA_H
