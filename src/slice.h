#ifndef SLICE_H
#define SLICE_H

#include <inttypes.h>
#include <limits>

#include "tensor_config.h"

namespace tensor
{

class TENSOR_API Slice
{
public:
    static constexpr int64_t MaxStop = std::numeric_limits<int64_t>::max() - 1;

public:   
    Slice(const int64_t start=0, const int64_t stop=MaxStop, const int64_t step=1):
        m_start(start), m_stop(stop), m_step(step)
    {}

    int64_t start() const { return m_start; }
    int64_t stop() const { return m_stop; }
    int64_t step() const { return m_step; }

private:

    int64_t m_start;
    int64_t m_stop;
    int64_t m_step;
};


}

#endif // SLICE_H
