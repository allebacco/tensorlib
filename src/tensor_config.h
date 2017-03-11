#ifndef TENSOR_CONFIG_H
#define TENSOR_CONFIG_H

#ifdef _MSC_VER
    #ifdef TENSOR_API_EXPORT
        #define TENSOR_API __declspec(dllexport)
    #else
        #define TENSOR_API __declspec(dllimport)
    #endif
#else
    #define TENSOR_API
#endif



#endif // TENSOR_CONFIG_H
