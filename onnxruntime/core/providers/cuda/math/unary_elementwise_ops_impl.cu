// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/unary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

#define OP(name, expr)                                     \
  template <typename T>                                    \
  struct OP_##name {                                       \
    __device__ __inline__ T operator()(const T& a) const { \
      return expr;                                         \
    }                                                      \
  };

#define UNARY_ELEMENTWISE_IMPL(name)         \
  UNARY_ELEMENTWISE_IMPL_DECLARATION(name) { \
    UnaryElementWiseImpl(stream,             \
                         input_data,         \
                         output_data,        \
                         OP_##name<T>(),     \
                         count);             \
  }

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, T) \
  template void Impl_##name<T>(cudaStream_t stream, const T* input_data, T* output_data, size_t count);

#define UNARY_OP_NAME_EXPR(name, expr) \
  OP(name, expr)                       \
  UNARY_ELEMENTWISE_IMPL(name)

UNARY_OPS()
#undef UNARY_OP_NAME_EXPR

// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, half)     \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, float)    \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, double)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFDB(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name)        \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, BFloat16)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int8_t)       \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int16_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int32_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int64_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_BWUZCSILHFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint8_t)          \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint16_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint32_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint64_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(name)

SPECIALIZED_UNARY_ELEMENTWISE_IMPL_BWUZCSILHFD(Abs)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(Neg)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Floor)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Ceil)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Reciprocal)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Sqrt)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFDB(Log)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFDB(Exp)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Erf)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Round)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Sin)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Cos)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL(Not, bool)

// When casting, half needs to be converted via float type from most other types
template <typename T>
struct ViaTypeMap {
  typedef T ViaT;
};

template <>
struct ViaTypeMap<half> {
  typedef float ViaT;
};

template <>
struct ViaTypeMap<BFloat16> {
  typedef float ViaT;
};

template <typename InT, typename OutT>
struct OP_Cast {
  __device__ __inline__ OutT operator()(const InT& a) const {
    const bool any_float16 = std::is_same<half, InT>::value || std::is_same<half, OutT>::value;
    const bool any_bf16 = std::is_same<BFloat16, InT>::value || std::is_same<BFloat16, OutT>::value;
    typedef typename std::conditional<any_bf16, BFloat16, OutT>::type T1;
    typedef typename std::conditional<any_float16, half, T1>::type T;
    typedef typename ViaTypeMap<T>::ViaT ViaT;
    return (OutT)((ViaT)a);
  }
};

#define IMPL_CAST_IMPL(InT, OutT) \
  void _Explicit_Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count) { \
    UnaryElementWiseImpl(stream, input_data, output_data, OP_Cast<InT, OutT>(), count); \
  }

#define IMPL_CAST_IMPL_THROW(InT, OutT) \
  void _Explicit_Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count) { \
    ORT_THROW("Cast from " #InT " to " #OutT " must define saturate."); \
  }

#define IMPL_CAST_IMPL_FROM(T)      \
  IMPL_CAST_IMPL(T, half)     \
  IMPL_CAST_IMPL(T, float)    \
  IMPL_CAST_IMPL(T, double)   \
  IMPL_CAST_IMPL(T, int8_t)   \
  IMPL_CAST_IMPL(T, int16_t)  \
  IMPL_CAST_IMPL(T, int32_t)  \
  IMPL_CAST_IMPL(T, int64_t)  \
  IMPL_CAST_IMPL(T, uint8_t)  \
  IMPL_CAST_IMPL(T, uint16_t) \
  IMPL_CAST_IMPL(T, uint32_t) \
  IMPL_CAST_IMPL(T, uint64_t) \
  IMPL_CAST_IMPL(T, bool)     \
  IMPL_CAST_IMPL(T, BFloat16) \
  IMPL_CAST_IMPL_THROW(T, Float8E4M3FN) \
  IMPL_CAST_IMPL_THROW(T, Float8E4M3FNUZ) \
  IMPL_CAST_IMPL_THROW(T, Float8E5M2) \
  IMPL_CAST_IMPL_THROW(T, Float8E5M2FNUZ)

IMPL_CAST_IMPL_FROM(half)
IMPL_CAST_IMPL_FROM(float)
IMPL_CAST_IMPL_FROM(double)
IMPL_CAST_IMPL_FROM(int8_t)
IMPL_CAST_IMPL_FROM(int16_t)
IMPL_CAST_IMPL_FROM(int32_t)
IMPL_CAST_IMPL_FROM(int64_t)
IMPL_CAST_IMPL_FROM(uint8_t)
IMPL_CAST_IMPL_FROM(uint16_t)
IMPL_CAST_IMPL_FROM(uint32_t)
IMPL_CAST_IMPL_FROM(uint64_t)
IMPL_CAST_IMPL_FROM(bool)
IMPL_CAST_IMPL_FROM(BFloat16)
IMPL_CAST_IMPL_FROM(Float8E4M3FN)
IMPL_CAST_IMPL_FROM(Float8E4M3FNUZ)
IMPL_CAST_IMPL_FROM(Float8E5M2)
IMPL_CAST_IMPL_FROM(Float8E5M2FNUZ)



}  // namespace cuda
}  // namespace onnxruntime
