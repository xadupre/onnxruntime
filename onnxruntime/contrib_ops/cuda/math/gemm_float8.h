// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "gemm_float8_impl.cuh"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace contrib {
namespace cuda {

class GemmFloat8 final : public CudaKernel {
  using Base = CudaKernel;

 public:
  GemmFloat8(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    params_.trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    params_.trans_B_ = (temp != 0);

    params_.fast_accumulation_mode_ = info.GetAttrOrDefault<int64_t>("fastAccumulationMode", 1) != 0;

    std::string stemp = info.GetAttrOrDefault<std::string>("computeType", "CUBLAS_COMPUTE_32F");
    if (stemp == "CUBLAS_COMPUTE_16F") {
      params_.compute_type_ = CUBLAS_COMPUTE_16F;
      params_.scale_type_ = CUDA_R_16F;
    } else if (stemp == "CUBLAS_COMPUTE_32F") {
      params_.compute_type_ = CUBLAS_COMPUTE_32F;
      params_.scale_type_ = CUDA_R_32F;
    } else if (stemp == "CUBLAS_COMPUTE_32F_FAST_16F") {
      params_.compute_type_ = CUBLAS_COMPUTE_32F_FAST_16F;
      params_.scale_type_ = CUDA_R_16F;
    } else if (stemp == "CUBLAS_COMPUTE_32F_FAST_16BF") {
      params_.compute_type_ = CUBLAS_COMPUTE_32F_FAST_16BF;
      params_.scale_type_ = CUDA_R_16BF;
    } else if (stemp == "CUBLAS_COMPUTE_32F_FAST_TF32") {
      params_.compute_type_ = CUBLAS_COMPUTE_32F_FAST_TF32;
      params_.scale_type_ = CUDA_R_32F;
    } else {
      ORT_THROW("Unexpected value for compute_type: ", stemp, ".");
    }

    params_.sm_count_ = info.GetAttrOrDefault<int64_t>("smCount", 0);
    ORT_ENFORCE(info.GetAttr<float>("alpha", &params_.alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &params_.beta_).IsOK());
    ORT_ENFORCE(params_.trans_A_ && (!params_.trans_B_), "transA must be true and transB false, other cases are not implemented.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  GemmFloat8_Impl params_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime