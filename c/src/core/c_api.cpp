/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "exceptions.hpp"

#include <cudf/core/c_api.h>
#include <cudf/utilities/error.hpp>
#include <cudf/version_config.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

struct CudfResources {
  cudaStream_t stream = nullptr;
};

CudfResources* get_resources(cudfResources_t res)
{
  CUDF_EXPECTS(res != 0, "cudfResources_t cannot be null");
  return reinterpret_cast<CudfResources*>(res);
}

thread_local std::string last_error_text;

}  // namespace

extern "C" CUDF_C_EXPORT const char* cudfGetLastErrorText(void)
{
  return last_error_text.empty() ? nullptr : last_error_text.c_str();
}

extern "C" CUDF_C_EXPORT void cudfSetLastErrorText(const char* error)
{
  last_error_text = error == nullptr ? "" : error;
}

extern "C" CUDF_C_EXPORT cudfError_t cudfResourcesCreate(cudfResources_t* res)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(res != nullptr, "output cudfResources_t pointer cannot be null");
    auto res_ptr = new CudfResources{};
    *res         = reinterpret_cast<uintptr_t>(res_ptr);
  });
}

extern "C" CUDF_C_EXPORT cudfError_t cudfResourcesDestroy(cudfResources_t res)
{
  return cudf::c::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<CudfResources*>(res);
    delete res_ptr;
  });
}

extern "C" CUDF_C_EXPORT cudfError_t cudfStreamSet(cudfResources_t res, cudaStream_t stream)
{
  return cudf::c::translate_exceptions([=] { get_resources(res)->stream = stream; });
}

extern "C" CUDF_C_EXPORT cudfError_t cudfStreamGet(cudfResources_t res, cudaStream_t* stream)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(stream != nullptr, "output cudaStream_t pointer cannot be null");
    *stream = get_resources(res)->stream;
  });
}

extern "C" CUDF_C_EXPORT cudfError_t cudfStreamSync(cudfResources_t res)
{
  return cudf::c::translate_exceptions(
    [=] { CUDF_CUDA_TRY(cudaStreamSynchronize(get_resources(res)->stream)); });
}

extern "C" CUDF_C_EXPORT cudfError_t cudfVersionGet(uint16_t* major,
                                                    uint16_t* minor,
                                                    uint16_t* patch)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(major != nullptr, "major version output pointer cannot be null");
    CUDF_EXPECTS(minor != nullptr, "minor version output pointer cannot be null");
    CUDF_EXPECTS(patch != nullptr, "patch version output pointer cannot be null");
    *major = CUDF_VERSION_MAJOR;
    *minor = CUDF_VERSION_MINOR;
    *patch = CUDF_VERSION_PATCH;
  });
}
