/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/operators.cuh>
#include <cudf/jit/lto/types.cuh>

#include <cuda/std/cmath>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

#define DEF_UNOP(name, type, expr) \
  __device__ void name(type* out, type const& a) { *out = expr; }

#define DEF_BINOP(name, type, expr) \
  __device__ void name(type* out, type const& a, type const& b) { *out = expr; }

DEF_UNOP(sin, f32, (cuda::std::sin(a)))
DEF_UNOP(sin, f64, (cuda::std::sin(a)))

DEF_UNOP(cos, f32, (cuda::std::cos(a)))
DEF_UNOP(cos, f64, (cuda::std::cos(a)))

DEF_UNOP(tan, f32, (cuda::std::tan(a)))
DEF_UNOP(tan, f64, (cuda::std::tan(a)))

DEF_UNOP(arcsin, f32, (cuda::std::asin(a)))
DEF_UNOP(arcsin, f64, (cuda::std::asin(a)))

DEF_UNOP(arccos, f32, (cuda::std::acos(a)))
DEF_UNOP(arccos, f64, (cuda::std::acos(a)))

DEF_UNOP(arctan, f32, (cuda::std::atan(a)))
DEF_UNOP(arctan, f64, (cuda::std::atan(a)))

DEF_UNOP(sinh, f32, (cuda::std::sinh(a)))
DEF_UNOP(sinh, f64, (cuda::std::sinh(a)))

DEF_UNOP(cosh, f32, (cuda::std::cosh(a)))
DEF_UNOP(cosh, f64, (cuda::std::cosh(a)))

DEF_UNOP(tanh, f32, (cuda::std::tanh(a)))
DEF_UNOP(tanh, f64, (cuda::std::tanh(a)))

DEF_UNOP(arcsinh, f32, (cuda::std::asinh(a)))
DEF_UNOP(arcsinh, f64, (cuda::std::asinh(a)))

DEF_UNOP(arccosh, f32, (cuda::std::acosh(a)))
DEF_UNOP(arccosh, f64, (cuda::std::acosh(a)))

DEF_UNOP(arctanh, f32, (cuda::std::atanh(a)))
DEF_UNOP(arctanh, f64, (cuda::std::atanh(a)))

DEF_UNOP(exp, f32, (cuda::std::exp(a)))
DEF_UNOP(exp, f64, (cuda::std::exp(a)))

DEF_UNOP(log, f32, (cuda::std::log(a)))
DEF_UNOP(log, f64, (cuda::std::log(a)))

DEF_UNOP(cbrt, f32, (cuda::std::cbrt(a)))
DEF_UNOP(cbrt, f64, (cuda::std::cbrt(a)))

DEF_UNOP(ceil, f32, (cuda::std::ceil(a)))
DEF_UNOP(ceil, f64, (cuda::std::ceil(a)))

DEF_UNOP(floor, f32, (cuda::std::floor(a)))
DEF_UNOP(floor, f64, (cuda::std::floor(a)))

DEF_UNOP(rint, f32, (cuda::std::rint(a)))
DEF_UNOP(rint, f64, (cuda::std::rint(a)))

DEF_BINOP(mod, f32, (cuda::std::fmod(a, b)))
DEF_BINOP(mod, f64, (cuda::std::fmod(a, b)))

DEF_BINOP(pymod, f32, (cuda::std::fmod(cuda::std::fmod(a, b) + b, b)))
DEF_BINOP(pymod, f64, (cuda::std::fmod(cuda::std::fmod(a, b) + b, b)))

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
