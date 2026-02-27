/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/column_view.cuh>
#include <cudf/jit/lto/thunk.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

template <typename T>
__device__ T const* column_device_view::head() const
{
  return lower(lift(this)->head<lifted_type_of<T>>());
}

__device__ size_type column_device_view::size() const { return lift(this)->size(); }

__device__ bool column_device_view::nullable() const { return lift(this)->nullable(); }

__device__ bitmask_type const* column_device_view::null_mask() const
{
  return lift(this)->null_mask();
}

__device__ size_type column_device_view::offset() const { return lift(this)->offset(); }

__device__ bool column_device_view::is_valid(size_type index) const
{
  return lift(this)->is_valid(index);
}

__device__ bool column_device_view::is_valid_nocheck(size_type index) const
{
  return lift(this)->is_valid_nocheck(index);
}

__device__ bool column_device_view::is_null(size_type index) const
{
  return lift(this)->is_null(index);
}

__device__ bool column_device_view::is_null_nocheck(size_type index) const
{
  return lift(this)->is_null_nocheck(index);
}

__device__ bitmask_type column_device_view::get_mask_word(size_type index) const
{
  return lift(this)->get_mask_word(index);
}

template <typename T>
__device__ T column_device_view::element(size_type index) const
{
  auto ret = lift(this)->element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

template <typename T>
__device__ optional<T> column_device_view::nullable_element(size_type index) const
{
  auto ret = lift(this)->nullable_element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

__device__ size_type column_device_view::num_child_columns() const
{
  return lift(this)->num_child_columns();
}

template <typename T>
__device__ T* mutable_column_device_view::head() const
{
  return lower(lift(this)->head<lifted_type_of<T>>());
}

__device__ size_type mutable_column_device_view::size() const { return lift(this)->size(); }

__device__ bool mutable_column_device_view::nullable() const { return lift(this)->nullable(); }

__device__ bitmask_type* mutable_column_device_view::null_mask() const
{
  return lift(this)->null_mask();
}

__device__ size_type mutable_column_device_view::offset() const { return lift(this)->offset(); }

__device__ bool mutable_column_device_view::is_valid(size_type index) const
{
  return lift(this)->is_valid(index);
}

__device__ bool mutable_column_device_view::is_valid_nocheck(size_type index) const
{
  return lift(this)->is_valid_nocheck(index);
}

__device__ bool mutable_column_device_view::is_null(size_type index) const
{
  return lift(this)->is_null(index);
}

__device__ bool mutable_column_device_view::is_null_nocheck(size_type index) const
{
  return lift(this)->is_null_nocheck(index);
}

__device__ bitmask_type mutable_column_device_view::get_mask_word(size_type index) const
{
  return lift(this)->get_mask_word(index);
}

template <typename T>
__device__ T mutable_column_device_view::element(size_type index) const
{
  auto ret = lift(this)->element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

template <typename T>
__device__ optional<T> mutable_column_device_view::nullable_element(size_type index) const
{
  auto ret = lift(this)->nullable_element<lifted_type_of<T>>(index);
  return *lower(&ret);
}

template <typename T>
__device__ void mutable_column_device_view::assign(size_type index, T value) const
{
  lift(this)->assign<lifted_type_of<T>>(index, *lift(&value));
}

/// Explicit template instantiations

#define DO_IT(T) template __device__ T const* column_device_view::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) template __device__ T column_device_view::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) \
  template __device__ optional<T> column_device_view::nullable_element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) template __device__ T* mutable_column_device_view::head<T>() const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

#define DO_IT(T) template __device__ T mutable_column_device_view::element<T>(size_type idx) const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T)                                                                                 \
  template __device__ optional<T> mutable_column_device_view::nullable_element<T>(size_type idx) \
    const;

FOREACH_CUDF_LTO_COLUMN_ELEMENT_TYPE

#undef DO_IT

#define DO_IT(T) \
  template __device__ void mutable_column_device_view::assign<T>(size_type idx, T value) const;

FOREACH_CUDF_LTO_COLUMN_HEAD_TYPE

#undef DO_IT

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
