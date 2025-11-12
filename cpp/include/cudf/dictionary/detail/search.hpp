/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace dictionary {
namespace detail {

/**
 * @copydoc cudf::dictionary::get_index(dictionary_column_view const&,scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<scalar> get_index(dictionary_column_view const& dictionary,
                                  scalar const& key,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Get the index for a key if it were added to the given dictionary.
 *
 * The actual index is returned if the `key` is already part of the dictionary's key set.
 *
 * @code{.pseudo}
 * d1 = {["a","c","d"],[2,0,1,0]}
 * idx = get_insert_index(d1,"b")
 * idx is 1
 * @endcode{.pseudo}
 *
 * @throw cudf::logic_error if `key.type() != dictionary.keys().type()`
 *
 * @param dictionary The dictionary to search for the key.
 * @param key The value to search for in the dictionary keyset.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Numeric scalar index value of the key within the dictionary
 */
std::unique_ptr<scalar> get_insert_index(dictionary_column_view const& dictionary,
                                         scalar const& key,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace dictionary
}  // namespace CUDF_EXPORT cudf
