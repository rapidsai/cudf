/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/text/tokenize.hpp>
#include <cudf/utilities/error.hpp>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/transform.h>

namespace cudf
{
namespace nvtext
{
namespace detail
{
namespace
{

// common pattern for token_count functions
template<typename TokenCounter>
std::unique_ptr<column> token_count_fn( size_type strings_count, TokenCounter tcfn,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream )
{
    // create output column
    auto token_counts = make_numeric_column( data_type{INT32}, strings_count, UNALLOCATED, stream, mr);
    auto d_token_counts = token_counts->mutable_view().data<int32_t>();
    // add the counts to the column
    thrust::transform( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_token_counts, tcfn );
    return token_counts;
}

// common pattern for tokenize functions
template<typename Tokenizer>
std::unique_ptr<column> tokenize_fn( size_type strings_count, Tokenizer ttfn,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    // get the number of tokens in each string
    auto token_counts = token_count_fn( strings_count, ttfn, mr, stream );
    auto d_token_counts = token_counts->view();
    // create token-index offsets from the counts
    rmm::device_vector<int32_t> token_offsets(strings_count+1);
    thrust::inclusive_scan( execpol->on(stream),
                            d_token_counts.template begin<int32_t>(),
                            d_token_counts.template end<int32_t>(),
                            token_offsets.begin()+1 );
    cudaMemsetAsync( token_offsets.data().get(), 0, sizeof(int32_t), stream );
    auto total_tokens = token_offsets.back();
    // build a list of pointers to each token
    rmm::device_vector<string_index_pair> tokens(total_tokens);
    // now go get the tokens
    ttfn.d_offsets = token_offsets.data().get();
    ttfn.d_tokens = tokens.data().get();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count, ttfn );
    // create the strings column using the tokens pointers
    return make_strings_column(tokens,stream,mr);
}

} // namespace

// detail APIs

// zero or more character tokenizer
std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  string_scalar const& delimiter = string_scalar(""),
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                  cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    string_view d_delimiter( delimiter.data(), delimiter.size() );
    auto strings_column = column_device_view::create(strings.parent(),stream);
    return tokenize_fn( strings.size(), tokenator_fn{*strings_column,d_delimiter}, mr, stream );
}

// zero or more character token counter
std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     string_scalar const& delimiter = string_scalar(""),
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    string_view d_delimiter( delimiter.data(), delimiter.size() );
    auto strings_column = column_device_view::create(strings.parent(),stream);
    return token_count_fn( strings.size(), tokenator_fn{*strings_column,d_delimiter}, mr, stream );
}

// one or more string delimiter tokenizer
std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  strings_column_view const& delimiters,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                  cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiters.size()>0, "Parameter delimiters must not be empty");
    CUDF_EXPECTS( !delimiters.has_nulls(), "Parameter delimiters must not have nulls");
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto delimiters_column = column_device_view::create(delimiters.parent(),stream);
    return tokenize_fn( strings.size(),
                        multi_delimiter_tokenizer_fn{*strings_column,
                                                     delimiters_column->begin<string_view>(),
                                                     delimiters_column->end<string_view>()},
                        mr, stream );
}

// one or more string delimiter token counter
std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     strings_column_view const& delimiters,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiters.size()>0, "Parameter delimiters must not be empty");
    CUDF_EXPECTS( !delimiters.has_nulls(), "Parameter delimiters must not have nulls");
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto delimiters_column = column_device_view::create(delimiters.parent(),stream);
    return token_count_fn( strings.size(),
                           multi_delimiter_tokenizer_fn{*strings_column,
                                                        delimiters_column->begin<string_view>(),
                                                        delimiters_column->end<string_view>()},
                           mr, stream );
}

} // namespace detail

// external APIs

std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  string_scalar const& delimiter,
                                  rmm::mr::device_memory_resource* mr )
{
    return detail::tokenize( strings, delimiter, mr );
}

std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  strings_column_view const& delimiters,
                                  rmm::mr::device_memory_resource* mr )
{
    return detail::tokenize( strings, delimiters, mr );
}

std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     string_scalar const& delimiter,
                                     rmm::mr::device_memory_resource* mr)
{
    return detail::token_count( strings, delimiter, mr );
}

std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     strings_column_view const& delimiters,
                                     rmm::mr::device_memory_resource* mr)
{
    return detail::token_count( strings, delimiters, mr );
}

} // namespace nvtext
} // namespace cudf
