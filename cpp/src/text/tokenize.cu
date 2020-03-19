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
#include <cudf/utilities/error.hpp>
#include <nvtext/tokenize.hpp>
#include <nvtext/detail/tokenize.hpp>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/transform.h>

namespace nvtext
{
namespace detail
{
namespace
{

// common pattern for token_count functions
template<typename TokenCounter>
std::unique_ptr<cudf::column> token_count_fn( cudf::size_type strings_count, TokenCounter tokenizer,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream )
{
    // create output column
    auto token_counts = cudf::make_numeric_column( cudf::data_type{cudf::INT32}, strings_count,
                                                   cudf::mask_state::UNALLOCATED, stream, mr);
    auto d_token_counts = token_counts->mutable_view().data<int32_t>();
    // add the counts to the column
    thrust::transform( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(strings_count),
        d_token_counts, tokenizer );
    return token_counts;
}

// common pattern for tokenize functions
template<typename Tokenizer>
std::unique_ptr<cudf::column> tokenize_fn( cudf::size_type strings_count, Tokenizer tokenizer,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    // get the number of tokens in each string
    auto const token_counts = token_count_fn( strings_count, tokenizer, mr, stream );
    auto d_token_counts = token_counts->view();
    // create token-index offsets from the counts
    rmm::device_vector<int32_t> token_offsets(strings_count+1);
    thrust::inclusive_scan( execpol->on(stream),
                            d_token_counts.template begin<int32_t>(),
                            d_token_counts.template end<int32_t>(),
                            token_offsets.begin()+1 );
    CUDA_TRY(cudaMemsetAsync( token_offsets.data().get(), 0, sizeof(int32_t), stream ));
    auto const total_tokens = token_offsets.back();
    // build a list of pointers to each token
    rmm::device_vector<string_index_pair> tokens(total_tokens);
    // now go get the tokens
    tokenizer.d_offsets = token_offsets.data().get();
    tokenizer.d_tokens = tokens.data().get();
    thrust::for_each_n(execpol->on(stream),
        thrust::make_counting_iterator<cudf::size_type>(0), strings_count, tokenizer );
    // create the strings column using the tokens pointers
    return cudf::make_strings_column(tokens,stream,mr);
}

} // namespace

// detail APIs

// zero or more character tokenizer
std::unique_ptr<cudf::column> tokenize( cudf::strings_column_view const& strings,
                                        cudf::string_scalar const& delimiter,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    cudf::string_view d_delimiter( delimiter.data(), delimiter.size() );
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    return tokenize_fn( strings.size(), strings_tokenizer{*strings_column,d_delimiter}, mr, stream );
}

// zero or more character token counter
std::unique_ptr<cudf::column> count_tokens( cudf::strings_column_view const& strings,
                                            cudf::string_scalar const& delimiter,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    cudf::string_view d_delimiter( delimiter.data(), delimiter.size() );
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    return token_count_fn( strings.size(), strings_tokenizer{*strings_column,d_delimiter}, mr, stream );
}

// one or more string delimiter tokenizer
std::unique_ptr<cudf::column> tokenize( cudf::strings_column_view const& strings,
                                        cudf::strings_column_view const& delimiters,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream )
{
    CUDF_EXPECTS( delimiters.size()>0, "Parameter delimiters must not be empty");
    CUDF_EXPECTS( !delimiters.has_nulls(), "Parameter delimiters must not have nulls");
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    auto delimiters_column = cudf::column_device_view::create(delimiters.parent(),stream);
    return tokenize_fn( strings.size(),
                        multi_delimiter_strings_tokenizer{*strings_column,
                            delimiters_column->begin<cudf::string_view>(),
                            delimiters_column->end<cudf::string_view>()},
                        mr, stream );
}

// one or more string delimiter token counter
std::unique_ptr<cudf::column> count_tokens( cudf::strings_column_view const& strings,
                                            cudf::strings_column_view const& delimiters,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream )
{
    CUDF_EXPECTS( delimiters.size()>0, "Parameter delimiters must not be empty");
    CUDF_EXPECTS( !delimiters.has_nulls(), "Parameter delimiters must not have nulls");
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    auto delimiters_column = cudf::column_device_view::create(delimiters.parent(),stream);
    return token_count_fn( strings.size(),
                           multi_delimiter_strings_tokenizer{*strings_column,
                                delimiters_column->begin<cudf::string_view>(),
                                delimiters_column->end<cudf::string_view>()},
                           mr, stream );
}

} // namespace detail

// external APIs

std::unique_ptr<cudf::column> tokenize( cudf::strings_column_view const& strings,
                                        cudf::string_scalar const& delimiter,
                                        rmm::mr::device_memory_resource* mr )
{
    return detail::tokenize( strings, delimiter, mr );
}

std::unique_ptr<cudf::column> tokenize( cudf::strings_column_view const& strings,
                                        cudf::strings_column_view const& delimiters,
                                        rmm::mr::device_memory_resource* mr )
{
    return detail::tokenize( strings, delimiters, mr );
}

std::unique_ptr<cudf::column> count_tokens( cudf::strings_column_view const& strings,
                                            cudf::string_scalar const& delimiter,
                                            rmm::mr::device_memory_resource* mr)
{
    return detail::count_tokens( strings, delimiter, mr );
}

std::unique_ptr<cudf::column> count_tokens( cudf::strings_column_view const& strings,
                                            cudf::strings_column_view const& delimiters,
                                            rmm::mr::device_memory_resource* mr)
{
    return detail::count_tokens( strings, delimiters, mr );
}

} // namespace nvtext
