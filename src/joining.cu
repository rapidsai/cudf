/*

Uses code from https://github.com/moderngpu/moderngpu which has the following license:

> Copyright (c) 2016, Sean Baxter
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
>    list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> The views and conclusions contained in the software and documentation are those
> of the authors and should not be interpreted as representing official policies,
> either expressed or implied, of the FreeBSD Project.
*/


#include <gdf/gdf.h>
#include <gdf/errorutils.h>


#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>


#include <memory>

namespace {

using namespace mgpu;

template<typename launch_arg_t = empty_t,
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count,
  comp_t comp, context_t& context) {

  // Compute lower and upper bounds of a into b.
  mem_t<int> lower(a_count, context);
  mem_t<int> upper(a_count, context);
  sorted_search<bounds_lower, launch_arg_t>(a, a_count, b, b_count,
    lower.data(), comp, context);
  sorted_search<bounds_upper, launch_arg_t>(a, a_count, b, b_count,
    upper.data(), comp, context);

  // Compute output ranges by scanning upper - lower. Retrieve the reduction
  // of the scan, which specifies the size of the output array to allocate.
  mem_t<int> scanned_sizes(a_count, context);
  const int* lower_data = lower.data();
  const int* upper_data = upper.data();

  mem_t<int> count(1, context);
  transform_scan<int>([=]MGPU_DEVICE(int index) {
    return upper_data[index] - lower_data[index];
  }, a_count, scanned_sizes.data(), plus_t<int>(), count.data(), context);

  // Allocate an int2 output array and use load-balancing search to compute
  // the join.
  int join_count = from_mem(count)[0];
  mem_t<int2> output(join_count, context);
  int2* output_data = output.data();

  // Use load-balancing search on the segmens. The output is a pair with
  // a_index = seg and b_index = lower_data[seg] + rank.
  //
  // **libgdf changes**
  //  - tuple<int> lower -> tuple<int, int> lower
  //    to workaround error with 1-tuple
  auto k = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int, int> lower) {
    output_data[index] = make_int2(seg, get<0>(lower) + rank);
  };
  // **libgdf changes**
  //  - make_tuple(lower_data) -> make_tuple(lower_data, lower_data)
  //    to workaround error with 1-tuple
  transform_lbs<launch_arg_t>(k, join_count, scanned_sizes.data(), a_count,
    make_tuple(lower_data, lower_data), context);

  return output;
}


struct join_result_base {
    virtual ~join_result_base() {}
    virtual void* data() = 0;
    virtual size_t size() = 0;
};

template <typename T>
struct join_result : public join_result_base {
    standard_context_t context;
    mem_t<T> result;

    join_result() : context(false) {}
    virtual void* data() {
        return result.data();
    }
    virtual size_t size() {
        return result.size();
    }
};

gdf_join_result_type* cffi_wrap(join_result_base *obj) {
    return reinterpret_cast<gdf_join_result_type*>(obj);
}

join_result_base* cffi_unwrap(gdf_join_result_type* hdl) {
    return reinterpret_cast<join_result_base*>(hdl);
}

} // end anony namespace

gdf_error gdf_join_result_free(gdf_join_result_type *result) {
    delete cffi_unwrap(result);
    CUDA_CHECK_LAST();
    return GDF_SUCCESS;
}

void* gdf_join_result_data(gdf_join_result_type *result) {
    return cffi_unwrap(result)->data();
}

size_t gdf_join_result_size(gdf_join_result_type *result) {
    return cffi_unwrap(result)->size();
}


// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
#define MAX_JOIN_SIZE (0xffffffffu)


#define DEF_INNER_JOIN(Fn, T)                                               \
gdf_error gdf_inner_join_##Fn(gdf_column *leftcol, gdf_column *rightcol,    \
                              gdf_join_result_type **out_result) {          \
    using namespace mgpu;                                                   \
    if ( leftcol->dtype != rightcol->dtype) return GDF_UNSUPPORTED_DTYPE;   \
    if ( leftcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;   \
    if ( rightcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;  \
    std::unique_ptr<join_result<int2> > result_ptr(new join_result<int2>);  \
    result_ptr->result = inner_join((T*)leftcol->data, leftcol->size,       \
                                    (T*)rightcol->data, rightcol->size,     \
                                    less_t<T>(), result_ptr->context);      \
    CUDA_CHECK_LAST();                                                      \
    *out_result = cffi_wrap(result_ptr.release());                          \
    return GDF_SUCCESS;                                                     \
}


DEF_INNER_JOIN(i32, int32_t)
DEF_INNER_JOIN(i64, int64_t)
DEF_INNER_JOIN(f32, float)
DEF_INNER_JOIN(f64, double)


gdf_error gdf_inner_join_generic(gdf_column *leftcol, gdf_column * rightcol,
                                 gdf_join_result_type **out_result)
{
    switch ( leftcol->dtype ){
    case GDF_INT32:
        return gdf_inner_join_i32(leftcol, rightcol, out_result);
    case GDF_INT64:
        return gdf_inner_join_i64(leftcol, rightcol, out_result);
    case GDF_FLOAT32:
        return gdf_inner_join_f32(leftcol, rightcol, out_result);
    case GDF_FLOAT64:
        return gdf_inner_join_f64(leftcol, rightcol, out_result);
    default:
        return GDF_UNSUPPORTED_DTYPE;
    }
}
