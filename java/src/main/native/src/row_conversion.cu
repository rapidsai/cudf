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

#include <iostream>
#include <limits>

#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include "row_conversion.hpp"

namespace cudf {
namespace java {

/**
 * Copy a simple vector to device memory asynchronously. Be sure to read
 * the data on the same stream as is used to copy it.
 */
template <typename T>
std::unique_ptr<rmm::device_uvector<T>> copy_to_dev_async(
        const std::vector<T> & input,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr) {
    std::unique_ptr<rmm::device_uvector<T>> ret(new rmm::device_uvector<T>(
                input.size(), stream, mr));
    CUDA_TRY(cudaMemcpyAsync(ret->data(),
                input.data(),
                sizeof(T) * input.size(),
                cudaMemcpyHostToDevice,
                stream));
    return ret;
}

__global__
void copy_to_fixed_width_columns(
        const cudf::size_type num_rows,
        const cudf::size_type num_columns,
        const cudf::size_type row_size,
        const cudf::size_type* input_offset_in_row,
        const cudf::size_type* num_bytes,
        int8_t ** output_data,
        cudf::bitmask_type ** output_nm,
        const int8_t * input_data) {

    // We are going to copy the data in two passes.
    // The first pass copies a chunk of data into shared memory.
    // The second pass copies that chunk from shared memory out to the final location.
    
    // Because shared memory is limited we copy a subset of the rows at a time.
    // For simplicity we will refer to this as a row_group

    // In practice we have found writing more than 4 columns of data per thread
    // results in performance loss. As such we are using a 2 dimensional
    // kernel in terms of threads, but not in terms of blocks. Columns are
    // controlled by the y dimension (there is no y dimension in blocks). Rows
    // are controlled by the x dimension (there are multiple blocks in the x
    // dimension).

    cudf::size_type rows_per_group = blockDim.x;
    cudf::size_type row_group_start = blockIdx.x;
    cudf::size_type row_group_stride = gridDim.x;
    cudf::size_type row_group_end = (num_rows + rows_per_group - 1)/rows_per_group + 1;

    extern __shared__ int8_t shared_data[];

    // Because we are copying fixed width only data and we stride the rows
    // this thread will always start copying from shared data in the same place
    int8_t * row_tmp = &shared_data[row_size * threadIdx.x];
    int8_t * row_vld_tmp = &row_tmp[input_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

    for (cudf::size_type row_group_index = row_group_start;
            row_group_index < row_group_end;
            row_group_index += row_group_stride) {
        // Step 1: Copy the data into shared memory
        // We know row_size is always aligned with and a multiple of int64_t;
        int64_t * long_shared = reinterpret_cast<int64_t *>(shared_data);
        const int64_t * long_input = reinterpret_cast<int64_t const *>(input_data);

        cudf::size_type shared_output_index = threadIdx.x + (threadIdx.y * blockDim.x);
        cudf::size_type shared_output_stride = blockDim.x * blockDim.y;
        cudf::size_type row_index_end = ((row_group_index + 1) * rows_per_group);
        if (row_index_end > num_rows) {
            row_index_end = num_rows;
        }
        cudf::size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
        cudf::size_type shared_length = row_size * num_rows_in_group;

        cudf::size_type shared_output_end = shared_length/sizeof(int64_t);

        cudf::size_type start_input_index = (row_size * row_group_index * rows_per_group)/sizeof(int64_t);

        for (cudf::size_type shared_index = shared_output_index; 
                shared_index < shared_output_end;
                shared_index += shared_output_stride) {
            long_shared[shared_index] = long_input[start_input_index + shared_index];
        }
        // Wait for all of the data to be in shared memory
        __syncthreads();

        // Step 2 copy the data back out

        // Within the row group there should be 1 thread for each row.  This is a
        // requirement for launching the kernel
        cudf::size_type row_index = (row_group_index * rows_per_group) + threadIdx.x;
        // But we might not use all of the threads if the number of rows does not go
        // evenly into the thread count. We don't want those threads to exit yet
        // because we may need them to copy data in for the next row group.
        uint32_t active_mask = __ballot_sync(0xffffffff, row_index < num_rows);
        if (row_index < num_rows) {
            cudf::size_type col_index_start = threadIdx.y;
            cudf::size_type col_index_stride = blockDim.y;
            for (cudf::size_type col_index = col_index_start;
                    col_index < num_columns;
                    col_index += col_index_stride) {

                cudf::size_type col_size = num_bytes[col_index];
                const int8_t * col_tmp = &(row_tmp[input_offset_in_row[col_index]]);
                int8_t * col_output = output_data[col_index];
                switch(col_size) {
                    case 1:
                    {
                        col_output[row_index] = *col_tmp;
                        break;
                    }
                    case 2:
                    {
                        int16_t * short_col_output = reinterpret_cast<int16_t *>(col_output);
                        short_col_output[row_index] = *reinterpret_cast<const int16_t *>(col_tmp);
                        break;
                    }
                    case 4:
                    {
                        int32_t * int_col_output = reinterpret_cast<int32_t *>(col_output);
                        int_col_output[row_index] = *reinterpret_cast<const int32_t *>(col_tmp);
                        break;
                    }
                    case 8:
                    {
                        int64_t * long_col_output = reinterpret_cast<int64_t *>(col_output);
                        long_col_output[row_index] = *reinterpret_cast<const int64_t *>(col_tmp);
                        break;
                    }
                    default:
                    {
                        cudf::size_type output_offset = col_size * row_index;
                        // TODO this should just not be supported for fixed width columns, but just in case...
                        for (cudf::size_type b = 0; b < col_size; b++) {
                            col_output[b + output_offset] = col_tmp[b];
                        }
                        break;
                    }
                }

                cudf::bitmask_type * nm = output_nm[col_index];
                int8_t * valid_byte = &row_vld_tmp[col_index/8];
                cudf::size_type byte_bit_offset = col_index % 8;
                int predicate = *valid_byte & (1 << byte_bit_offset);
                uint32_t bitmask = __ballot_sync(active_mask, predicate);
                if (row_index % 32 == 0) {
                    nm[word_index(row_index)] = bitmask;
                }
            } // end column loop
        } // end row copy
        // wait for the row_group to be totally copied before starting on the next row group
        __syncthreads();
    }
}


__global__
void copy_from_fixed_width_columns(
        const cudf::size_type start_row,
        const cudf::size_type num_rows,
        const cudf::size_type num_columns,
        const cudf::size_type row_size,
        const cudf::size_type* output_offset_in_row,
        const cudf::size_type* num_bytes,
        const int8_t ** input_data,
        const cudf::bitmask_type ** input_nm,
        int8_t * output_data) {
    // We are going to copy the data in two passes.
    // The first pass copies a chunk of data into shared memory.
    // The second pass copies that chunk from shared memory out to the final location.
    
    // Because shared memory is limited we copy a subset of the rows at a time.
    // We do not support copying a subset of the columns in a row yet, so we don't
    // currently support a row that is wider than shared memory.
    // For simplicity we will refer to this as a row_group

    // In practice we have found reading more than 4 columns of data per thread
    // results in performance loss. As such we are using a 2 dimensional
    // kernel in terms of threads, but not in terms of blocks. Columns are
    // controlled by the y dimension (there is no y dimension in blocks). Rows
    // are controlled by the x dimension (there are multiple blocks in the x
    // dimension).

    cudf::size_type rows_per_group = blockDim.x;
    cudf::size_type row_group_start = blockIdx.x;
    cudf::size_type row_group_stride = gridDim.x;
    cudf::size_type row_group_end = (num_rows + rows_per_group - 1)/rows_per_group + 1;

    extern __shared__ int8_t shared_data[];


    // Because we are copying fixed width only data and we stride the rows
    // this thread will always start copying to shared data in the same place
    int8_t * row_tmp = &shared_data[row_size * threadIdx.x];
    int8_t * row_vld_tmp = &row_tmp[output_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

    for (cudf::size_type row_group_index = row_group_start;
            row_group_index < row_group_end;
            row_group_index += row_group_stride) {

        // Within the row group there should be 1 thread for each row.  This is a
        // requirement for launching the kernel
        cudf::size_type row_index = start_row + (row_group_index * rows_per_group) + threadIdx.x;
        // But we might not use all of the threads if the number of rows does not go
        // evenly into the thread count. We don't want those threads to exit yet
        // because we may need them to copy data back out.
        if (row_index < (start_row + num_rows)) {
            cudf::size_type col_index_start = threadIdx.y;
            cudf::size_type col_index_stride = blockDim.y;
            for (cudf::size_type col_index = col_index_start;
                    col_index < num_columns;
                    col_index += col_index_stride) {

                cudf::size_type col_size = num_bytes[col_index];
                int8_t * col_tmp = &(row_tmp[output_offset_in_row[col_index]]);
                const int8_t * col_input = input_data[col_index];
                switch(col_size) {
                    case 1:
                    {
                        *col_tmp = col_input[row_index];
                        break;
                    }
                    case 2:
                    {
                        const int16_t * short_col_input = reinterpret_cast<const int16_t *>(col_input);
                        *reinterpret_cast<int16_t *>(col_tmp) = short_col_input[row_index];
                        break;
                    }
                    case 4:
                    {
                        const int32_t * int_col_input = reinterpret_cast<const int32_t *>(col_input);
                        *reinterpret_cast<int32_t *>(col_tmp) = int_col_input[row_index];
                        break;
                    }
                    case 8:
                    {
                        const int64_t * long_col_input = reinterpret_cast<const int64_t *>(col_input);
                        *reinterpret_cast<int64_t *>(col_tmp) = long_col_input[row_index];
                        break;
                    }
                    default:
                    {
                        cudf::size_type input_offset = col_size * row_index;
                        // TODO this should just not be supported for fixed width columns, but just in case...
                        for (cudf::size_type b = 0; b < col_size; b++) {
                            col_tmp[b] = col_input[b + input_offset];
                        }
                        break;
                    }
                }
                // atomicOr only works on 32 bit or 64 bit  aligned values, and not byte aligned
                // so we have to rewrite the addresses to make sure that it is 4 byte aligned
                int8_t * valid_byte = &row_vld_tmp[col_index/8];
                cudf::size_type byte_bit_offset = col_index % 8;
                uint64_t fixup_bytes = reinterpret_cast<uint64_t>(valid_byte) % 4;
                int32_t * valid_int = reinterpret_cast<int32_t *>(valid_byte - fixup_bytes);
                cudf::size_type int_bit_offset = byte_bit_offset + (fixup_bytes * 8);
                // Now copy validity for the column
                if (input_nm[col_index]) {
                  if (bit_is_set(input_nm[col_index], row_index)) {
                    atomicOr_block(valid_int, 1 << int_bit_offset);
                  } else {
                    atomicAnd_block(valid_int, ~(1 << int_bit_offset));
                  }
                } else {
                  // It is valid so just set the bit
                  atomicOr_block(valid_int, 1 << int_bit_offset);
                }
            } // end column loop
        } // end row copy
        // wait for the row_group to be totally copied into shared memory
        __syncthreads();

        // Step 2: Copy the data back out
        // We know row_size is always aligned with and a multiple of int64_t;
        int64_t * long_shared = reinterpret_cast<int64_t *>(shared_data);
        int64_t * long_output = reinterpret_cast<int64_t *>(output_data);

        cudf::size_type shared_input_index = threadIdx.x + (threadIdx.y * blockDim.x);
        cudf::size_type shared_input_stride = blockDim.x * blockDim.y;
        cudf::size_type row_index_end = ((row_group_index + 1) * rows_per_group);
        if (row_index_end > num_rows) {
            row_index_end = num_rows;
        }
        cudf::size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
        cudf::size_type shared_length = row_size * num_rows_in_group;

        cudf::size_type shared_input_end = shared_length/sizeof(int64_t);

        cudf::size_type start_output_index = (row_size * row_group_index * rows_per_group)/sizeof(int64_t);

        for (cudf::size_type shared_index = shared_input_index; 
                shared_index < shared_input_end;
                shared_index += shared_input_stride) {
            long_output[start_output_index + shared_index] = long_shared[shared_index];
        }
        __syncthreads();
        // Go for the next round
    }
}

/**
 * Calculate the dimensions of the kernel for fixed width only columns.
 * @param [in] num_columns the number of columns being copied.
 * @param [in] num_rows the number of rows being copied.
 * @param [in] size_per_row the size each row takes up when padded.
 * @param [out] blocks the size of the blocks for the kernel
 * @param [out] threads the size of the threads for the kernel
 * @return the size in bytes of shared memory needed for each block.
 */
static int calc_fixed_width_kernel_dims(
    const cudf::size_type num_columns,
    const cudf::size_type num_rows,
    const cudf::size_type size_per_row,
    dim3 & blocks,
    dim3 & threads) {

    // We have found speed degrades when a thread handles more than 4 columns.
    // Each block is 2 dimensional. The y dimension indicates the columns.
    // We limit this to 32 threads in the y dimension so we can still
    // have at least 32 threads in the x dimension (1 warp) which should
    // result in better coalescing of memory operations. We also
    // want to guarantee that we are processing a multiple of 32 threads
    // in the x dimension because we use atomic operations at the block
    // level when writing validity data out to main memory, and that would
    // need to change if we split a word of validity data between blocks.
    int y_block_size = (num_columns + 3) / 4;
    if (y_block_size > 32) {
      y_block_size = 32;
    }
    int x_possible_block_size = 1024/y_block_size;
    // 48KB is the default setting for shared memory per block according to the cuda tutorials
    // If someone configures the GPU to only have 16 KB this might not work.
    int max_shared_size = 48 * 1024;
    int max_block_size = max_shared_size/size_per_row;
    // If we don't have enough shared memory there is no point in having more threads
    // per block that will just sit idle
    max_block_size = max_block_size > x_possible_block_size ? x_possible_block_size : max_block_size;
    // Make sure that the x dimension is a multiple of 32 this not only helps
    // coalesce memory access it also lets us do a ballot sync for validity to write
    // the data back out the warp level.  If x is a multiple of 32 then each thread in the y
    // dimension is associated with one or more warps, that should correspond to the validity
    // words directly.
    int block_size = (max_block_size / 32) * 32;
    CUDF_EXPECTS(block_size != 0, "Row size is too large to fit in shared memory");

    int num_blocks = (num_rows + block_size - 1) / block_size;
    if (num_blocks < 1) {
        num_blocks = 1;
    } else if (num_blocks > 10240) {
        // The maximum number of blocks supported in the x dimension is 2 ^ 31 - 1
        // but in practice haveing too many can cause some overhead that I don't totally
        // understand. Playing around with this haveing as little as 600 blocks appears
        // to be able to saturate memory on V100, so this is an order of magnitude higher
        // to try and future proof this a bit.
        num_blocks = 10240;
    }
    blocks.x = num_blocks;
    blocks.y = 1;
    blocks.z = 1;
    threads.x = block_size;
    threads.y = y_block_size;
    threads.z = 1;
    return size_per_row * block_size;
}

/**
 * When converting to rows it is possible that the size of the table was too big to fit
 * in a single column. This creates an output column for a subset of the rows in a table
 * going from start row and containing the next num_rows.  Most of the parameters passed
 * into this function are common between runs and should be calculated once.
 */
static std::unique_ptr<cudf::column> fixed_width_convert_to_rows(
        const cudf::size_type start_row,
        const cudf::size_type num_rows,
        const cudf::size_type num_columns,
        const cudf::size_type size_per_row,
        std::unique_ptr<rmm::device_uvector<cudf::size_type>> & column_start,
        std::unique_ptr<rmm::device_uvector<cudf::size_type>> & column_size,
        std::unique_ptr<rmm::device_uvector<const int8_t *>> & input_data,
        std::unique_ptr<rmm::device_uvector<const cudf::bitmask_type *>> & input_nm,
        const cudf::scalar & zero,
        const cudf::scalar & scalar_size_per_row,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
        ) {
    int64_t total_allocation = size_per_row * num_rows;
    // We made a mistake in the split somehow
    CUDF_EXPECTS(total_allocation < std::numeric_limits<int>::max(), "Table is too large to fit!");

    // Allocate and set the offsets row for the byte array
    std::unique_ptr<cudf::column> offsets = cudf::detail::sequence(
            num_rows + 1, zero, scalar_size_per_row);

    std::unique_ptr<cudf::column> data =
        cudf::make_numeric_column(
                cudf::data_type(cudf::type_id::INT8),
                static_cast<cudf::size_type>(total_allocation));

    dim3 blocks;
    dim3 threads;
    int shared_size = calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

    copy_from_fixed_width_columns<<<blocks, threads, shared_size, stream>>>(
            start_row,
            num_rows,
            num_columns,
            size_per_row,
            column_start->data(),
            column_size->data(),
            input_data->data(),
            input_nm->data(),
            data->mutable_view().data<int8_t>());

    return cudf::make_lists_column(num_rows,
                std::move(offsets),
                std::move(data),
                0,
                rmm::device_buffer{0, 0, mr});
}

static cudf::data_type get_data_type(const cudf::column_view & v) {
    return v.type();
}

static bool is_fixed_width(const cudf::data_type & t) {
    return cudf::is_fixed_width(t);
}

static inline int32_t align_offset(int32_t offset, std::size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

static inline bool are_all_fixed_width(std::vector<cudf::data_type> const & schema) {
    return std::all_of(schema.begin(), schema.end(), cudf::java::is_fixed_width);
}

/**
 * Given a set of fixed width columns, calculate how the data will be laid out in memory.
 * @param [in] schema the types of columns that need to be laid out.
 * @param [out] column_start the byte offset where each column starts in the row.
 * @param [out] column_size the size in bytes of the data for each columns in the row.
 * @return the size in bytes each row needs.
 */
static inline int32_t compute_fixed_width_layout(
        std::vector<cudf::data_type> const & schema,
        std::vector<cudf::size_type> & column_start,
        std::vector<cudf::size_type> & column_size) {
    // We guarantee that the start of each column is 64-bit aligned so anything can go
    // there, but to make the code simple we will still do an alignment for it.
    int32_t at_offset = 0;
    for (auto col = schema.begin(); col < schema.end(); col++) {
        cudf::size_type s = cudf::size_of(*col);
        column_size.emplace_back(s);
        std::size_t allocation_needed = s;
        std::size_t alignment_needed = allocation_needed; // They are the same for fixed width types
        at_offset = align_offset(at_offset, alignment_needed);
        column_start.emplace_back(at_offset);
        at_offset += allocation_needed;
    }

    // Now we need to add in space for validity
    // Eventually we can think about nullable vs not nullable, but for now we will just always add it in
    int32_t validity_bytes_needed = (schema.size() + 7)/8;
    // validity comes at the end and is byte aligned so we can pack more in.
    at_offset += validity_bytes_needed;
    // Now we need to pad the end so all rows are 64 bit aligned
    return align_offset(at_offset, 8); // 8 bytes (64 bits)
}

std::vector<std::unique_ptr<cudf::column>> convert_to_rows(
        cudf::table_view const& tbl,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr) {

    const cudf::size_type num_columns = tbl.num_columns();

    std::vector<cudf::data_type> schema;
    schema.resize(num_columns);
    std::transform(tbl.begin(), tbl.end(), schema.begin(), cudf::java::get_data_type);

    if (are_all_fixed_width(schema)) {
        std::vector<cudf::size_type> column_start;
        std::vector<cudf::size_type> column_size;

        int32_t size_per_row = compute_fixed_width_layout(schema, column_start, column_size);
        auto dev_column_start = copy_to_dev_async(column_start, stream, mr);
        auto dev_column_size = copy_to_dev_async(column_size, stream, mr);

        int32_t max_rows_per_batch = std::numeric_limits<int>::max() / size_per_row;
        // Make the number of rows per batch a multiple of 32 so we don't have to worry about
        // splitting validity at a specific row offset.  This might change in the future.
        max_rows_per_batch = (max_rows_per_batch/32) * 32;

        cudf::size_type num_rows = tbl.num_rows();

        // Get the pointers to the input columnar data ready
        std::vector<const int8_t *> input_data;
        std::vector<cudf::bitmask_type const *> input_nm;
        for (cudf::size_type column_number = 0; column_number < num_columns; column_number++) {
            cudf::column_view cv = tbl.column(column_number);
            input_data.emplace_back(cv.data<int8_t>());
            input_nm.emplace_back(cv.null_mask());
        }
        auto dev_input_data = copy_to_dev_async(input_data, stream, mr);
        auto dev_input_nm = copy_to_dev_async(input_nm, stream, mr);

        using ScalarType = cudf::scalar_type_t<cudf::size_type>;
        auto zero = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
        zero->set_valid(true);
        static_cast<ScalarType *>(zero.get())->set_value(0);

        auto step = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
        step->set_valid(true);
        static_cast<ScalarType *>(step.get())->set_value(static_cast<cudf::size_type>(size_per_row));

        std::vector<std::unique_ptr<cudf::column>> ret;
        for (cudf::size_type row_start = 0; row_start < num_rows; row_start += max_rows_per_batch) {
            cudf::size_type row_count = num_rows - row_start;
            row_count = row_count > max_rows_per_batch ? max_rows_per_batch : row_count;
            ret.emplace_back(fixed_width_convert_to_rows(
                        row_start,
                        row_count,
                        num_columns,
                        size_per_row,
                        dev_column_start,
                        dev_column_size,
                        dev_input_data,
                        dev_input_nm,
                        *zero,
                        *step,
                        stream,
                        mr));
        }

        return ret;
    } else {
        CUDF_FAIL("Only fixed width types are currently supported");
    }
}

std::unique_ptr<cudf::table> convert_from_rows(
        cudf::lists_column_view const& input,
        std::vector<cudf::data_type> const& schema,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr) {

    // verify that the types are what we expect
    cudf::column_view child = input.child();
    cudf::type_id list_type = child.type().id();
    CUDF_EXPECTS(list_type == cudf::type_id::INT8 || list_type == cudf::type_id::UINT8,
            "Only a list of bytes is supported as input");

    cudf::size_type num_columns = schema.size();

    if (are_all_fixed_width(schema)) {
        std::vector<cudf::size_type> column_start;
        std::vector<cudf::size_type> column_size;

        cudf::size_type num_rows = input.parent().size();
        int32_t size_per_row = compute_fixed_width_layout(schema, column_start, column_size);

        // Ideally we would check that the offsets are all the same, etc. but for now
        // this is probably fine
        CUDF_EXPECTS(size_per_row * num_rows == child.size(),
                "The layout of the data appears to be off");
        auto dev_column_start = copy_to_dev_async(column_start, stream, mr);
        auto dev_column_size = copy_to_dev_async(column_size, stream, mr);

        // Allocate the columns we are going to write into
        std::vector<std::unique_ptr<cudf::column>> output_columns;
        std::vector<int8_t *> output_data;
        std::vector<cudf::bitmask_type *> output_nm; 
        for (cudf::size_type i = 0; i < num_columns; i++) {
            auto column = cudf::make_fixed_width_column(schema[i],
                        num_rows,
                        cudf::mask_state::UNINITIALIZED,
                        stream,
                        mr);
            auto mut = column->mutable_view();
            output_data.emplace_back(mut.data<int8_t>());
            output_nm.emplace_back(mut.null_mask());
            output_columns.emplace_back(std::move(column));
        }

        auto dev_output_data = copy_to_dev_async(output_data, stream, mr);
        auto dev_output_nm = copy_to_dev_async(output_nm, stream, mr);

        dim3 blocks;
        dim3 threads;
        int shared_size = calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

        copy_to_fixed_width_columns<<<blocks, threads, shared_size, stream>>>(
                num_rows,
                num_columns,
                size_per_row,
                dev_column_start->data(),
                dev_column_size->data(),
                dev_output_data->data(),
                dev_output_nm->data(),
                child.data<int8_t>());

        return std::make_unique<cudf::table>(std::move(output_columns));
    } else {
        CUDF_FAIL("Only fixed width types are currently supported");
    }
}

} // namespace java
} // namespace cudf
