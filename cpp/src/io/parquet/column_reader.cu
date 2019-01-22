/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#include <arrow/util/bit-util.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "column_reader.h"
#include "dictionary_decoder.cuh"
#include "plain_decoder.cuh"

#include "util/bit_util.cuh"

namespace gdf {
namespace parquet {

template <class DataType, class DecoderType>
static inline void
_ConfigureDictionary(
  const ::parquet::Page *                                page,
  std::unordered_map<int, std::shared_ptr<DecoderType>> &decoders,
  const ::parquet::ColumnDescriptor *const               column_descriptor,
  ::arrow::MemoryPool *const                             pool,
  DecoderType **                                         out_decoder) {
    const ::parquet::DictionaryPage *dictionary_page =
      static_cast<const ::parquet::DictionaryPage *>(page);

    int encoding = static_cast<int>(dictionary_page->encoding());
    if (dictionary_page->encoding() == ::parquet::Encoding::PLAIN_DICTIONARY
        || dictionary_page->encoding() == ::parquet::Encoding::PLAIN) {
        encoding = static_cast<int>(::parquet::Encoding::RLE_DICTIONARY);
    }

    auto it = decoders.find(encoding);
    if (it != decoders.end()) {
        throw ::parquet::ParquetException(
          "Column cannot have more than one dictionary.");
    }

    if (dictionary_page->encoding() == ::parquet::Encoding::PLAIN_DICTIONARY
        || dictionary_page->encoding() == ::parquet::Encoding::PLAIN) {
        internal::PlainDecoder<DataType> dictionary(column_descriptor);
        dictionary.SetData(
          dictionary_page->num_values(), page->data(), page->size());

        auto decoder = std::make_shared<
          internal::DictionaryDecoder<DataType,
                                      gdf::arrow::internal::RleDecoder>>(
          column_descriptor, pool);
        decoder->SetDict(&dictionary);
        decoders[encoding] = decoder;
    } else {
        ::parquet::ParquetException::NYI(
          "only plain dictionary encoding has been implemented");
    }

    *out_decoder = decoders[encoding].get();
}

static inline bool
_IsDictionaryIndexEncoding(const ::parquet::Encoding::type &e) {
    return e == ::parquet::Encoding::RLE_DICTIONARY
           || e == ::parquet::Encoding::PLAIN_DICTIONARY;
}

template <class DecoderType, class T>
static inline std::int64_t
_ReadValues(DecoderType *decoder, std::int64_t batch_size, T *out) {
    std::int64_t num_decoded =
      decoder->Decode(out, static_cast<int>(batch_size));
    return num_decoded;
}

template <class DataType>
bool
ColumnReader<DataType>::HasNext() {
    if (num_buffered_values_ == 0
        || num_decoded_values_ == num_buffered_values_) {
        if (!ReadNewPage() || num_buffered_values_ == 0) { return false; }
    }
    return true;
}

template <class DataType>
bool
ColumnReader<DataType>::ReadNewPage() {
    const std::uint8_t *buffer;

    for (;;) {
        current_page_ = pager_->NextPage();
        if (!current_page_) { return false; }

        if (current_page_->type() == ::parquet::PageType::DICTIONARY_PAGE) {
            _ConfigureDictionary<DataType>(
              current_page_.get(), decoders_, descr_, pool_, &current_decoder_);
            continue;
        } else if (current_page_->type() == ::parquet::PageType::DATA_PAGE) {
            const ::parquet::DataPage *page =
              static_cast<const ::parquet::DataPage *>(current_page_.get());

            num_buffered_values_ = page->num_values();
            num_decoded_values_  = 0;
            buffer               = page->data();

            std::int64_t data_size = page->size();

            if (descr_->max_repetition_level() > 0) {
                std::int64_t rep_levels_bytes =
                  repetition_level_decoder_.SetData(
                    page->repetition_level_encoding(),
                    descr_->max_repetition_level(),
                    static_cast<int>(num_buffered_values_),
                    buffer);
                buffer += rep_levels_bytes;
                data_size -= rep_levels_bytes;
            }

            if (descr_->max_definition_level() > 0) {
                std::int64_t def_levels_bytes = def_level_decoder_.SetData(
                  page->definition_level_encoding(),
                  descr_->max_definition_level(),
                  static_cast<int>(num_buffered_values_),
                  buffer);
                buffer += def_levels_bytes;
                data_size -= def_levels_bytes;
            }

            ::parquet::Encoding::type encoding = page->encoding();

            if (_IsDictionaryIndexEncoding(encoding)) {
                encoding = ::parquet::Encoding::RLE_DICTIONARY;
            }

            auto it = decoders_.find(static_cast<int>(encoding));
            if (it != decoders_.end()) {
                if (encoding == ::parquet::Encoding::RLE_DICTIONARY) {
                    DCHECK(current_decoder_->encoding()
                           == ::parquet::Encoding::RLE_DICTIONARY);
                }
                current_decoder_ = it->second.get();
            } else {
                switch (encoding) {
                case ::parquet::Encoding::PLAIN: {
                    std::shared_ptr<DecoderType> decoder(
                      new internal::PlainDecoder<DataType>(descr_));
                    decoders_[static_cast<int>(encoding)] = decoder;
                    current_decoder_                      = decoder.get();
                    break;
                }
                case ::parquet::Encoding::RLE_DICTIONARY:
                    throw ::parquet::ParquetException(
                      "Dictionary page must be before data page.");

                case ::parquet::Encoding::DELTA_BINARY_PACKED:
                case ::parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY:
                case ::parquet::Encoding::DELTA_BYTE_ARRAY:
                    ::parquet::ParquetException::NYI("Unsupported encoding");

                default:
                    throw ::parquet::ParquetException("Unknown encoding type.");
                }
            }
            current_decoder_->SetData(static_cast<int>(num_buffered_values_),
                                      buffer,
                                      static_cast<int>(data_size));
            return true;
        } else {
            continue;
        }
    }    
}

static inline bool
_HasSpacedValues(const ::parquet::ColumnDescriptor *descr) {
    if (descr->max_repetition_level() > 0) {
        return !descr->schema_node()->is_required();
    } else {
        const ::parquet::schema::Node *node = descr->schema_node().get();
        while (node) {
            if (node->is_optional()) { return true; }
            node = node->parent();
        }
        return false;
    }
}

struct is_equal {
    int16_t max_definition_level;

    is_equal(int16_t max_definition_level)
      : max_definition_level(max_definition_level) {}
    __host__ __device__ bool
             operator()(const int16_t &x) {
        return x == max_definition_level;
    }
};

// expands data vector that does not contain nulls into a representation that has indeterminate values where there should be nulls
// A vector of int work_space needs to be allocated to hold the map for the scatter operation. The workspace should be of size batch_size
template <typename T>
void
compact_to_sparse_for_nulls(T *            data_in,
                            T *            data_out,
                            const int16_t *definition_levels,
                            int16_t        max_definition_level,
                            int64_t        batch_size,
                            int *          work_space) {
    is_equal op(max_definition_level);
    auto     out_iter     = thrust::copy_if(thrust::device,
                                    thrust::counting_iterator<int>(0),
                                    thrust::counting_iterator<int>(batch_size),
                                    definition_levels,
                                    work_space,
                                    op);
    int      num_not_null = out_iter - work_space;
    thrust::scatter(
      thrust::device, data_in, data_in + num_not_null, work_space, data_out);
}

#define WARP_BYTE 4
#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF
constexpr unsigned int THREAD_BLOCK_SIZE{256};

template <typename Functor>
__global__ void
transform_valid_kernel(uint8_t *valid, const int64_t size, Functor is_valid) {
    size_t tid    = threadIdx.x;
    size_t blkid  = blockIdx.x;
    size_t blksz  = blockDim.x;
    size_t gridsz = gridDim.x;

    size_t step = blksz * gridsz;
    size_t i    = tid + blkid * blksz;

    while (i < size) {
        uint32_t bitmask = 0;
        uint32_t result  = is_valid(i);
        bitmask          = (-result << (i % WARP_SIZE));

#pragma unroll
        for (size_t offset = 16; offset > 0; offset /= 2) {
            bitmask += __shfl_down_sync(WARP_MASK, bitmask, offset);
        }

        if ((i % WARP_SIZE) == 0) {
            int index        = i / WARP_SIZE * WARP_BYTE;
            valid[index + 0] = 0xFF & bitmask;
            valid[index + 1] = 0xFF & (bitmask >> 8);
            valid[index + 2] = 0xFF & (bitmask >> 16);
            valid[index + 3] = 0xFF & (bitmask >> 24);
        }
        i += step;
    }
}

template <typename Functor>
__global__ void
transform_valid_kernel(uint8_t *     valid,
                       const int64_t size,
                       size_t        num_chars,
                       Functor       is_valid) {
    size_t tid    = threadIdx.x;
    size_t blkid  = blockIdx.x;
    size_t blksz  = blockDim.x;
    size_t gridsz = gridDim.x;

    size_t step = blksz * gridsz;
    size_t i    = tid + blkid * blksz;

    while (i < size) {
        uint32_t bitmask = 0;
        uint32_t result  = is_valid(i);
        bitmask          = (-result << (i % WARP_SIZE));

#pragma unroll
        for (size_t offset = 16; offset > 0; offset /= 2) {
            bitmask += __shfl_down_sync(WARP_MASK, bitmask, offset);
        }

        if ((i % WARP_SIZE) == 0) {
            int index = i / WARP_SIZE * WARP_BYTE;
            if (index + 0 < num_chars) valid[index + 0] = 0xFF & bitmask;
            if (index + 1 < num_chars) valid[index + 1] = 0xFF & (bitmask >> 8);
            if (index + 2 < num_chars)
                valid[index + 2] = 0xFF & (bitmask >> 16);
            if (index + 3 < num_chars)
                valid[index + 3] = 0xFF & (bitmask >> 24);
        }
        i += step;
    }
}

/// See #transform_valid
namespace {

inline gdf_size_type
gdf_get_num_chars_bitmask(gdf_size_type size) {
    return ((size + (GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE);
}

}  // namespace

template <typename Functor>
void
transform_valid(uint8_t *valid, const int64_t size, Functor is_valid) {
    const dim3 grid((size + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);
    const dim3 block(THREAD_BLOCK_SIZE, 1, 1);
    if (size % 32 == 0) {
        transform_valid_kernel<Functor><<<grid, block>>>(valid, size, is_valid);
    } else {
        size_t num_chars = gdf_get_num_chars_bitmask(size);
        transform_valid_kernel<Functor>
          <<<grid, block>>>(valid, size, num_chars, is_valid);
    }
}

struct TurnOnFunctor {
    __host__ __device__ uint32_t
                        operator()(size_t index) {
        return 0xFFFFFFFF;
    }
};

static inline void
_TurnBitOnForValids(std::int64_t       def_length,
                    std::uint8_t *     d_valid_ptr,
                    const std::int64_t valid_bits_offset) {
    if (valid_bits_offset % 8 == 0) {
        transform_valid(
          d_valid_ptr + valid_bits_offset / 8, def_length, TurnOnFunctor{});
    } else {
        size_t  left_bits_length  = valid_bits_offset % 8;
        size_t  rigth_bits_length = 8 - left_bits_length;
        uint8_t mask;
        cudaMemcpy(&mask,
                   d_valid_ptr + (valid_bits_offset / 8),
                   1,
                   cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < rigth_bits_length; i++) {
            mask |= gdf::util::byte_bitmask(i + left_bits_length);
        }
        cudaMemcpy(d_valid_ptr + valid_bits_offset / 8,
                   &mask,
                   sizeof(uint8_t),
                   cudaMemcpyHostToDevice);
        transform_valid((d_valid_ptr + valid_bits_offset / 8 + 1),
                        def_length,
                        TurnOnFunctor{});
    }
}

struct IsValidFunctor {
    const std::int16_t *d_def_levels;
    std::int16_t        max_definition_level;
    IsValidFunctor(const std::int16_t *d_def_levels,
                   std::int16_t        max_definition_level)
      : d_def_levels{d_def_levels}, max_definition_level{max_definition_level} {
    }
    __host__ __device__ uint32_t
                        operator()(size_t index) {
        return d_def_levels[index] == max_definition_level ? 0xFFFFFFFF
                                                           : 0x00000000;
    }
};

static inline void
_DefinitionLevelsToBitmap(const std::int16_t *d_def_levels,
                          std::int64_t        def_length,
                          const std::int16_t  max_definition_level,
                          std::int64_t *      values_read,
                          std::int64_t *      null_count,
                          std::uint8_t *      d_valid_ptr,
                          const std::int64_t  valid_bits_offset) {

    if (valid_bits_offset % 8 == 0) {
        transform_valid((d_valid_ptr + valid_bits_offset / 8),
                        def_length,
                        IsValidFunctor{d_def_levels, max_definition_level});
    } else {
        int     left_bits_length  = valid_bits_offset % 8;
        int     right_bits_length = 8 - left_bits_length;
        uint8_t mask;
        cudaMemcpy(&mask,
                   d_valid_ptr + (valid_bits_offset / 8),
                   1,
                   cudaMemcpyDeviceToHost);

        thrust::host_vector<int16_t> h_def_levels(right_bits_length);
        cudaMemcpy(h_def_levels.data(),
                   d_def_levels,
                   right_bits_length * sizeof(int16_t),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h_def_levels.size(); i++) {
            if (h_def_levels[i] == max_definition_level) {
                mask |= gdf::util::byte_bitmask(i + left_bits_length);
            } else {
                if (h_def_levels[i] < max_definition_level) {
                    mask &= gdf::util::flipped_bitmask(i + left_bits_length);
                }
            }
        }
        cudaMemcpy(d_valid_ptr + valid_bits_offset / 8,
                   &mask,
                   sizeof(uint8_t),
                   cudaMemcpyHostToDevice);
        transform_valid(d_valid_ptr + valid_bits_offset / 8 + 1,
                        def_length - right_bits_length,
                        IsValidFunctor{d_def_levels + right_bits_length,
                                       max_definition_level});
    }
    int not_null_count =
      thrust::count(thrust::device_pointer_cast(d_def_levels),
                    thrust::device_pointer_cast(d_def_levels) + def_length,
                    max_definition_level);
    *null_count  = def_length - not_null_count;
    *values_read = not_null_count;
}

template <class DecoderType, class T>
static inline std::int64_t
_ReadValuesSpaced(DecoderType * decoder,
                  std::int64_t  batch_size,
                  T *           out,
                  std::int64_t  null_count,
                  std::uint8_t *valid_bits,
                  std::int64_t  valid_bits_offset) {
    return decoder->DecodeSpaced(out,
                                 static_cast<int>(batch_size),
                                 static_cast<int>(null_count),
                                 valid_bits,
                                 valid_bits_offset);
}

template <typename DataType>
inline std::int64_t
ColumnReader<DataType>::ReadBatchSpaced(std::int64_t  batch_size,
                                        std::int16_t *definition_levels,
                                        std::int16_t *repetition_levels,
                                        T *           values,
                                        std::uint8_t *valid_bits,
                                        std::int64_t  valid_bits_offset,  //
                                        std::int64_t *levels_read,
                                        std::int64_t *values_read,
                                        std::int64_t *nulls_count) {
    if (!HasNext()) {
        *levels_read = 0;
        *values_read = 0;
        *nulls_count = 0;
        return 0;
    }

    std::int64_t total_values;

    batch_size =
      std::min(batch_size, num_buffered_values_ - num_decoded_values_);

    if (descr_->max_definition_level() > 0) {
        std::int64_t num_def_levels =
          ReadDefinitionLevels(batch_size, definition_levels);

        const bool has_spaced_values = _HasSpacedValues(descr_);

        std::int64_t null_count = 0;
        if (!has_spaced_values) {
            int result = thrust::count(
              thrust::device_pointer_cast(definition_levels),
              thrust::device_pointer_cast(definition_levels) + num_def_levels,
              descr_->max_definition_level());
            int values_to_read = result;

            total_values =
              _ReadValues(current_decoder_, values_to_read, values);
            _TurnBitOnForValids(total_values, valid_bits, valid_bits_offset);
            *values_read = total_values;
        } else {
            std::int16_t max_definition_level = descr_->max_definition_level();
            std::int16_t max_repetition_level = descr_->max_repetition_level();

            _DefinitionLevelsToBitmap(definition_levels,
                                      num_def_levels,
                                      max_definition_level,
                                      values_read,
                                      &null_count,
                                      valid_bits,
                                      valid_bits_offset);

            total_values = _ReadValues(current_decoder_, *values_read, values);
            total_values = num_def_levels;

            if (total_values != *values_read) {
                thrust::device_vector<int> work_space_vector(total_values);
                int *                      work_space =
                  thrust::raw_pointer_cast(work_space_vector.data());
                thrust::device_vector<T> d_values_in(values,
                                                     values + total_values);
                compact_to_sparse_for_nulls(
                  thrust::raw_pointer_cast(d_values_in.data()),
                  values,
                  definition_levels,
                  max_definition_level,
                  total_values,
                  work_space);
            }
        }
        *levels_read = num_def_levels;
        *nulls_count = null_count;
    } else {
        total_values = _ReadValues(current_decoder_, batch_size, values);
        _TurnBitOnForValids(total_values, valid_bits, valid_bits_offset);
        *nulls_count = 0;
        *levels_read = total_values;
    }

    ConsumeBufferedValues(*levels_read);

    return total_values;
}

template <class DataType>
inline std::int64_t
ColumnReader<DataType>::ReadBatch(std::int64_t  batch_size,
                                  std::int16_t *def_levels,
                                  std::int16_t *rep_levels,
                                  T *           values,
                                  std::int64_t *values_read) {
    // assert(rep_levels == nullptr);
    if (!HasNext()) {
        *values_read = 0;
        return 0;
    }
    batch_size =
      std::min(batch_size, num_buffered_values_ - num_decoded_values_);

    std::int64_t num_def_levels = 0;

    std::int64_t values_to_read = 0;

    if (descr_->max_definition_level() > 0 && def_levels) {
        num_def_levels = ReadDefinitionLevels(batch_size, def_levels);
        int result     = thrust::count(thrust::device_pointer_cast(def_levels),
                                   thrust::device_pointer_cast(def_levels)
                                     + num_def_levels,
                                   descr_->max_definition_level());
        values_to_read = result;
    } else {
        values_to_read = batch_size;
    }

    *values_read = _ReadValues(current_decoder_, values_to_read, values);
    std::int64_t total_values = std::max(num_def_levels, *values_read);
    ConsumeBufferedValues(total_values);

    return total_values;
}

template <class DataType>
struct ParquetTraits {};

#define TYPE_TRAITS_FACTORY(ParquetType, GdfDType)                             \
    template <>                                                                \
    struct ParquetTraits<ParquetType> {                                        \
        static constexpr gdf_dtype gdfDType = GdfDType;                        \
    }

TYPE_TRAITS_FACTORY(::parquet::BooleanType, GDF_INT8);
TYPE_TRAITS_FACTORY(::parquet::Int32Type, GDF_INT32);
TYPE_TRAITS_FACTORY(::parquet::Int64Type, GDF_INT64);
TYPE_TRAITS_FACTORY(::parquet::FloatType, GDF_FLOAT32);
TYPE_TRAITS_FACTORY(::parquet::DoubleType, GDF_FLOAT64);

#undef TYPE_TRAITS_FACTORY

template <class DataType>
std::size_t
ColumnReader<DataType>::ToGdfColumn(const gdf_column &   column,
                                    const std::ptrdiff_t offset,
                                    std::uint8_t &       first_valid_byte,
                                    std::uint8_t &       last_valid_byte) {

    if (!HasNext()) { return 0; }
    std::int64_t values_to_read = num_buffered_values_ - num_decoded_values_;

    thrust::device_vector<int16_t> d_def_levels(
      values_to_read);  //this size is work group size
    std::int16_t *d_definition_levels =
      thrust::raw_pointer_cast(d_def_levels.data());

    std::size_t rows_read_total =
      ToGdfColumn(column, offset, d_definition_levels);

    std::int16_t max_definition_level = descr_->max_definition_level();

    if (offset > 0
        && offset % 8 != 0) {  // need to figure out the first_valid_byte
        first_valid_byte = 0;

        int left_bits_length  = offset % 8;
        int right_bits_length = 8 - left_bits_length;

        thrust::host_vector<int16_t> h_def_levels(right_bits_length);
        cudaMemcpy(h_def_levels.data(),
                   d_definition_levels,
                   right_bits_length * sizeof(int16_t),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h_def_levels.size(); i++) {
            if (h_def_levels[i] == max_definition_level) {
                first_valid_byte |=
                  gdf::util::byte_bitmask(i + left_bits_length);
            } else {
                if (h_def_levels[i] < max_definition_level) {
                    first_valid_byte &=
                      gdf::util::flipped_bitmask(i + left_bits_length);
                }
            }
        }
    }
    if ((offset + values_to_read) % 8
        != 0) {  // need to figure out the last_valid_byte
        last_valid_byte = 0;

        int left_bits_length  = (offset + values_to_read) % 8;
        
        thrust::host_vector<int16_t> h_def_levels(left_bits_length);
        cudaMemcpy(h_def_levels.data(),
                   d_definition_levels + values_to_read - left_bits_length,
                   left_bits_length * sizeof(int16_t),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h_def_levels.size(); i++) {
            if (h_def_levels[i] == max_definition_level) {
                last_valid_byte |= gdf::util::byte_bitmask(i);
            } else {
                if (h_def_levels[i] < max_definition_level) {
                    last_valid_byte &= gdf::util::flipped_bitmask(i);
                }
            }
        }
    }

    return rows_read_total;
}

template <class DataType>
std::size_t
ColumnReader<DataType>::ToGdfColumn(const gdf_column &   column,
                                    const std::ptrdiff_t offset) {
    if (!HasNext()) { return 0; }
    std::int64_t values_to_read = num_buffered_values_ - num_decoded_values_;

    thrust::device_vector<int16_t> d_def_levels(
      values_to_read);  //this size is work group size
    std::int16_t *d_definition_levels =
      thrust::raw_pointer_cast(d_def_levels.data());

    return ToGdfColumn(column, offset, d_definition_levels);
}

template <class DataType>
std::size_t
ColumnReader<DataType>::ToGdfColumn(const gdf_column &   column,
                                    const std::ptrdiff_t offset,
                                    std::int16_t *       d_definition_levels) {
    if (!HasNext()) { return 0; }
    using c_type = typename DataType::c_type;

    c_type *const values = static_cast<c_type *>(column.data) + offset;
    std::uint8_t *const d_valid_bits =
      static_cast<std::uint8_t *>(column.valid) + (offset / 8);

    static std::int64_t levels_read = 0;
    static std::int64_t values_read = 0;
    static std::int64_t nulls_count = 0;

    int64_t      rows_read_total = 0;
    std::int64_t values_to_read  = num_buffered_values_ - num_decoded_values_;

    do {
        values_to_read = num_buffered_values_ - num_decoded_values_;
        int64_t rows_read =
          ReadBatchSpaced(values_to_read,
                          d_definition_levels + rows_read_total,
                          nullptr,
                          static_cast<T *>(values + rows_read_total),
                          d_valid_bits,
                          rows_read_total + (offset % 8),
                          &levels_read,
                          &values_read,
                          &nulls_count);

        rows_read_total += rows_read;
    } while (this->HasNext());
    return static_cast<std::size_t>(rows_read_total);
}

template class ColumnReader<::parquet::BooleanType>;
template class ColumnReader<::parquet::Int32Type>;
template class ColumnReader<::parquet::Int64Type>;
template class ColumnReader<::parquet::FloatType>;
template class ColumnReader<::parquet::DoubleType>;

}  // namespace parquet
}  // namespace gdf
