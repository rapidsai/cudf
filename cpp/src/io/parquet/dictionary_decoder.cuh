/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include "arrow/util/rle-encoding.h"
#include "decoder/arrow/rle_decoder.h"
#include <thrust/device_vector.h>

namespace parquet {
class ColumnDescriptor;
}

namespace gdf {
namespace parquet {
namespace internal {

template <typename Type, typename RleDecoder>
class DictionaryDecoder : public ::parquet::Decoder<Type> {
public:
    typedef typename Type::c_type T;

    explicit DictionaryDecoder(const ::parquet::ColumnDescriptor *descr,
                               ::arrow::MemoryPool *pool = nullptr)
      : ::parquet::Decoder<Type>(descr, ::parquet::Encoding::RLE_DICTIONARY),
        dictionary_(0) {}

    void
    SetDict(::parquet::Decoder<Type> *dictionary);

    void
    SetData(int num_values, const std::uint8_t *data, int len) override {
        num_values_ = num_values;
        if (len == 0) return;
        std::uint8_t bit_width = *data;
        ++data;
        --len;
        idx_decoder_ = RleDecoder(data, len, bit_width);
    }

    int
    Decode(T *buffer, int max_values) override {
        max_values         = std::min(max_values, num_values_);
        int decoded_values = idx_decoder_.GetBatchWithDict(
          thrust::raw_pointer_cast(dictionary_.data()),
          num_dictionary_values_,
          buffer,
          max_values);
        if (decoded_values != max_values) {
            ::parquet::ParquetException::EofException();
        }
        num_values_ -= max_values;
        return max_values;
    }

    int
    DecodeSpaced(T *                 buffer,
                 int                 num_values,
                 int                 null_count,
                 const std::uint8_t *valid_bits,
                 std::int64_t        valid_bits_offset) override {
        int decoded_values = idx_decoder_.GetBatchWithDictSpaced(
          thrust::raw_pointer_cast(dictionary_.data()),
          num_dictionary_values_,
          buffer,
          num_values,
          null_count,
          valid_bits,
          valid_bits_offset);
        if (decoded_values != num_values) {
            ::parquet::ParquetException::EofException();
        }
        return decoded_values;
    }

private:
    using ::parquet::Decoder<Type>::num_values_;

    thrust::device_vector<T> dictionary_;

    RleDecoder idx_decoder_;

    int num_dictionary_values_;
};

template <typename Type, typename RleDecoder>
inline void
DictionaryDecoder<Type, RleDecoder>::SetDict(
  ::parquet::Decoder<Type> *dictionary) {
    int num_dictionary_values = dictionary->values_left();
    num_dictionary_values_    = num_dictionary_values;
    dictionary_.resize(num_dictionary_values);
    dictionary->Decode(thrust::raw_pointer_cast(dictionary_.data()),
                       num_dictionary_values);
}

template <>
inline void
DictionaryDecoder<::parquet::BooleanType, ::arrow::util::RleDecoder>::SetDict(
  ::parquet::Decoder<::parquet::BooleanType> *) {
    ::parquet::ParquetException::NYI(
      "Dictionary encoding is not implemented for boolean values");
}

}  // namespace internal
}  // namespace parquet
}  // namespace gdf
