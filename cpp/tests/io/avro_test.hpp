#include <map>
#include <string>

namespace cudf {
namespace io {
namespace avro {

enum class avro_compression_codec { null, deflate };

struct avro_metadata {
  std::string codec;
};

constexpr uint64_t avro_magic = (('O' << 0) | ('b' << 8) | ('j' << 16) | (0x01 << 24));

template <typename Iter>
inline constexpr Iter read_avro_magic(Iter begin, Iter end, bool& result)
{
  uint32_t sig4 = 0;

  sig4 |= *begin++ << 0;
  sig4 |= *begin++ << 8;
  sig4 |= *begin++ << 16;
  sig4 |= *begin++ << 24;

  result = sig4 == avro_magic;

  return begin;
}

template <typename Iter>
inline constexpr Iter parse_uint64_t(Iter begin, Iter end, uint64_t& result)
{
  result = 0;

  for (uint64_t l = 0;; l += 7) {
    uint64_t c = *begin++;
    result |= (c & 0x7f) << l;
    if (c < 0x80) { break; }
  }

  return begin;
}

template <typename Iter>
inline constexpr Iter parse_int64_t(Iter begin, Iter end, int64_t& result)
{
  uint64_t temp = 0;

  begin = parse_uint64_t(begin, end, temp);

  result = static_cast<int64_t>((temp >> 1u) ^ -(int64_t)(temp & 1));

  return begin;
}

template <typename Iter>
inline constexpr Iter parse_string_length(Iter begin, Iter end, uint64_t& result)
{
  begin = parse_uint64_t(begin, end, result);

  if (result & 1) { result = 0; }

  result >>= 1;

  // assume full string comes next, so ignore these
  // if (begin >= end) { return 0; }
  // result = std::min(result, std::distance(begin, end));

  return begin;
}

template <typename Iter>
inline constexpr Iter parse_string(Iter begin, Iter end, std::string& result)
{
  uint64_t str_length = 0;

  begin = parse_string_length(begin, end, str_length);

  Iter const str_end = begin + str_length;

  result.assign(begin, str_end);

  return str_end;
}

struct avro_schema {
};

template <typename Iter>
inline constexpr Iter parse_avro_schema(Iter begin, Iter end, avro_schema& schema)
{
  return end;
}

template <typename Iter, typename OutputIter>
inline constexpr Iter parse_avro_metadata_syncmarker(Iter begin, Iter end, OutputIter output)
{
  for (auto i = 0; i < 16; i++) { *output++ = *begin++; }

  return begin;
}

// TODO: try to make constexpr
template <typename Iter, typename OutputIter>
inline Iter parse_avro_metadata_kvps(Iter begin, Iter end, uint32_t num_entries, OutputIter out)
{
  for (uint32_t i = 0; i < num_entries; i++) {
    std::string key;
    std::string value;
    begin = parse_string(begin, end, key);
    begin = parse_string(begin, end, value);

    *out++ = {key, value};
  }

  return begin;
}

// TODO: try to make constexpr
template <typename Iter, typename OutputIter>
inline Iter parse_avro_metadata_kvps_blocks(Iter begin, Iter end, OutputIter out)
{
  while (true) {
    uint64_t num_kvp = 0;

    begin = parse_uint64_t(begin, end, num_kvp);

    // uint32_t num_kvp_unsigned = static_cast<uint32_t>(num_kvp);

    if (num_kvp == 0) { break; }

    begin = parse_avro_metadata_kvps(begin, end, num_kvp, out);
  }

  return begin;
}

}  // namespace avro
}  // namespace io
}  // namespace cudf
