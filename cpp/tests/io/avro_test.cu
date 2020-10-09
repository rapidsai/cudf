#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <iterator>

#include "avro_test.hpp"

class AvroReaderTest : public cudf::test::BaseFixture {
};

TEST_F(AvroReaderTest, CanParseAvroMagic)
{
  auto const input = std::vector<uint8_t>{'O', 'b', 'j', 0x01};

  bool is_avro;
  auto const position = cudf::io::avro::read_avro_magic(input.begin(), input.end(), is_avro);

  EXPECT_TRUE(is_avro);
  EXPECT_EQ(input.begin() + 4, position);
}

TEST_F(AvroReaderTest, CanParseLargeUnsignedInt)
{
  auto const input = std::vector<uint8_t>{
    0b10000000,
    0b10000001,
    0b10000000,
    0b10000000,
    0b10000001,
    0b10000000,
    0b10000000,
    0b10000001,
    0b10000000,
    0b00000000,
  };

  uint64_t result;
  auto const position = cudf::io::avro::parse_uint64_t(input.begin(), input.end(), result);

  EXPECT_EQ(
    static_cast<uint64_t>(0b0000000000000010000000000000000000010000000000000000000010000000),
    result);

  EXPECT_EQ(input.begin() + 10, position);
}

TEST_F(AvroReaderTest, CanParseMultipleUnsignedInts)
{
  auto const input = std::vector<uint8_t>{
    0b10000001,  // a
    0b00000010,
    0b00000011,  // b
    0b10000100,  // c
    0b10000101,
    0b10000110,
    0b00000111,
    0b10001001,  // d
    0b00001010,
    0b00001011,  // e
  };

  auto position = input.begin();

  uint64_t result_a;
  uint64_t result_b;
  uint64_t result_c;
  uint64_t result_d;
  uint64_t result_e;

  position = cudf::io::avro::parse_uint64_t(position, input.end(), result_a);
  position = cudf::io::avro::parse_uint64_t(position, input.end(), result_b);
  position = cudf::io::avro::parse_uint64_t(position, input.end(), result_c);
  position = cudf::io::avro::parse_uint64_t(position, input.end(), result_d);
  position = cudf::io::avro::parse_uint64_t(position, input.end(), result_e);

  EXPECT_EQ(static_cast<uint64_t>(0b00000100000001), result_a);
  EXPECT_EQ(static_cast<uint64_t>(0b0000011), result_b);
  EXPECT_EQ(static_cast<uint64_t>(0b0000111000011000001010000100), result_c);
  EXPECT_EQ(static_cast<uint64_t>(0b00010100001001), result_d);
  EXPECT_EQ(static_cast<uint64_t>(0b0001011), result_e);

  EXPECT_EQ(input.begin() + 10, position);
}

TEST_F(AvroReaderTest, CanParseStringLengthZero)
{
  auto input = std::vector<uint8_t>{0b0000000'0};

  uint64_t result;
  auto position = cudf::io::avro::parse_string_length(input.begin(), input.end(), result);

  ASSERT_EQ(static_cast<uint64_t>(0), result);
  ASSERT_EQ(input.begin() + 1, position);
}

TEST_F(AvroReaderTest, CanParseStringLengthZeroWithFlagBit)
{
  auto input = std::vector<uint8_t>{0b0001000'1};

  uint64_t result;
  auto position = cudf::io::avro::parse_string_length(input.begin(), input.end(), result);

  ASSERT_EQ(static_cast<uint64_t>(0), result);
  ASSERT_EQ(input.begin() + 1, position);
}

TEST_F(AvroReaderTest, CanParseStringLengthNonZero)
{
  auto input = std::vector<uint8_t>{0b0001011'0};

  uint64_t result;
  auto position = cudf::io::avro::parse_string_length(input.begin(), input.end(), result);

  ASSERT_EQ(static_cast<uint64_t>(11), result);
  ASSERT_EQ(input.begin() + 1, position);
}

TEST_F(AvroReaderTest, CanParseString)
{
  auto input =
    std::vector<uint8_t>{0b00010110, 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '~'};

  std::string result;
  auto position = cudf::io::avro::parse_string(input.begin(), input.end(), result);

  ASSERT_EQ("hello world", result);
  ASSERT_EQ(input.begin() + 12, position);
}

TEST_F(AvroReaderTest, CanParseStringEmpty)
{
  auto input = std::vector<uint8_t>{0b00000000, 'a', 'n', 'y'};

  std::string result;
  auto position = cudf::io::avro::parse_string(input.begin(), input.end(), result);

  ASSERT_EQ("", result);
  ASSERT_EQ(input.begin() + 1, position);
}

TEST_F(AvroReaderTest, CanParseAvroMetadataKvpsBlocks)
{
  auto input = std::vector<uint8_t>{0b010,    // 2 key value pairs
                                    0b011'0,  // key str length 3
                                    static_cast<uint8_t>('a'),
                                    static_cast<uint8_t>('b'),
                                    static_cast<uint8_t>('c'),
                                    0b100'0,  // value str length 4
                                    static_cast<uint8_t>('d'),
                                    static_cast<uint8_t>('e'),
                                    static_cast<uint8_t>('f'),
                                    static_cast<uint8_t>('g'),
                                    0b101'0,  // key str length 5
                                    static_cast<uint8_t>('h'),
                                    static_cast<uint8_t>('i'),
                                    static_cast<uint8_t>('j'),
                                    static_cast<uint8_t>('k'),
                                    static_cast<uint8_t>('l'),
                                    0b110'0,  // value str length 6
                                    static_cast<uint8_t>('m'),
                                    static_cast<uint8_t>('n'),
                                    static_cast<uint8_t>('o'),
                                    static_cast<uint8_t>('p'),
                                    static_cast<uint8_t>('q'),
                                    static_cast<uint8_t>('r')};

  std::map<std::string, std::string> kvps;

  cudf::io::avro::parse_avro_metadata_kvps_blocks(input.begin(),  //
                                                  input.end(),
                                                  std::inserter(kvps, kvps.end()));

  EXPECT_EQ(static_cast<size_t>(2), kvps.size());

  ASSERT_NE(kvps.end(), kvps.find("abc"));
  ASSERT_NE(kvps.end(), kvps.find("hijkl"));

  EXPECT_EQ("defg", kvps["abc"]);
  EXPECT_EQ("mnopqr", kvps["hijkl"]);
}

CUDF_TEST_PROGRAM_MAIN()
