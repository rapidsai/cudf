/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "corpus_benchmark.hpp"

#include <nvbench/nvbench.cuh>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace {

using regex_ir_benchmark::corpus_case;
using regex_ir_benchmark::corpus_source;

constexpr auto cpp_tokens =
  R"REGEX((^[ ]*#(?:[^\\\n]|\\[^\n_[:punct:][:alnum:]]*[\n[:punct:][:word:]])*)|(//[^\n]*|/\*.*?\*/)|\b([+-]?(?:(?:0x[[:xdigit:]]+)|(?:(?:[[:digit:]]*\.)?[[:digit:]]+(?:[eE][+-]?[[:digit:]]+)?))u?(?:(?:int(?:8|16|32|64))|L)?)\b|('(?:[^\\']|\\.)*'|"(?:[^\\"]|\\.)*")|\b(__asm|__cdecl|__declspec|__export|__far16|__fastcall|__fortran|__import|__pascal|__rtti|__stdcall|_asm|_cdecl|__except|_export|_far16|_fastcall|__finally|_fortran|_import|_pascal|_stdcall|__thread|__try|asm|auto|bool|break|case|catch|cdecl|char|class|const|const_cast|continue|default|delete|do|double|dynamic_cast|else|enum|explicit|extern|false|float|for|friend|goto|if|inline|int|long|mutable|namespace|new|operator|pascal|private|protected|public|register|reinterpret_cast|return|short|signed|sizeof|static|static_cast|struct|switch|template|this|throw|true|try|typedef|typeid|typename|union|unsigned|using|virtual|void|volatile|wchar_t|while)\b)REGEX";

constexpr auto cudf_cpp_tokens =
  R"REGEX((^[ ]*#[^\n]*)|(//[^\n]*|/\*.*?\*/)|\b([+-]?(?:(?:0x[A-Fa-f0-9]+)|(?:(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?[0-9]+)?))u?(?:(?:int(?:8|16|32|64))|L)?)\b|('(?:[^\\']|\\.)*'|"(?:[^\\"]|\\.)*")|\b(__asm|__cdecl|__declspec|__export|__far16|__fastcall|__fortran|__import|__pascal|__rtti|__stdcall|_asm|_cdecl|__except|_export|_far16|_fastcall|__finally|_fortran|_import|_pascal|_stdcall|__thread|__try|asm|auto|bool|break|case|catch|cdecl|char|class|const|const_cast|continue|default|delete|do|double|dynamic_cast|else|enum|explicit|extern|false|float|for|friend|goto|if|inline|int|long|mutable|namespace|new|operator|pascal|private|protected|public|register|reinterpret_cast|return|short|signed|sizeof|static|static_cast|struct|switch|template|this|throw|true|try|typedef|typeid|typename|union|unsigned|using|virtual|void|volatile|wchar_t|while)\b)REGEX";

constexpr auto email_expression =
  R"REGEX(^([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$)REGEX";

// expressions and exact inputs follow Boost.Regex 1.41's GCC comparison
constexpr std::array boost_cases{
  corpus_case{.name       = "01_long_twain",
              .family     = "english_long",
              .expression = "Twain",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "02_long_huck",
              .family     = "english_long",
              .expression = "Huck[[:alpha:]]+",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "03_long_ing",
              .family     = "english_long",
              .expression = "[[:alpha:]]+ing",
              .corpus     = corpus_source::Mtent12},
  corpus_case{.name       = "04_long_line_twain",
              .family     = "english_long",
              .expression = R"REGEX(^[^\n]*?Twain)REGEX",
              .corpus     = corpus_source::Mtent12,
              .multiline  = true},
  corpus_case{.name       = "05_long_names",
              .family     = "english_long",
              .expression = "Tom|Sawyer|Huckleberry|Finn",
              .corpus     = corpus_source::Mtent12},
  corpus_case{
    .name   = "06_long_names_near_river",
    .family = "english_long",
    .expression =
      R"REGEX((Tom|Sawyer|Huckleberry|Finn).{0,30}river|river.{0,30}(Tom|Sawyer|Huckleberry|Finn))REGEX",
    .corpus = corpus_source::Mtent12},
  corpus_case{.name       = "07_medium_twain",
              .family     = "english_medium",
              .expression = "Twain",
              .corpus     = corpus_source::Mtent12Prefix50K},
  corpus_case{.name       = "08_medium_huck",
              .family     = "english_medium",
              .expression = "Huck[[:alpha:]]+",
              .corpus     = corpus_source::Mtent12Prefix50K},
  corpus_case{.name       = "09_medium_ing",
              .family     = "english_medium",
              .expression = "[[:alpha:]]+ing",
              .corpus     = corpus_source::Mtent12Prefix50K},
  corpus_case{.name       = "10_medium_line_twain",
              .family     = "english_medium",
              .expression = R"REGEX(^[^\n]*?Twain)REGEX",
              .corpus     = corpus_source::Mtent12Prefix50K,
              .multiline  = true},
  corpus_case{.name       = "11_medium_names",
              .family     = "english_medium",
              .expression = "Tom|Sawyer|Huckleberry|Finn",
              .corpus     = corpus_source::Mtent12Prefix50K},
  corpus_case{
    .name   = "12_medium_names_near_river",
    .family = "english_medium",
    .expression =
      R"REGEX((Tom|Sawyer|Huckleberry|Finn).{0,30}river|river.{0,30}(Tom|Sawyer|Huckleberry|Finn))REGEX",
    .corpus = corpus_source::Mtent12Prefix50K},
  corpus_case{
    .name   = "13_cpp_declaration",
    .family = "cpp_search",
    .expression =
      R"REGEX(^(template[[:space:]]*<[^;:{]+>[[:space:]]*)?(class|struct)[[:space:]]*(\b\w+\b([ ]*\([^)]*\))?[[:space:]]*)*(\b\w*\b)[[:space:]]*(<[^;:{]+>[[:space:]]*)?(\{|:[^;\{()]*\{))REGEX",
    .corpus    = corpus_source::BoostCrc,
    .multiline = true},
  corpus_case{.name                  = "14_cpp_tokens",
              .family                = "cpp_search",
              .expression            = cpp_tokens,
              .corpus                = corpus_source::BoostCrc,
              .comparison_expression = cudf_cpp_tokens,
              .multiline             = true},
  corpus_case{.name       = "15_cpp_include",
              .family     = "cpp_search",
              .expression = R"REGEX(^[ ]*#[ ]*include[ ]+("[^"]+"|<[^>]+>))REGEX",
              .corpus     = corpus_source::BoostCrc,
              .multiline  = true},
  corpus_case{.name       = "16_boost_include",
              .family     = "cpp_search",
              .expression = R"REGEX(^[ ]*#[ ]*include[ ]+("boost/[^"]+"|<boost/[^>]+>))REGEX",
              .corpus     = corpus_source::BoostCrc,
              .multiline  = true},
  corpus_case{.name             = "17_html_names",
              .family           = "html_search",
              .expression       = "beman|john|dave",
              .corpus           = corpus_source::BoostLibraries,
              .case_insensitive = true},
  corpus_case{.name             = "18_html_paragraph",
              .family           = "html_search",
              .expression       = "<p>.*?</p>",
              .corpus           = corpus_source::BoostLibraries,
              .case_insensitive = true},
  corpus_case{.name             = "19_html_anchor",
              .family           = "html_search",
              .expression       = R"REGEX(<a[^>]+href=("[^"]*"|[^[:space:]]+)[^>]*>)REGEX",
              .corpus           = corpus_source::BoostLibraries,
              .case_insensitive = true},
  corpus_case{.name             = "20_html_heading",
              .family           = "html_search",
              .expression       = "<h[12345678][^>]*>.*?</h[12345678]>",
              .corpus           = corpus_source::BoostLibraries,
              .case_insensitive = true},
  corpus_case{.name             = "21_html_image",
              .family           = "html_search",
              .expression       = R"REGEX(<img[^>]+src=("[^"]*"|[^[:space:]]+)[^>]*>)REGEX",
              .corpus           = corpus_source::BoostLibraries,
              .case_insensitive = true},
  corpus_case{.name       = "22_html_font",
              .family     = "html_search",
              .expression = R"REGEX(<font[^>]+face=("[^"]*"|[^[:space:]]+)[^>]*>.*?</font>)REGEX",
              .corpus     = corpus_source::BoostLibraries,
              .case_insensitive = true},
  corpus_case{.name        = "23_simple_literal",
              .family      = "simple_match",
              .expression  = "abc",
              .corpus      = corpus_source::Inline,
              .inline_text = "abc"},
  corpus_case{.name        = "24_ftp_response",
              .family      = "simple_match",
              .expression  = R"REGEX(^([0-9]+)(\-| |$)(.*)$)REGEX",
              .corpus      = corpus_source::Inline,
              .inline_text = "100- this is a line of ftp response which contains a message string"},
  corpus_case{.name        = "25_credit_card",
              .family      = "simple_match",
              .expression  = "([[:digit:]]{4}[- ]){3}[[:digit:]]{3,4}",
              .corpus      = corpus_source::Inline,
              .inline_text = "1234-5678-1234-456"},
  corpus_case{.name        = "26_email_uk",
              .family      = "simple_match",
              .expression  = email_expression,
              .corpus      = corpus_source::Inline,
              .inline_text = "john@johnmaddock.co.uk"},
  corpus_case{.name        = "27_email_edu",
              .family      = "simple_match",
              .expression  = email_expression,
              .corpus      = corpus_source::Inline,
              .inline_text = "foo12@foo.edu"},
  corpus_case{.name        = "28_email_tv",
              .family      = "simple_match",
              .expression  = email_expression,
              .corpus      = corpus_source::Inline,
              .inline_text = "bob.smith@foo.tv"},
  corpus_case{.name        = "29_postcode_eh",
              .family      = "simple_match",
              .expression  = "^[a-zA-Z]{1,2}[0-9][0-9A-Za-z]{0,1} {0,1}[0-9][A-Za-z]{2}$",
              .corpus      = corpus_source::Inline,
              .inline_text = "EH10 2QQ"},
  corpus_case{.name        = "30_postcode_g",
              .family      = "simple_match",
              .expression  = "^[a-zA-Z]{1,2}[0-9][0-9A-Za-z]{0,1} {0,1}[0-9][A-Za-z]{2}$",
              .corpus      = corpus_source::Inline,
              .inline_text = "G1 1AA"},
  corpus_case{.name        = "31_postcode_sw",
              .family      = "simple_match",
              .expression  = "^[a-zA-Z]{1,2}[0-9][0-9A-Za-z]{0,1} {0,1}[0-9][A-Za-z]{2}$",
              .corpus      = corpus_source::Inline,
              .inline_text = "SW1 1ZZ"},
  corpus_case{.name        = "32_date_short",
              .family      = "simple_match",
              .expression  = "^[[:digit:]]{1,2}/[[:digit:]]{1,2}/[[:digit:]]{4}$",
              .corpus      = corpus_source::Inline,
              .inline_text = "4/1/2001"},
  corpus_case{.name        = "33_date_long",
              .family      = "simple_match",
              .expression  = "^[[:digit:]]{1,2}/[[:digit:]]{1,2}/[[:digit:]]{4}$",
              .corpus      = corpus_source::Inline,
              .inline_text = "12/12/2001"},
  corpus_case{.name        = "34_integer",
              .family      = "simple_match",
              .expression  = R"REGEX(^[-+]?[[:digit:]]*\.?[[:digit:]]*$)REGEX",
              .corpus      = corpus_source::Inline,
              .inline_text = "123"},
  corpus_case{.name        = "35_positive_float",
              .family      = "simple_match",
              .expression  = R"REGEX(^[-+]?[[:digit:]]*\.?[[:digit:]]*$)REGEX",
              .corpus      = corpus_source::Inline,
              .inline_text = "+3.14159"},
  corpus_case{.name        = "36_negative_float",
              .family      = "simple_match",
              .expression  = R"REGEX(^[-+]?[[:digit:]]*\.?[[:digit:]]*$)REGEX",
              .corpus      = corpus_source::Inline,
              .inline_text = "-3.14159"}};

corpus_case const& get_case(nvbench::state& state)
{
  auto index = state.get_int64("Case");
  if (index < 1 || static_cast<std::size_t>(index) > boost_cases.size()) {
    throw std::invalid_argument("Case is outside the registered Boost range");
  }
  return boost_cases[static_cast<std::size_t>(index - 1)];
}

void regex_ir_boost(nvbench::state& state)
{
  regex_ir_benchmark::run_regex_ir(state, get_case(state));
}

void cudf_boost(nvbench::state& state) { regex_ir_benchmark::run_cudf(state, get_case(state)); }

}  // namespace

NVBENCH_BENCH(regex_ir_boost)
  .set_name("regex_ir/boost")
  .add_int64_axis("Case", {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36})
  .add_int64_axis("Rows", {1024, 4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});

NVBENCH_BENCH(cudf_boost)
  .set_name("cudf/boost")
  .add_int64_axis("Case", {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36})
  .add_int64_axis("Rows", {1024, 4096, 32768, 262144})
  .add_int64_axis("Columns", {1, 8})
  .add_int64_axis("MaxStringBytes", {64, 256, 1024});
