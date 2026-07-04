/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// diagnostics

namespace regex_ir {

/**
 * @brief Byte range in the source regular expression
 */
struct source_span {
  std::size_t offset = 0;  ///< Zero-based byte offset
  std::size_t length = 0;  ///< Length in bytes
};

/**
 * @brief Stable categories for compile and verification diagnostics
 */
enum class diagnostic_code : std::uint8_t {
  UNEXPECTED_END          = 0,   ///< Pattern ended before a construct was complete
  UNEXPECTED_TOKEN        = 1,   ///< Token is not valid at its source position
  INVALID_ESCAPE          = 2,   ///< Escape sequence is malformed or unknown
  INVALID_CHARACTER_CLASS = 3,   ///< Character class is malformed
  INVALID_QUANTIFIER      = 4,   ///< Repetition syntax or bounds are invalid
  UNMATCHED_PARENTHESIS   = 5,   ///< Opening or closing parenthesis has no match
  UNSUPPORTED_FEATURE     = 6,   ///< Pattern uses syntax outside the supported subset
  RESOURCE_LIMIT          = 7,   ///< Configured compilation resource limit was exceeded
  INVALID_REPLACEMENT     = 8,   ///< Replacement template is malformed
  INVALID_AUTOMATA_IR     = 9,   ///< Automata IR invariant was violated
  INVALID_INSTRUCTION_IR  = 10,  ///< Instruction IR invariant was violated
};

/**
 * @brief Structured compiler or verifier diagnostic
 */
struct diagnostic {
  diagnostic_code code = diagnostic_code::UNEXPECTED_END;  ///< Diagnostic category
  source_span span     = {};                               ///< Related source range
  std::string message  = {};                               ///< Human-readable explanation
};

}  // namespace regex_ir

// compile options and operations

namespace regex_ir {

/**
 * @brief Unit used to decode and advance through input text
 */
enum class character_mode : std::uint8_t {
  UTF8  = 0,  ///< Decode UTF-8 and advance by Unicode code point
  BYTES = 1,  ///< Treat every input byte as one character
};

/**
 * @brief Resource limits applied while compiling a regular expression
 */
struct compile_limits {
  std::size_t max_pattern_bytes = 1U << 20U;  ///< Maximum pattern size in bytes
  std::size_t max_nesting       = 256;        ///< Maximum parser nesting depth
  std::size_t max_states        = 1U << 18U;  ///< Maximum number of automata states
  std::size_t max_transitions   = 1U << 20U;  ///< Maximum number of automata edges
  std::size_t max_captures      = 256;        ///< Maximum number of capture groups
  std::uint32_t max_repeat      = 1000;       ///< Maximum finite repetition bound
};

/**
 * @brief Syntax, text-decoding, and resource options for regex compilation
 */
struct compile_options {
  bool case_insensitive : 1 = false;  ///< Enable Unicode case-insensitive matching
  bool multiline        : 1 = false;  ///< Make input anchors recognize line boundaries
  bool dot_all          : 1 = false;  ///< Make dot match every configured line terminator
  bool ascii_classes    : 1 = true;   ///< Restrict predefined classes to ASCII characters
  bool extended_newline : 1 = false;  ///< Recognize CR, NEL, LS, and PS as line terminators
  character_mode characters = character_mode::UTF8;  ///< Input character decoding mode
  compile_limits limits     = {};                    ///< Compilation resource limits
};

/**
 * @brief Operation encoded by Instruction IR
 */
enum class operation_kind : std::uint8_t {
  CONTAINS = 0,  ///< Return whether the input contains a match
  MATCHES  = 1,  ///< Return whether the complete input matches
  COUNT    = 2,  ///< Count non-overlapping matches
  EXTRACT  = 3,  ///< Return captures for the first match
  FIND     = 4,  ///< Return the span of the first match
  REPLACE  = 5,  ///< Replace non-overlapping matches
  SPLIT    = 6,  ///< Split the input around non-overlapping matches
};

/**
 * @brief Requested regex operation and its operation-specific data
 */
struct operation {
  operation_kind kind     = operation_kind::MATCHES;  ///< Selected operation
  std::string replacement = {};                       ///< Replacement template for `REPLACE`

  /**
   * @brief Create a boolean operation that searches for the first match
   *
   * @return Contains operation
   */
  static operation contains() { return {operation_kind::CONTAINS, {}}; }

  /**
   * @brief Create a boolean operation that requires a full-input match
   *
   * @return Matches operation
   */
  static operation matches() { return {operation_kind::MATCHES, {}}; }

  /**
   * @brief Create an operation that counts non-overlapping matches
   *
   * @return Count operation
   */
  static operation count() { return {operation_kind::COUNT, {}}; }

  /**
   * @brief Create an operation that returns capture spans for the first match
   *
   * @return Extract operation
   */
  static operation extract() { return {operation_kind::EXTRACT, {}}; }

  /**
   * @brief Create an operation that returns the span of the first match
   *
   * @return Find operation
   */
  static operation find() { return {operation_kind::FIND, {}}; }

  /**
   * @brief Create an operation that replaces non-overlapping matches
   *
   * @param value Replacement template using `$N` capture references
   * @return Replace operation
   */
  static operation replace(std::string value)
  {
    return {operation_kind::REPLACE, std::move(value)};
  }

  /**
   * @brief Create an operation that splits input around non-overlapping matches
   *
   * @return Split operation
   */
  static operation split() { return {operation_kind::SPLIT, {}}; }
};

/**
 * @brief Pass controls for Instruction IR optimization
 */
struct optimization_options {
  bool remove_unreachable        : 1 = true;  ///< Remove blocks unreachable from the entry
  bool fold_epsilon_jumps        : 1 = true;  ///< Bypass trivial jump-only blocks
  bool fuse_literals             : 1 = true;  ///< Fuse linear singleton matches into literals
  bool strip_unobserved_captures : 1 = true;  ///< Remove capture writes not used by the result
  std::size_t literal_fusion_limit   = 64;    ///< Maximum code points in one fused literal
};

/**
 * @brief Sentinel upper bound representing an unbounded repetition
 */
inline constexpr std::uint32_t unbounded_repeat = std::numeric_limits<std::uint32_t>::max();

}  // namespace regex_ir

// automata IR

namespace regex_ir {

/**
 * @brief Numeric identifier of an Automata IR state
 */
using state_id = std::uint32_t;

/**
 * @brief Sentinel that does not identify an Automata IR state
 */
inline constexpr state_id invalid_state = static_cast<state_id>(-1);

/**
 * @brief Inclusive Unicode code-point interval
 */
struct codepoint_range {
  char32_t first = U'\0';  ///< First code point in the interval
  char32_t last  = U'\0';  ///< Last code point in the interval
};

/**
 * @brief Recognized character-class category retained alongside normalized ranges
 */
enum class predicate_class : std::uint8_t {
  NONE      = 0,  ///< Predicate has no recognized shorthand category
  DIGIT     = 1,  ///< Configured digit class
  NOT_DIGIT = 2,  ///< Negated configured digit class
  WORD      = 3,  ///< Configured word-character class
  NOT_WORD  = 4,  ///< Negated configured word-character class
  SPACE     = 5,  ///< Configured whitespace class
  NOT_SPACE = 6,  ///< Negated configured whitespace class
  ANY       = 7,  ///< Dot wildcard
};

/**
 * @brief Normalized predicate evaluated by a consuming automata state
 */
struct character_predicate {
  std::vector<codepoint_range> ranges = {};  ///< Sorted inclusive code-point ranges
  predicate_class recognized          = predicate_class::NONE;  ///< Original shorthand category
  bool negated          : 1           = false;                  ///< Invert range membership
  bool matches_newline  : 1           = true;  ///< Whether dot accepts configured line terminators
  bool extended_newline : 1 = false;  ///< Whether dot uses the extended line-terminator set

  /**
   * @brief Check whether a code point satisfies this predicate
   *
   * @param value Code point to test
   * @return true if `value` satisfies this predicate
   */
  [[nodiscard]] bool matches(char32_t value) const noexcept;

  /**
   * @brief Check whether this predicate matches exactly one code point
   *
   * @return true if this predicate is a non-negated singleton range
   */
  [[nodiscard]] bool is_singleton() const noexcept;

  /**
   * @brief Return the code point in a singleton predicate
   *
   * @return Singleton code point, or `U'\0'` when this predicate is not a singleton
   */
  [[nodiscard]] char32_t singleton() const noexcept;
};

/**
 * @brief Zero-width assertion evaluated at the current input position
 */
enum class assertion_kind : std::uint8_t {
  BEGIN_INPUT       = 0,  ///< Absolute beginning of input
  END_INPUT         = 1,  ///< Absolute end of input
  WORD_BOUNDARY     = 2,  ///< Transition between configured word and non-word characters
  NOT_WORD_BOUNDARY = 3,  ///< Position that is not a configured word boundary
  BEGIN_LINE        = 4,  ///< Beginning of input or a configured multiline boundary
  END_LINE          = 5,  ///< End of input or a configured line boundary
};

/**
 * @brief Capture-boundary update performed by a tagged automata state
 */
enum class capture_action : std::uint8_t {
  BEGIN = 0,  ///< Record the beginning of a capture
  END   = 1,  ///< Record the end of a capture
};

/**
 * @brief Operation performed by an Automata IR state
 */
enum class automata_state_kind : std::uint8_t {
  JUMP      = 0,  ///< Epsilon transition to one successor
  BRANCH    = 1,  ///< Ordered epsilon transition to multiple successors
  CONSUME   = 2,  ///< Match and consume one character
  ASSERTION = 3,  ///< Evaluate a zero-width assertion
  CAPTURE   = 4,  ///< Record a capture boundary
  ACCEPT    = 5,  ///< Accept the current match
};

/**
 * @brief Ordered transition between Automata IR states
 */
struct automata_edge {
  state_id target        = invalid_state;  ///< Destination state
  std::uint32_t priority = 0;              ///< Lower values are attempted first
};

/**
 * @brief One node in an ordered Thompson automaton
 */
struct automata_state {
  state_id id                      = invalid_state;                ///< Dense state identifier
  automata_state_kind kind         = automata_state_kind::JUMP;    ///< State operation
  source_span source               = {};                           ///< Related pattern range
  std::vector<automata_edge> edges = {};                           ///< Ordered outgoing edges
  character_predicate predicate    = {};                           ///< Predicate for `CONSUME`
  assertion_kind assertion         = assertion_kind::BEGIN_INPUT;  ///< Assertion for `ASSERTION`
  capture_action capture           = capture_action::BEGIN;        ///< Action for `CAPTURE`
  std::uint32_t capture_index      = 0;                            ///< Capture index for `CAPTURE`
};

/**
 * @brief Ordered Thompson automaton produced from a regular expression
 */
struct automata_ir {
  std::string pattern                = {};             ///< Original UTF-8 pattern bytes
  compile_options options            = {};             ///< Options used to parse the pattern
  std::vector<automata_state> states = {};             ///< Dense state table
  state_id entry                     = invalid_state;  ///< Entry state
  state_id accept                    = invalid_state;  ///< Unique accepting state
  std::uint32_t capture_count        = 0;              ///< Number of explicit capture groups
};

/**
 * @brief Validate an Automata IR graph
 *
 * @param ir Automata IR to validate
 * @return Diagnostics describing every detected invariant violation
 */
[[nodiscard]] std::vector<diagnostic> verify(automata_ir const& ir);

/**
 * @brief Render Automata IR as deterministic diagnostic text
 *
 * the returned text is intended for inspection and is not a serialization format.
 *
 * @param ir Automata IR to render
 * @return Human-readable Automata IR
 */
[[nodiscard]] std::string to_string(automata_ir const& ir);

}  // namespace regex_ir

// instruction IR

namespace regex_ir {

/**
 * @brief Numeric identifier of an Instruction IR block
 */
using block_id = std::uint32_t;

/**
 * @brief Sentinel that does not identify an Instruction IR block
 */
inline constexpr block_id invalid_block = static_cast<block_id>(-1);

/**
 * @brief Require a number of characters to remain at the cursor
 */
struct can_peek {
  std::uint32_t characters = 1;  ///< Required number of logical characters
};

/**
 * @brief Decode the character at the current cursor
 */
struct read_character {};

/**
 * @brief Test the current character against a predicate
 */
struct match_character {
  character_predicate predicate = {};  ///< Predicate that must match
};

/**
 * @brief Test consecutive characters against a fixed literal
 */
struct match_literal {
  std::u32string value = {};  ///< Literal Unicode code points
};

/**
 * @brief Advance the input cursor by logical characters
 */
struct advance_cursor {
  std::uint32_t characters = 1;  ///< Number of characters to advance
};

/**
 * @brief Evaluate a zero-width assertion at the cursor
 */
struct test_assertion {
  assertion_kind kind = assertion_kind::BEGIN_INPUT;  ///< Assertion to evaluate
};

/**
 * @brief Record one boundary of a capture group
 */
struct write_capture {
  capture_action action       = capture_action::BEGIN;  ///< Boundary to record
  std::uint32_t capture_index = 0;                      ///< Capture group index
};

/**
 * @brief Accept the current candidate match
 */
struct emit_accept {};

/**
 * @brief Instruction variant used inside an Instruction IR block
 */
using instruction = std::variant<can_peek,
                                 read_character,
                                 match_character,
                                 match_literal,
                                 advance_cursor,
                                 test_assertion,
                                 write_capture,
                                 emit_accept>;

/**
 * @brief Ordered control-flow edge between Instruction IR blocks
 */
struct block_edge {
  block_id target        = invalid_block;  ///< Destination block
  std::uint32_t priority = 0;              ///< Lower values are attempted first
};

/**
 * @brief Straight-line instruction sequence with ordered successors
 */
struct instruction_block {
  block_id id                           = invalid_block;  ///< Dense block identifier
  source_span source                    = {};             ///< Related pattern range
  std::vector<instruction> instructions = {};             ///< Instructions executed in order
  std::vector<block_edge> successors    = {};             ///< Ordered control-flow successors
};

/**
 * @brief Parsed component of a replacement template
 */
struct replacement_token {
  /**
   * @brief Replacement token category
   */
  enum class kind : std::uint8_t {
    LITERAL = 0,  ///< Literal UTF-8 bytes
    CAPTURE = 1,  ///< Captured input span
  };

  kind type                   = kind::LITERAL;  ///< Token category
  std::string literal         = {};             ///< Bytes for a literal token
  std::uint32_t capture_index = 0;              ///< Capture index for a capture token
};

/**
 * @brief Logical result produced when Instruction IR is executed
 */
enum class result_shape : std::uint8_t {
  BOOLEAN      = 0,  ///< Boolean match result
  MATCH_SPAN   = 1,  ///< First match span
  MATCH_COUNT  = 2,  ///< Number of matches
  CAPTURES     = 3,  ///< Capture spans
  REPLACEMENT  = 4,  ///< Replaced string
  SPLIT_FIELDS = 5,  ///< Fields split around matches
};

/**
 * @brief Operation-specific execution policy encoded in Instruction IR
 */
struct operation_control {
  bool scan_input          : 1 = false;  ///< Try candidates after input position zero
  bool require_end         : 1 = false;  ///< Require acceptance at end of input
  bool first_only          : 1 = true;   ///< Stop after the first accepted match
  bool advance_after_empty : 1 = true;   ///< Advance after an empty global match
  result_shape result          = result_shape::BOOLEAN;  ///< Produced result shape
};

/**
 * @brief Static complexity metrics for an Instruction IR graph
 */
struct ir_metrics {
  std::size_t blocks             = 0;  ///< Number of blocks
  std::size_t branches           = 0;  ///< Number of blocks with multiple successors
  std::size_t predicates         = 0;  ///< Number of character predicate tests
  std::size_t stream_reads       = 0;  ///< Number of character-read operations
  std::size_t capture_writes     = 0;  ///< Number of capture-boundary writes
  std::size_t literal_codepoints = 0;  ///< Number of code points in fused literals
};

/**
 * @brief Typed operation-specialized control-flow IR
 */
struct instruction_ir {
  std::string pattern                        = {};             ///< Original UTF-8 pattern bytes
  compile_options options                    = {};             ///< Options used for compilation
  operation selected_operation               = {};             ///< Operation encoded by this IR
  operation_control control                  = {};             ///< Execution policy
  std::vector<instruction_block> blocks      = {};             ///< Dense block table
  block_id entry                             = invalid_block;  ///< Entry block
  block_id accept                            = invalid_block;  ///< Block containing acceptance
  std::uint32_t capture_count                = 0;              ///< Explicit capture count
  std::vector<replacement_token> replacement = {};             ///< Parsed replacement template
  ir_metrics metrics                         = {};             ///< Static graph metrics
};

/**
 * @brief Options controlling CUDA-oriented NVVM IR generation
 */
struct nvvm_ir_codegen_options {
  std::string symbol_prefix    = "regex_ir_generated";  ///< Prefix for internal symbols
  std::string execute_function = "regex_ir_execute";    ///< Public matcher function name
  bool prefix_filter : 1       = true;  ///< Enable ASCII-prefix filtering in the recursive fallback
  bool branch_hints  : 1       = true;  ///< Emit fallback `llvm.expect` branch hints
};

/**
 * @brief Validate an Instruction IR graph
 *
 * @param ir Instruction IR to validate
 * @return Diagnostics describing every detected invariant violation
 */
[[nodiscard]] std::vector<diagnostic> verify(instruction_ir const& ir);

/**
 * @brief Render Instruction IR as deterministic diagnostic text
 *
 * the returned text is intended for inspection and is not a serialization format.
 *
 * @param ir Instruction IR to render
 * @return Human-readable Instruction IR
 */
[[nodiscard]] std::string to_string(instruction_ir const& ir);

/**
 * @brief Measure static properties of an Instruction IR graph
 *
 * @param ir Instruction IR to inspect
 * @return Block, branch, predicate, read, capture, and literal metrics
 */
[[nodiscard]] ir_metrics measure(instruction_ir const& ir);

/**
 * @brief Generate NVVM IR for a boolean or capture-enumeration regex operation
 *
 * the returned module is self-contained device code with a public function named by
 * `options.execute_function` and prefixed internal symbols. NVVM IR uses LLVM syntax but follows
 * NVIDIA's stricter NVVM target contract. Assertion-free contains and matches graphs are
 * determinized into a bounded Unicode-class transition table; other valid boolean graphs use the
 * ordered fallback executor. Boolean functions have the ABI `i1(i8*, i64)`. Extract functions use
 * `i1(i8*, i64, i64, i64*)`, where the third argument is the first search byte and the final
 * argument receives begin/end pairs for the whole match followed by each explicit capture.
 *
 * @param ir Optimized boolean Instruction IR to render
 * @param options Symbol names and optimization hints
 * @return Textual NVVM IR accepted by libNVVM
 * @throw std::invalid_argument If the IR is invalid, a symbol is invalid, or the selected operation
 * does not produce a boolean or capture result
 */
[[nodiscard]] std::string generate_nvvm_ir(instruction_ir const& ir,
                                           nvvm_ir_codegen_options const& options = {});

}  // namespace regex_ir

// compiler API

namespace regex_ir {

/**
 * @brief Value-or-diagnostics return type used by compiler stages
 *
 * @tparam T Successful value type
 */
template <typename T>
struct result {
  std::optional<T> value              = {};  ///< Compiled value when the stage succeeds
  std::vector<diagnostic> diagnostics = {};  ///< Diagnostics emitted by the stage

  /**
   * @brief Check whether this result contains a compiled value
   *
   * @return true if compilation succeeded and `value` is populated
   */
  [[nodiscard]] explicit operator bool() const noexcept { return value.has_value(); }
};

/**
 * @brief Result of compiling a pattern to Automata IR
 */
using automata_result = result<automata_ir>;

/**
 * @brief Result of lowering or compiling Instruction IR
 */
using instruction_result = result<instruction_ir>;

/**
 * @brief Parse a regex and construct ordered Thompson Automata IR
 *
 * @param pattern Regex pattern encoded as UTF-8 source bytes
 * @param options Syntax, character-mode, and resource-limit options
 * @return Automata IR on success, otherwise structured diagnostics
 */
[[nodiscard]] automata_result compile_automata(std::string_view pattern,
                                               compile_options const& options = {});

/**
 * @brief Lower Automata IR to operation-specialized Instruction IR
 *
 * @param automata Verified Automata IR
 * @param selected Operation whose control and result shape should be encoded
 * @return Unoptimized Instruction IR on success, otherwise structured diagnostics
 */
[[nodiscard]] instruction_result lower(automata_ir const& automata, operation const& selected);

/**
 * @brief Optimize and verify Instruction IR
 *
 * @param ir Instruction IR to consume and optimize
 * @param options Optimization-pass configuration
 * @return Optimized Instruction IR on success, otherwise structured diagnostics
 */
[[nodiscard]] instruction_result optimize(instruction_ir ir,
                                          optimization_options const& options = {});

/**
 * @brief Compile a regex directly to optimized operation-specialized Instruction IR
 *
 * this convenience function performs parsing, Thompson construction, lowering,
 * optimization, and verifier checks.
 *
 * @param pattern Regex pattern encoded as UTF-8 source bytes
 * @param selected Operation whose matching and result policy should be encoded
 * @param options Syntax, character-mode, and resource-limit options
 * @param optimization Optimization-pass configuration
 * @return Optimized Instruction IR on success, otherwise structured diagnostics
 */
[[nodiscard]] instruction_result compile(std::string_view pattern,
                                         operation const& selected,
                                         compile_options const& options           = {},
                                         optimization_options const& optimization = {});

}  // namespace regex_ir

// host interpreter used by tests and CPU benchmarks

namespace regex_ir::testing {

/**
 * @brief Half-open byte span produced by the host interpreter
 */
struct match_span {
  std::size_t begin = 0;  ///< Inclusive first byte
  std::size_t end   = 0;  ///< Exclusive final byte

  /**
   * @brief Compare two byte spans
   *
   * @param lhs Left span
   * @param rhs Right span
   * @return true when both endpoints are equal
   */
  friend bool operator==(match_span const& lhs, match_span const& rhs)
  {
    return lhs.begin == rhs.begin && lhs.end == rhs.end;
  }
};

/**
 * @brief Operation result materialized by the host interpreter
 *
 * this type exists for correctness tests, fuzzing, and CPU comparisons. It is
 * not a production host regex runtime.
 */
struct execution_result {
  bool matched                                                        = false;  ///< Any match
  std::size_t count                                                   = 0;      ///< Match count
  std::vector<match_span> matches                                     = {};     ///< Match spans
  std::vector<std::optional<match_span>> captures                     = {};     ///< First captures
  std::vector<std::vector<std::optional<match_span>>> capture_matches = {};     ///< All captures
  std::vector<std::string> pieces                                     = {};     ///< Split fields
  std::string replaced                                                = {};  ///< Replacement result
};

/**
 * @brief Execute operation-specialized Instruction IR on the host
 *
 * @param ir Instruction IR to interpret
 * @param input UTF-8 or byte input selected by `ir.options`
 * @return Materialized operation result
 */
[[nodiscard]] execution_result execute(instruction_ir const& ir, std::string_view input);

/**
 * @brief Enumerate ordered non-overlapping matches and captures on the host
 *
 * @param ir Capture-preserving Instruction IR
 * @param input UTF-8 or byte input selected by `ir.options`
 * @return Every match and its capture spans
 */
[[nodiscard]] execution_result enumerate(instruction_ir const& ir, std::string_view input);

}  // namespace regex_ir::testing

// version

#define REGEX_IR_VERSION_MAJOR 0
#define REGEX_IR_VERSION_MINOR 1
#define REGEX_IR_VERSION_PATCH 0
