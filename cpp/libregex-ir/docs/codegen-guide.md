# Generating CUDA from Optimized Instruction IR

This guide shows how a downstream consumer turns the final regex_ir
Instruction IR into specialized device code. The library includes the public
`generate_nvvm_ir()` correctness-first renderer. Use this guide when extending
that path or building a target-specific CUDA renderer with a different ABI or
control-flow strategy. Your integration still owns compiler invocation,
input/output storage, kernel launch policy, and allocation.

For an end-to-end introduction, including the target function ABI and how to
compile and execute generated CUDA device code, start with the
[usage guide](usage.md). This document focuses on renderer design and
optimization details.

The examples use C++-like pseudocode that can be rendered as CUDA device code
or another target language.

## 1. Compile and validate the final IR

The compile convenience function performs all three stages and returns optimized Instruction IR:

    #include <regex_ir.hpp>

    auto compiled = regex_ir::compile(
      "abc[0-9]+",
      regex_ir::operation::contains());

    if (!compiled) {
      for (auto& diagnostic : compiled.diagnostics) {
        report(diagnostic.span, diagnostic.message);
      }
      return;
    }

    auto& ir = *compiled.value;
    auto validation = regex_ir::verify(ir);
    if (!validation.empty()) {
      throw codegen_error("invalid Instruction IR");
    }

Generate from instruction_ir, not automata_ir. The Automata IR is useful for analysis and debugging, while Instruction IR already contains fused literals, removed epsilon plumbing, capture specialization, operation behavior, and final metrics.

The IR is pre-1.0 in-memory data. Generate and compile code in the same process
or pin the Regex IR version. The text printer is diagnostic output, not a
serialization format.

## 2. Runtime ABI required by generated code

Define a small target-side stream/cursor abstraction. A practical contract is:

    struct cursor {
      byte_pointer data;
      size_type byte_position;
      size_type byte_end;

      bool can_peek(uint32_t logical_characters) const;
      char32_t peek() const;
      void advance(uint32_t logical_characters);
      bool at_begin() const;
      bool at_end() const;
      bool previous_is_newline() const;
      bool current_is_newline() const;
      bool previous_is_word() const;
      bool current_is_word() const;
    };

Important requirements:

- byte_position is the externally visible offset used by match and capture spans;
- can_peek and advance count logical characters, not bytes, in UTF-8 mode;
- peek returns the current decoded code point without advancing;
- byte mode treats one byte as one character;
- invalid UTF-8 must follow the behavior chosen by your integration; the test executor consumes an invalid leading byte as one character;
- current/previous word tests use the IR's configured ASCII or Unicode word definition;
- cursor copies must be inexpensive because ambiguous branches require transactional cursor state.

For CUDA, make the cursor trivially copyable and annotate its methods for device use. Keep the input pointer and byte offsets in registers. Avoid storing decoded strings or per-character heap state.

## 3. Understand the block graph

instruction_ir contains:

- blocks: dense blocks addressed by block_id;
- entry: the block used by one match attempt;
- accept: the terminal accepting block;
- control: the outer search/iteration/result plan;
- replacement: parsed replacement tokens for replace mode;
- capture_count: numbered capture groups, excluding group zero;
- metrics: static code-generation cost proxies.

Each instruction_block has an ordered instruction list and zero or more successors.

Successors must be attempted by ascending block_edge.priority. Do not rely on their storage order. Priority preserves:

- left-to-right alternation;
- greedy repetition, which prefers the repetition edge;
- lazy repetition, which prefers the exit edge.

A block is transactional. If its instructions or a successor fail, cursor and capture changes made along that path must not leak into the next lower-priority successor.

Nullable repeated expressions can create a cycle that returns to a block without advancing the cursor. During graph analysis, identify strongly connected components with a path that consumes no character. The generated matcher must suppress re-entering the same no-progress component at the same byte offset on one active path. Preserve the first priority-ordered visit, then try its exit continuation. This is separate from operation_control.advance_after_empty, which governs progress between completed matches.

## 4. Map each instruction

Emit instructions in their stored order.

| Instruction | Generated behavior | Failure behavior |
| --- | --- | --- |
| can_peek N | Check that N logical characters are available. | Fail the current path. |
| read_character | Load cursor.peek() into the block's current-character temporary. | Cannot fail after a matching can_peek. |
| match_character P | Evaluate the normalized predicate P against the current character. | Fail the current path. |
| match_literal S | Compare S with consecutive logical characters at the cursor without advancing. | Fail the current path. |
| advance_cursor N | Advance by N logical characters. | Cannot fail after the preceding bounds check. |
| test_assertion A | Evaluate a zero-width assertion at the cursor. | Fail the current path. |
| write_capture A,I | Store the current byte offset as capture I begin or end. | Cannot fail. |
| emit_accept | Produce success at the current byte offset. | Reject if control.require_end and the cursor is not at end. |

Do not advance inside match_character or match_literal. The following advance_cursor is the sole cursor mutation for a consuming block.

### Character predicates

For `predicate_class::ANY`, accept every character when `matches_newline` is
true. Otherwise exclude line feed in default mode or CR, LF, NEL, LS, and PS
when `extended_newline` is true.

For all other predicates, ranges are normalized, sorted, and non-overlapping. Generate:

    bool contained =
      (cp >= first_0 && cp <= last_0) ||
      (cp >= first_1 && cp <= last_1) ||
      ...;

    bool matched = predicate.negated ? !contained : contained;

Small range sets should become inline comparisons. For larger sets, a target
may choose binary search, an ASCII bitmap, or constant-memory tables.
Case-insensitive alternatives were already expanded by the compiler; the
renderer does not perform additional case folding.

### Assertions

Generate assertion_kind as follows:

- begin_input: byte offset is zero, or multiline is enabled and the previous character is a newline;
- end_input: byte offset is byte_end, or multiline is enabled and the current character is a newline;
- word_boundary: previous_is_word differs from current_is_word;
- not_word_boundary: previous_is_word equals current_is_word.

Assertions never consume input.

### Captures

Allocate capture_count + 1 spans. Index zero is the complete match and is managed by the operation wrapper. write_capture only addresses numbered groups.

A capture slot needs a begin offset, end offset, and matched flag. An optional group that did not participate remains unmatched. When a repeated capture succeeds several times, the final successful path state wins.

## 5. Emit one match attempt

Generate a specialized try_match_at function for the pattern. Its conceptual signature is:

    match_result try_match_at(input_view input, size_type start);

The function:

1. initializes cursor.byte_position to start;
2. initializes capture zero begin to start;
3. enters ir.entry;
4. executes block instructions;
5. tries successors in ascending priority;
6. restores cursor and captures before trying a lower-priority successor;
7. prevents a nullable cycle from re-entering at the same byte offset;
8. on emit_accept, checks control.require_end;
9. sets capture zero end and returns success.

A correctness-first block emitter can generate one target function per IR block:

    bool block_12(match_context& ctx) {
      if (!ctx.cursor.can_peek(3)) return false;
      if (!equals_literal(ctx.cursor, U"abc")) return false;
      ctx.cursor.advance(3);

      match_context saved = ctx;
      if (block_18(ctx)) return true;
      ctx = saved;
      return block_27(ctx);
    }

Generate forward declarations for every block before definitions. A zero-successor non-accepting block returns false. A one-successor block directly calls that target. A multiple-successor block snapshots context and tries targets in priority order.

This simple form is useful for validating a renderer, but literal recursion is not the preferred CUDA form for loops.

## 6. Emit GPU-friendly control flow

For boolean contains and matches, first determine whether captures and path
priority are unobservable and assertions are absent. In that case, determinize
the NFA at generation time and partition Unicode by predicate-equivalence. A
single-pass state/class transition table removes retry state, recursion, and
per-start scanning. The built-in NVVM renderer uses this form with bounded
state and table growth, falling back to the ordered graph when it is unsafe.

Also consider a Glushkov position plan for a non-nullable graph with at most 64
consuming positions. Give each position one bit, precompute the first and
accepting masks, group common positive follow deltas into masked shifts, and
map alphabet classes to position reach masks. Then a scanning boolean step is
`reach(character) & (first | follow(active))`. This processes every possible
start in one pass with no per-thread state array. Do not apply that
simplification to an API that observes the winning start, path priority, or
captures without a separate correctness construction for those values. The
built-in renderer uses a measured cost gate because exception-heavy follow
graphs can be slower than a compact DFA.

For the ordered fallback, analyze the block graph before printing:

1. Compute predecessors and strongly connected components.
2. Merge linear one-predecessor/one-successor block chains.
3. Turn reducible cyclic components into while or do-while loops.
4. Clone small blocks when doing so removes a join or dynamic continuation.
5. Emit priority branches as direct if/else control flow.
6. Keep a small explicit alternative stack only for irreducible ambiguous paths.
7. Store only the live cursor and capture slots in each alternative entry.

An alternative stack entry conceptually contains:

    struct alternative {
      continuation_id continuation;
      size_type byte_position;
      live_capture_snapshot captures;
    };

continuation_id is generated per pattern, not a generic regex opcode. Dispatch covers only statically known continuation labels. The preferred path runs directly; the lower-priority continuation is pushed before entering it.

Use the metrics in ir.metrics to choose between code shapes:

- many blocks or branches may justify limited continuation dispatch rather than aggressive cloning;
- high literal_codepoints favors wide or vectorized byte comparisons on ASCII-safe literals;
- high capture_writes suggests capture liveness analysis before snapshotting;
- high stream_reads suggests combining decoding and predicate checks.

Never drop or reorder priority edges for result shapes that expose a preferred
match or captures. Boolean existence is the exception: any accepting path has
the same observable result, which permits deterministic language-equivalent
lowering.

The built-in NVVM renderer marks small leaf helpers with supported
`alwaysinline`, `readonly`, `readnone`, and `nounwind` attributes. It routes all
input-byte reads through one helper and emits ordinary loads so the CUDA
toolchain selects cache behavior.

## 7. Generate the operation wrapper

try_match_at performs one attempt. operation_control defines how to call it.

| selected operation | scan_input | require_end | first_only | result |
| --- | ---: | ---: | ---: | --- |
| matches | false | true | true | boolean |
| contains | true | false | true | boolean |
| find | true | false | true | match_span |
| extract | true | false | true | captures |
| count | true | false | false | match_count |
| replace | true | false | false | replacement |
| split | true | false | false | split_fields |

### Anchored attempt

When scan_input is false, invoke try_match_at only at byte offset zero. matches additionally succeeds only when emit_accept reaches end-of-input because require_end is true.

### Scanning attempt

When scan_input is true, try each logical-character boundary from left to right, including end-of-input:

    position = 0;
    while (position <= input.byte_size()) {
      result = try_match_at(input, position);
      if (result.matched) {
        consume_result(result);
        if (control.first_only) break;
        position = result.end;
        if (result.end == result.begin && control.advance_after_empty) {
          if (position == input.byte_size()) break;
          position = advance_one_logical_character(input, position);
        }
      } else {
        if (position == input.byte_size()) break;
        position = advance_one_logical_character(input, position);
      }
    }

Before emitting that general loop, inspect the entry block for a required
ASCII literal or singleton that occurs before any cursor advance or accept. A
contains wrapper may compare that byte first and invoke `try_match_at` only at
candidate positions. The built-in renderer does this when `prefix_filter` is
enabled, uses `llvm.expect` when `branch_hints` is enabled, advances ASCII
non-candidates with one integer add, and retains the end-of-input case for
nullable patterns. Do not derive a filter from one arm of an alternation or an
optional prefix; it must be required on every path entering the match.

For count, replace, and split, successful matches are non-overlapping. After a non-empty match, resume at match.end. After an empty match, apply advance_after_empty to prevent an infinite loop.

### Boolean, span, count, and capture results

- boolean returns whether a match was found;
- match_span returns capture zero begin/end;
- match_count increments for every successful non-overlapping match;
- captures returns capture zero plus numbered capture spans from the first match.

### Replacement

instruction_ir.replacement contains literal and capture tokens. For each match:

1. copy unmatched input from the previous output cursor to capture zero begin;
2. append literal tokens directly;
3. append a capture token's byte slice only when that capture participated;
4. set the output cursor to capture zero end;
5. after iteration, copy the remaining input suffix.

Capture token zero references the complete match.

For a GPU two-pass implementation, first compute each row's output byte size, exclusive-scan the sizes, allocate once, then run the write pass.

### Split

For each delimiter match, emit the byte slice from the previous output cursor to capture zero begin, then move the cursor to capture zero end. Emit the final suffix after iteration. Preserve zero-length leading, interior, and trailing fields according to docs/semantics.md.

A columnar GPU implementation normally computes field counts and byte sizes first, scans offsets, then writes child-string data and list offsets.

## 8. Example: fused final IR to direct code

For pattern abc[0-9]+ in contains mode, the optimizer fuses abc into one match_literal. A simplified target can become:

    device_match try_match_at(input_view input, size_type start) {
      cursor cur{input, start};

      if (!cur.can_peek(3)) return no_match;
      if (!equals_literal(cur, U"abc")) return no_match;
      cur.advance(3);

      if (!cur.can_peek(1)) return no_match;
      char32_t cp = cur.peek();
      if (!(cp >= U'0' && cp <= U'9')) return no_match;
      cur.advance(1);

      while (cur.can_peek(1)) {
        cp = cur.peek();
        if (!(cp >= U'0' && cp <= U'9')) break;
        cur.advance(1);
      }

      return matched_span{start, cur.byte_position};
    }

The contains wrapper filters by the required leading `a` byte, calls this
attempt at candidate logical-character boundaries, and returns true after the
first success.

A suffix after the plus quantifier can require backtracking. Do not always lower a prioritized repetition to a possessive while loop. Preserve its lower-priority exit continuation so a pattern such as [0-9]+7 can give characters back when necessary.

## 9. Minimal custom CUDA renderer outline

Regex IR does not ship this source renderer; this outline is for integrations
that choose to emit CUDA C++ instead of using `generate_nvvm_ir()`. Such a
renderer typically has these phases:

    class source_renderer {
     public:
      std::string render(regex_ir::instruction_ir const& ir) {
        require_valid(ir);
        analyze_graph(ir);
        emit_preamble_and_cursor_abi(ir.options);
        emit_predicate_helpers(ir);
        emit_block_declarations(ir);
        emit_try_match_body(ir);
        emit_operation_wrapper(ir.control, ir.replacement);
        return output_;
      }

     private:
      void emit_instruction(regex_ir::instruction const& instruction) {
        std::visit([&](auto& value) { emit(value); }, instruction);
      }

      void emit(regex_ir::can_peek const&);
      void emit(regex_ir::read_character const&);
      void emit(regex_ir::match_character const&);
      void emit(regex_ir::match_literal const&);
      void emit(regex_ir::advance_cursor const&);
      void emit(regex_ir::test_assertion const&);
      void emit(regex_ir::write_capture const&);
      void emit(regex_ir::emit_accept const&);

      std::string output_;
    };

Keep target-specific spelling outside IR analysis. The same analyzed graph can
feed the NVVM renderer, a custom CUDA renderer, a debug pseudocode printer, or
a cost model.

## 10. CUDA integration checklist

Before compiling generated CUDA:

- verify the IR;
- enforce a per-pattern generated-source/code-size budget;
- select size_type wide enough for input byte offsets;
- define invalid UTF-8 behavior consistently;
- keep literal data in immediate values or suitable constant storage;
- avoid heap allocation and unbounded device recursion;
- bound any ambiguity/continuation stack and define overflow behavior;
- snapshot only live captures at branches;
- preserve priority and zero-length progress;
- separate size and write passes for variable-length replace/split output;
- batch many strings per launch and measure divergence, registers, occupancy, and bandwidth on representative data;
- cache compiled modules by pattern, compile options, operation, optimizer
  options, target architecture, and Regex IR version.

## 11. Common code-generation errors

- Treating successor vector order as priority order.
- Advancing during match_literal and again during advance_cursor.
- Counting UTF-8 bytes instead of logical characters in can_peek.
- Returning character indices instead of byte offsets.
- Forgetting to restore captures when a prioritized path fails.
- Turning greedy repetition into possessive repetition.
- Omitting the end-of-input attempt for nullable patterns.
- Looping inside a nullable repetition or after a zero-length count, replace, or split match.
- Ignoring control.require_end for matches.
- Applying replacement capture tokens to unmatched groups.
- Generating from unverified or non-optimized IR.
