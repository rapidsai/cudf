# IR Contract

## Automata IR

Automata IR contains immutable-by-convention dense states:

- jump: one epsilon successor;
- branch: two or more priority-ordered epsilon successors;
- consume: a normalized character predicate and one successor;
- assertion: one zero-width test and one successor;
- capture: one start/end tag action and one successor;
- accept: terminal success.

Every state stores its source span. The verifier checks IDs, edge targets, state arity, accept termination, and capture indices.

## Instruction IR

Instruction IR contains blocks with typed instruction variants and priority-ordered successors:

- can_peek;
- read_character;
- match_character or match_literal;
- advance_cursor;
- test_assertion;
- write_capture;
- emit_accept.

Blocks and successors are structured data, not runtime bytecode. A renderer visits variants and emits its target language directly.

Each Instruction IR also carries operation_control: whether entry scans or is anchored, whether acceptance must reach end-of-input, whether matching stops at the first result or iterates non-overlapping results, the zero-length progress rule, and the output result shape.

## Optimization

The implemented passes:

1. remove capture writes for modes that do not observe captures;
2. resolve chains of empty one-successor blocks;
3. fuse single-character chains into match_literal;
4. remove unreachable blocks;
5. recompute metrics and verify.

All passes preserve priority edges and operation metadata. A caller can disable each category through optimization_options.

## Stability

The in-memory API is pre-1.0. No serialized schema is promised yet. Deterministic text printers are intended for diagnostics and golden testing, not durable interchange.

See the [usage guide](usage.md) for the complete target runtime and execution
contract. The [code-generation guide](codegen-guide.md) covers additional graph
shaping and CUDA optimization guidance.
