# Architecture

## Scope

The core library compiles IR and provides a correctness-first NVVM IR text
renderer. A downstream project still owns compiler
invocation, input/output storage, module loading, kernel launch policy, and
allocation. The CPU executor under tests is not installed. The optional GPU
benchmark owns those runtime pieces only for its cuDF comparison workload.

## Stages

1. The parser validates and normalizes the supported pattern language.
2. Ordered Thompson construction creates Automata IR with consuming, branch, assertion, capture, jump, and accept states.
3. Lowering creates one typed structured-control-flow block per Automata state.
4. Optimization specializes captures, bypasses empty jumps, fuses literal chains, and removes unreachable blocks.
5. Both IR levels are verified before being returned.
6. For assertion-free boolean programs, the NVVM generator determinizes the
   graph and partitions Unicode into predicate-equivalent alphabet classes.
   Other programs retain ordered graph lowering.
7. The selected executor is rendered directly to NVVM IR without a runtime
   regex opcode interpreter.

The Automata representation is Thompson rather than the initially considered Glushkov position NFA. It has linear construction size and directly preserves capture tags, embedded assertions, empty paths, and greedy/lazy branch order. Instruction lowering and optimization remove unnecessary epsilon plumbing from the consumer-facing form.

## Safety and determinism

- IDs are stable storage indices.
- Printers are deterministic.
- Pattern, nesting, repeat, state, transition, and capture limits are configurable.
- Failure returns diagnostics and no partial IR.
- Each optimization runs between verifier checks.
- Unsupported dialect constructs are rejected explicitly.

## GPU orientation

Instruction IR expresses bounded peeks, character predicates, literal windows,
cursor advancement, branches, assertions, capture writes, and acceptance.
Static metrics expose block, branch, predicate, read, capture-write, and literal
sizes. For assertion-free contains and matches, the NVVM renderer emits a
single-pass constant-table DFA because captures and priority are unobservable.
The ordered fallback retains required-prefix filtering, direct ASCII-literal
lowering, optimizer attributes, and branch hints. Both paths use ordinary input
loads so the CUDA compiler selects cache behavior.

Actual occupancy, register pressure, divergence, and bandwidth require
representative hardware. The optional `regex-ir-gpu-benchmark` reports
warm/cold latency and throughput for the built-in NVVM path beside
`cudf::strings::contains_re`; it deliberately makes no hardware-independent
performance claim.
