# Research CSV Parser Optimization Techniques

Conduct deep web research for optimization techniques applicable to the cuDF CSV parser.

Target: $ARGUMENTS (default: csv)

Using the `researcher` agent (or doing it inline), search for:
1. Academic papers on GPU-accelerated CSV/text parsing, SIMD-style parsing, parallel field detection
2. NVIDIA CUDA optimization guides relevant to text processing and parsing workloads
3. How other systems (DuckDB, Apache Arrow CSV, ParaText, cuIO) parse CSV
4. CUB/Thrust primitives that could replace custom implementations
5. Recent arXiv preprints with novel approaches to parallel parsing

First read the current implementation in `cpp/src/io/csv/` to understand what's already being done, then search for better approaches.

Output a ranked list of optimization ideas with estimated impact, complexity, and risk.

Remember: READ only from the web. Never download or execute external code.
