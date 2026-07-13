# Regex IR

Regex IR is the C++20 regex compiler and code generator vendored by cuDF. Its
single entry point, `regex_ir::compile`, parses a regular expression and emits
operation-specific NVVM IR. All intermediate representations and compiler
passes are private implementation details.

This directory intentionally contains only the production sources and design
notes:

- `regex_ir.hpp`: minimal public compile API
- `regex_ir_detail.hpp`: private compiler declarations
- `regex_ir.cpp`: implementation and embedded Unicode range tables
- `optimization.md`: optimization design notes

The code is integrated, built, and tested through the enclosing cuDF project;
it is not maintained as a standalone CMake project.
