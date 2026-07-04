# Versioning Policy

regex_ir uses semantic versioning for the installed C++ API and CMake package.

Before 1.0, minor releases may make source-incompatible changes to the in-memory IR when documented in CHANGELOG.md. Patch releases preserve source compatibility and only fix behavior or diagnostics.

The deterministic text printer is diagnostic output, not a stable serialization format. No durable binary or textual IR interchange format is promised in 0.1. A future serialized schema will carry an independent schema version and compatibility rules.

The exported CMake target is `regex_ir::regex_ir` and the package configuration
name is `regex_ir`.
