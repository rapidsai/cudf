## Summary

Add CP932 (Shift-JIS) encoding support for `df.to_csv()` and a binary string encoding option for the CSV writer.

### Changes

**Commit 1: String Encoding Option**
- Add `StringEncoding` enum (`UTF8`, `BINARY`) to `cudf/io/types.hpp`
- Add `encoding` option to `csv_writer_options`
- Implement `escape_strings_bytes_fn` for byte-level CSV escaping (preserves non-UTF-8 bytes)

**Commit 2: CP932 Encoding Support**
- Add `utf8_to_cp932()` GPU kernel for UTF-8 to CP932 conversion
- Add 7787-entry Unicode to CP932 mapping table with binary search lookup
- Add pylibcudf Cython bindings
- Support `df.to_csv(path, encoding="cp932")` in cudf Python API

## Motivation

cuDF currently assumes UTF-8 encoding for string data, and support for additional encodings has been requested (see #2957). Users working with non-UTF-8 datasets may encounter decoding failures (e.g., #2893).

While full multi-encoding support is complex and broader coverage will require additional design discussion, this PR provides a focused first step by adding CP932 support to improve real-world interoperability, particularly for Japanese enterprise systems that commonly use Shift-JIS encoding.

### Use Case

```python
import cudf

# Create DataFrame with Japanese text
df = cudf.DataFrame({
    "name": ["田中太郎", "鈴木花子"],
    "code": ["ABC123", "XYZ789"]
})

# Export as CP932 (Shift-JIS) encoded CSV
df.to_csv("output.csv", encoding="cp932", index=False)
```

## Test Plan

- [x] C++ unit tests for UTF-8 to CP932 conversion
  - ASCII passthrough
  - Hiragana, Katakana, Kanji conversion
  - Half-width Katakana (single-byte CP932)
  - Mixed content
  - Null/empty string handling
  - Emoji error handling (unsupported characters)
- [x] Python end-to-end tests (`TestCsvCp932Encoding` class)
- [x] Pre-commit checks pass (mypy, clang-format, ruff, etc.)

## Checklist

- [x] I am familiar with the [Contributing Guidelines](https://github.com/rapidsai/cudf/blob/HEAD/CONTRIBUTING.md)
- [x] New or existing tests cover these changes
- [x] The documentation is up to date
- [x] My code follows the style guidelines of this project

🤖 Generated with [Claude Code](https://claude.com/claude-code)
