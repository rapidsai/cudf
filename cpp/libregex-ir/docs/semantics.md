# Regex and Operation Semantics

## Characters

UTF-8 mode decodes valid scalar values and returns byte offsets. Invalid UTF-8 input is consumed as one byte by the test executor; downstream consumers must document equivalent behavior. Byte mode treats each byte as one character.

Case-insensitive mode adds the Unicode case counterpart selected by the
`C.UTF-8` locale. Predefined digit, word, and whitespace classes use ASCII by
default. Clearing `compile_options::ascii_classes` selects the compressed cuDF
Unicode tables through U+FFFF; non-word/non-digit complements continue through
the Unicode scalar range.

Dot excludes line feed by default. Extended-newline mode additionally excludes
carriage return, NEL, line separator, and paragraph separator. `dot_all`
accepts every scalar in either mode.

## Matching priority

Alternatives are ordered left to right. Greedy quantifiers prefer another repetition before exit. Lazy quantifiers prefer exit before another repetition. Search starts at each logical-character boundary from left to right.

## Assertions

`\A` and `\Z` refer to absolute input boundaries. `^` and `$` use line
boundaries when multiline is enabled; non-multiline `$` also accepts one final
configured terminator. Extended-newline mode treats CRLF as one terminator, so
neither anchor can match between its two characters. Word boundaries use the
configured ASCII or Unicode word class.

## Captures

Captures are numbered by opening parenthesis, starting at one. Group zero is the complete match. An unmatched optional capture is null. Repeated captures retain the final successful iteration.

## Operations

- matches requires a match spanning the complete input.
- contains reports whether any leftmost match exists.
- find reports the first leftmost match span.
- count counts non-overlapping leftmost matches.
- extract reports group zero and numbered capture spans for the first match.
- replace replaces every non-overlapping leftmost match. Replacement dollar-number expands a capture and double dollar emits one dollar.
- split emits text between every non-overlapping leftmost delimiter and preserves leading/trailing empty fields.

After a zero-length iterative match, searching resumes one logical character later unless already at end-of-input. Unmatched text is not discarded by replace or split.

## Unsupported syntax

Pattern backreferences, lookahead/lookbehind, recursion, conditionals,
subroutine calls, atomic groups, inline group extensions other than
non-capturing groups, and engine-specific verbs are rejected with diagnostics.
Three-digit octal escapes are accepted; shorter numeric escapes are rejected as
unsupported backreferences.
