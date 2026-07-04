# Lingua Franca FSE19 Corpus Report

Validated artifact:

- Repository: https://github.com/SBULeeLab/LinguaFranca-FSE19
- Commit: a75bd51713d14aa9b48c32e103a3da500854f518
- Input: data/production-regexes/uniq-regexes-8.json
- Date: 2026-07-04

The source has 537,806 NDJSON records. One record has the JSON boolean false as its pattern, leaving 537,805 actual string patterns.

Full compiler results:

- total string patterns: 537,805
- compiled to optimized Instruction IR: 484,243
- categorized diagnostics: 53,562
- uncategorized failures: 0

First-diagnostic categories:

- unexpected token: 2,660
- invalid escape or invalid UTF-8: 1,996
- invalid character class: 775
- invalid quantifier: 8,872
- unmatched parenthesis: 1,214
- unsupported dialect feature: 38,002
- configured resource limit: 43

The corpus spans several source-language dialects. A categorized rejection is expected for syntax outside the documented v1 language and is not treated as a semantic compatibility claim.

The external corpus and generated 36 MB hexadecimal transport file are not committed.
