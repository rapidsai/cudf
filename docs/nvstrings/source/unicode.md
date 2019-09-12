# nvstrings Unicode Features

The nvstrings currently supports only 16-bit Unicode character code-points (0-65535)
for functions that require character testing (e.g. isalpha(), isdecimal(), etc) and for
case conversion (e.g. lower(), upper(), swapcase(), capitalize()).

Case conversions that are context-sensitive or that transform between multiple characters
and single characters are currently not supported.