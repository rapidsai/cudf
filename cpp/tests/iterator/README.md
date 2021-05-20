
# Iterator Test decomposition

The Iterator tests have been decomposed across different types to
make sure that no single test file takes too long to compile.

The decomposition is that each of the following
categorizes of types should be placed in a separate file:
 - numeric
 - chrono ( timestamp, duration )
 - fixed point ( numeric::decimal32, numeric::decimal64 )
 - string

The `numeric` and `chrono` type lists have roughly the same
number of entries allowing for a balanced compile time between
those two. We follow the same pattern for `fixed point` and
`string` so it is clear where to test those types, even though
they have a smaller set of entries and will compile quickly.
