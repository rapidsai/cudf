# Third-party notices

## cuDF regex test and benchmark data

The `Cudf` unit-test fixture and CPU/GPU benchmark workloads are adapted to the
Regex IR API from NVIDIA RAPIDS cuDF's C++ regex tests and benchmarks at commit
`3652e808337cfd3b63591102398162404d3755a6`:
https://github.com/rapidsai/cudf/tree/3652e808337cfd3b63591102398162404d3755a6/cpp

The source test files carry the following notice:

> Copyright (c) 2019-2026, NVIDIA CORPORATION.
>
> SPDX-License-Identifier: Apache-2.0

cuDF is distributed under the Apache License 2.0, the same license included in
this component's `LICENSE` file.

## RE2 test and benchmark data

The `Re2` fixture contains cases adapted to the Regex IR API from the RE2 test
suite, and the CPU benchmark links RE2 as its comparison engine:
https://github.com/google/re2/tree/main/re2/testing

RE2 is distributed under the following license:

> Copyright (c) 2009 The RE2 Authors. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of Google Inc. nor the names of its contributors may be
>   used to endorse or promote products derived from this software without
>   specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## Rust regex test data

The `RustRegex` fixture contains cases adapted to the Regex IR API from the Rust
`regex` crate test data:
https://github.com/rust-lang/regex/tree/master/testdata

The Rust `regex` crate is distributed under the following MIT license:

> Copyright (c) 2014 The Rust Project Developers
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## External performance workload inventories

The variable-length GPU corpus driver adapts regular-expression inventories
and downloads checksum-pinned copies of the complete source datasets from:

- OpenResty's 31-case Regex Engine Matching Speed Benchmark at SRegex commit
  `ee5944f108f76217760f9b5548e842971c0281b2`:
  https://openresty.org/misc/re/bench/
- rust-leipzig/regex-performance commit
  `52cb0538eca86ad549f6895dbfa9a2f71bc82244`, distributed under Apache License
  2.0:
  https://github.com/rust-leipzig/regex-performance
- Boost.Regex's GCC performance comparison, distributed under Boost Software
  License 1.0:
  https://www.boost.org/doc/libs/1_41_0/libs/regex/doc/gcc-performance.html
- mariomka/regex-benchmark commit
  `17d073ec864931546e2694783f6231e4696a9ed4`, distributed under the MIT
  License:
  https://github.com/mariomka/regex-benchmark

The downloaded data is kept in the build directory, not vendored. OpenResty's
`abc.txt`, `rand-abc.txt`, and `delim.txt` are generated in memory from the
full-size upstream recipes; those recipes intentionally left the random seed
unspecified, so this project uses a fixed deterministic generator. The Mark
Twain texts are Project Gutenberg works whose use is subject to the notice
included in each source file.

## CPython and sihlfall regex cases

The `CPython` fixture adapts regular-language cases from CPython's `re_tests`
suite at commit `8e88bb56337a771f3af5edc5c3a5ba96ea2c3353`:
https://github.com/python/cpython/blob/8e88bb56337a771f3af5edc5c3a5ba96ea2c3353/Lib/test/re_tests.py

CPython is distributed under the Python Software Foundation License Version 2.

The `Sihlfall` fixture adapts categories and cases from regex-test-cases at
commit `0ab3381ad388eaa44576fe33fa66aa3458c61c43`:
https://github.com/sihlfall/regex-test-cases/tree/0ab3381ad388eaa44576fe33fa66aa3458c61c43/test-cases

regex-test-cases is distributed under the 3-clause BSD license included in its
repository.
