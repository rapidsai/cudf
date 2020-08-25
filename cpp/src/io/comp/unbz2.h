/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 *
 * bzip2 license information is available at
 * https://spdx.org/licenses/bzip2-1.0.6.html
 * https://github.com/asimonov-im/bzip2/blob/master/LICENSE
 * original source code available at
 * http://www.sourceware.org/bzip2/
 *
 */

/*--

Copyright (C) 1996-2002 Julian R Seward.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. The origin of this software must not be misrepresented; you must
not claim that you wrote the original software.  If you use this
software in a product, an acknowledgment in the product
documentation would be appreciated but is not required.

3. Altered source versions must be plainly marked as such, and must
not be misrepresented as being the original software.

4. The name of the author may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Julian Seward, Cambridge, UK.
jseward@acm.org
bzip2/libbzip2 version 1.0 of 21 March 2000

This program is based on (at least) the work of:
Mike Burrows
David Wheeler
Peter Fenwick
Alistair Moffat
Radford Neal
Ian H. Witten
Robert Sedgewick
Jon L. Bentley

For more information on these sources, see the manual.
--*/

#pragma once

namespace cudf {
namespace io {
// If bz_outbuff_full is returned and block_start is non-NULL, dstlen will be updated to point to
// the end of the last valid block, and block_start will contain the offset in bits of the beginning
// of the block, so it can be passed in to resume decoding later on.
constexpr int bz_ok = 0;
constexpr int bz_stream_end = 4;
constexpr int bz_param_error = -2;
constexpr int bz_mem_error = -3;
constexpr int bz_data_error = -4;
constexpr int bz_data_error_magic = -5;
constexpr int bz_unexpected_eof = -7;
constexpr int bz_outbuff_full = -8;

int32_t cpu_bz2_uncompress(const uint8_t *input,
                           size_t inlen,
                           uint8_t *dst,
                           size_t *dstlen,
                           uint64_t *block_start = nullptr);

}  // namespace io
}  // namespace cudf
