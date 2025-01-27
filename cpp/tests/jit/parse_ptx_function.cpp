/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "jit/parser.hpp"

#include <cudf_test/testing_main.hpp>

#include <algorithm>
#include <cctype>

struct JitParseTest : public ::testing::Test {};

TEST_F(JitParseTest, PTXNoFunction)
{
  std::string raw_ptx = R"(
.visible .entry _ZN3cub17CUB_101702_750_NS11EmptyKernelIvEEvv()
{
  ret;
})";

  EXPECT_THROW(cudf::jit::parse_single_function_ptx(raw_ptx, "GENERIC_OP", "float", {0}),
               cudf::logic_error);
}

inline bool ptx_equal(std::string input, std::string expected)
{
  // Remove all whitespace and newline characters and compare
  // This allows us to handle things like excess newline characters
  // and trailing whitespace in the 'input'

  auto whitespace_or_newline = [](unsigned char c) { return std::isspace(c) || c == '\n'; };
  input.erase(std::remove_if(input.begin(), input.end(), whitespace_or_newline), input.end());
  expected.erase(std::remove_if(expected.begin(), expected.end(), whitespace_or_newline),
                 expected.end());
  return input == expected;
}

TEST_F(JitParseTest, SimplePTX)
{
  std::string raw_ptx = R"(
.visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Eff(
  .param .b64 _ZN8__main__7add$241Eff_param_0,
  .param .b32 _ZN8__main__7add$241Eff_param_1,
  .param .b32 _ZN8__main__7add$241Eff_param_2
)
{
  ret;
}
)";

  std::string expected = R"(
__device__ __inline__ void GENERIC_OP(
  float* _ZN8__main__7add_241Eff_param_0,
  int _ZN8__main__7add_241Eff_param_1,
  int _ZN8__main__7add_241Eff_param_2
){
 asm volatile ("{");
 asm volatile ("bra RETTGT;");
 asm volatile ("RETTGT:}");}
)";

  std::string cuda_source =
    cudf::jit::parse_single_function_ptx(raw_ptx, "GENERIC_OP", "float", {0});

  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

TEST_F(JitParseTest, PTXWithPragma)
{
  std::string raw_ptx = R"(
.visible .func _ZN3cub17CUB_101702_750_NS11EmptyKernelIvEEvv()
{
$L__BB0_151:
  .pragma "nounroll";
  mov.u32 % r1517, % r1516;
  mov.u32 % r1516, % r1515;
  mov.u32 % r1515, % r1505;
  mov.u32 % r1457, 0;
$L__BB0_152:
  .pragma "nounroll";
})";

  std::string expected = R"(
__device__ __inline__ void EmptyKern(){
 asm volatile ("{");  asm volatile (" $L__BB0_151:  .pragma \"nounroll\";");
   /**   $L__BB0_151:
  .pragma "nounroll"  */

  asm volatile ("  mov.u32 _ r1517, _ r1516;");
   /**   mov.u32 % r1517, % r1516  */

  asm volatile ("  mov.u32 _ r1516, _ r1515;");
   /**   mov.u32 % r1516, % r1515  */

  asm volatile ("  mov.u32 _ r1515, _ r1505;");
   /**   mov.u32 % r1515, % r1505  */

  asm volatile ("  mov.u32 _ r1457, 0;");
   /**   mov.u32 % r1457, 0  */

  asm volatile (" $L__BB0_152:  .pragma \"nounroll\";");
   /**   $L__BB0_152:
  .pragma "nounroll"  */

 asm volatile ("RETTGT:}");}
)";

  std::string cuda_source = cudf::jit::parse_single_function_ptx(raw_ptx, "EmptyKern", "void", {0});
  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

TEST_F(JitParseTest, PTXWithPragmaWithSpaces)
{
  std::string raw_ptx = R"(
.visible .func _ZN3cub17CUB_101702_750_NS11EmptyKernelIvEEvv()
{
  $L__BB0_58:
    ld.param.u32 % r1419, [% rd419 + 80];
    setp.ne.s32 % p394, % r1419, 22;
    mov.u32 % r2050, 0;
    mov.u32 % r2048, % r2050;
    @ % p394 bra $L__BB0_380;

    ld.param.u8 % rs1369, [% rd419 + 208];
    setp.eq.s16 % p395, % rs1369, 0;
    selp.b32 % r1422, % r1925, 0, % p395;
    ld.param.u32 % r1423, [% rd419 + 112];
    add.s32 % r427, % r1422, % r1423;
    ld.param.u64 % rd1249, [% rd419 + 120];
    cvta.to.global.u64 % rd1250, % rd1249;
    .pragma "used_bytes_mask 4095";
    ld.global.v4.u32{ % r1424, % r1425, % r1426, % r1427}, [% rd1250];
    ld.global.v2.u64{ % rd1251, % rd1252}, [% rd1250 + 16];
    ld.global.s32 % rd230, [% rd1250 + 32];
    setp.gt.s32 % p396, % r1424, 6;
    @ % p396 bra $L__BB0_376;
}
}
)";

  std::string expected = R"(
__device__ __inline__ void LongKernel(){
 asm volatile ("{");  asm volatile (" $L__BB0_58:  cvt.u32.u32 _  %0, [_ rd419 + 80];": : "r"(r1419));
   /**   $L__BB0_58:
    ld.param.u32 % r1419, [% rd419 + 80]  */

  asm volatile ("  setp.ne.s32 _ p394, _ r1419, 22;");
   /**   setp.ne.s32 % p394, % r1419, 22  */

  asm volatile ("  mov.u32 _ r2050, 0;");
   /**   mov.u32 % r2050, 0  */

  asm volatile ("  mov.u32 _ r2048, _ r2050;");
   /**   mov.u32 % r2048, % r2050  */

  asm volatile ("  @ _ p394 bra $L__BB0_380;");
   /**   @ % p394 bra $L__BB0_380  */

  asm volatile ("  cvt.u8.u8 _  %0, [_ rd419 + 208];": : "h"( static_cast<short>(rs1369)));
   /**   ld.param.u8 % rs1369, [% rd419 + 208]  */

  asm volatile ("  setp.eq.s16 _ p395, _ rs1369, 0;");
   /**   setp.eq.s16 % p395, % rs1369, 0  */

  asm volatile ("  selp.b32 _ r1422, _ r1925, 0, _ p395;");
   /**   selp.b32 % r1422, % r1925, 0, % p395  */

  asm volatile ("  cvt.u32.u32 _  %0, [_ rd419 + 112];": : "r"(r1423));
   /**   ld.param.u32 % r1423, [% rd419 + 112]  */

  asm volatile ("  add.s32 _ r427, _ r1422, _ r1423;");
   /**   add.s32 % r427, % r1422, % r1423  */

  asm volatile ("  mov.u64 _  %0, [_ rd419 + 120];": : "l"(rd1249));
   /**   ld.param.u64 % rd1249, [% rd419 + 120]  */

  asm volatile ("  cvta.to.global.u64 _ rd1250, _ rd1249;");
   /**   cvta.to.global.u64 % rd1250, % rd1249  */

  asm volatile ("  .pragma \"used_bytes_mask 4095\";");
   /**   .pragma "used_bytes_mask 4095"  */

  asm volatile ("  ld.global.v4.u32{ _ r1424, _ r1425, _ r1426, _ r1427}, [_ rd1250];");
   /**   ld.global.v4.u32{ % r1424, % r1425, % r1426, % r1427}, [% rd1250]  */

  asm volatile ("  ld.global.v2.u64{ _ rd1251, _ rd1252}, [_ rd1250 + 16];");
   /**   ld.global.v2.u64{ % rd1251, % rd1252}, [% rd1250 + 16]  */

  asm volatile ("  ld.global.s32 _ rd230, [_ rd1250 + 32];");
   /**   ld.global.s32 % rd230, [% rd1250 + 32]  */

  asm volatile ("  setp.gt.s32 _ p396, _ r1424, 6;");
   /**   setp.gt.s32 % p396, % r1424, 6  */

  asm volatile ("  @ _ p396 bra $L__BB0_376;");
   /**   @ % p396 bra $L__BB0_376  */

  asm volatile ("RETTGT:}");}
 )";

  std::string cuda_source =
    cudf::jit::parse_single_function_ptx(raw_ptx, "LongKernel", "void", {0});
  EXPECT_TRUE(ptx_equal(cuda_source, expected));
}

CUDF_TEST_PROGRAM_MAIN()
