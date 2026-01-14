/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;

public class Aggregation128UtilsTest extends CudfTestBase {
  @Test
  public void testExtractInt32Chunks() {
    BigInteger[] intvals = new BigInteger[] {
        null,
        new BigInteger("123456789abcdef0f0debc9a78563412", 16),
        new BigInteger("123456789abcdef0f0debc9a78563412", 16),
        new BigInteger("123456789abcdef0f0debc9a78563412", 16),
        null
    };
    try (ColumnVector cv = ColumnVector.decimalFromBigInt(-38, intvals);
         ColumnVector chunk1 = Aggregation128Utils.extractInt32Chunk(cv, DType.UINT32, 0);
         ColumnVector chunk2 = Aggregation128Utils.extractInt32Chunk(cv, DType.UINT32, 1);
         ColumnVector chunk3 = Aggregation128Utils.extractInt32Chunk(cv, DType.UINT32, 2);
         ColumnVector chunk4 = Aggregation128Utils.extractInt32Chunk(cv, DType.INT32, 3);
         Table actualChunks = new Table(chunk1, chunk2, chunk3, chunk4);
         ColumnVector expectedChunk1 = ColumnVector.fromBoxedUnsignedInts(
             null, 0x78563412, 0x78563412, 0x78563412, null);
         ColumnVector expectedChunk2 = ColumnVector.fromBoxedUnsignedInts(
             null, -0x0f214366, -0x0f214366, -0x0f214366, null);
         ColumnVector expectedChunk3 = ColumnVector.fromBoxedUnsignedInts(
             null, -0x65432110, -0x65432110, -0x65432110, null);
         ColumnVector expectedChunk4 = ColumnVector.fromBoxedInts(
             null, 0x12345678, 0x12345678, 0x12345678, null);
         Table expectedChunks = new Table(expectedChunk1, expectedChunk2, expectedChunk3, expectedChunk4)) {
      AssertUtils.assertTablesAreEqual(expectedChunks, actualChunks);
    }
  }

  @Test
  public void testCombineInt64SumChunks() {
    try (ColumnVector chunks0 = ColumnVector.fromBoxedUnsignedLongs(
             null, 0L, 1L, 0L, 0L, 0x12345678L, 0x123456789L, 0x1234567812345678L, 0xfedcba9876543210L);
         ColumnVector chunks1 = ColumnVector.fromBoxedUnsignedLongs(
             null, 0L, 2L, 0L, 0L, 0x9abcdef0L, 0x9abcdef01L, 0x1122334455667788L, 0xaceaceaceaceaceaL);
         ColumnVector chunks2 = ColumnVector.fromBoxedUnsignedLongs(
             null, 0L, 3L, 0L, 0L, 0x11223344L, 0x556677889L, 0x99aabbccddeeff00L, 0xbdfbdfbdfbdfbdfbL);
         ColumnVector chunks3 = ColumnVector.fromBoxedLongs(
             null, 0L, -1L, 0x100000000L, 0x80000000L, 0x55667788L, 0x01234567L, 0x66554434L, -0x42042043L);
         Table chunksTable = new Table(chunks0, chunks1, chunks2, chunks3);
         Table actual = Aggregation128Utils.combineInt64SumChunks(chunksTable, DType.create(DType.DTypeEnum.DECIMAL128, -20));
         ColumnVector expectedOverflows = ColumnVector.fromBoxedBooleans(
             null, false, false, true, true, false, false, true, false);
         ColumnVector expectedValues = ColumnVector.decimalFromBigInt(-20,
             null,
             new BigInteger("0", 16),
             new BigInteger("-fffffffcfffffffdffffffff", 16),
             new BigInteger("0", 16),
             new BigInteger("-80000000000000000000000000000000", 16),
             new BigInteger("55667788112233449abcdef012345678", 16),
             new BigInteger("123456c56677892abcdef0223456789", 16),
             new BigInteger("ef113244679ace0012345678", 16),
             new BigInteger("7bf7bf7ba8ca8ca8e9ab678276543210", 16));
         Table expected = new Table(expectedOverflows, expectedValues)) {
      AssertUtils.assertTablesAreEqual(expected, actual);
    }
  }
}
