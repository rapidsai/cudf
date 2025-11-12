/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class DecimalUtilsTest extends CudfTestBase {
    @Test
    public void testOutOfBounds() {
        try (ColumnView cv = ColumnVector.fromDecimals(
                new BigDecimal("-1E+3"),
                new BigDecimal("1E+3"),
                new BigDecimal("9E+1"),
                new BigDecimal("-9E+1"),
                new BigDecimal("-91"));
             ColumnView expected = ColumnVector.fromBooleans(true, true, false, false, true);
             ColumnView result = DecimalUtils.outOfBounds(cv, 1, -1)) {
            assertColumnsAreEqual(expected, result);
        }
    }
}
