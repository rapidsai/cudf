/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
