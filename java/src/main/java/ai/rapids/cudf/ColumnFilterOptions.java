/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Supplier;

/**
 * Base options class for input formats that can filter columns.
 */
public abstract class ColumnFilterOptions {
  // Names of the columns to be returned (other columns are skipped)
  // If empty all columns are returned.
  private final String[] includeColumnNames;
  private final Long sizeGuess;

  protected ColumnFilterOptions(Builder<?> builder) {
    includeColumnNames = builder.includeColumnNames.toArray(
        new String[builder.includeColumnNames.size()]);
    sizeGuess = builder.sizeGuess;
  }

  String[] getIncludeColumnNames() {
    return includeColumnNames;
  }

  long getSizeGuessOrElse(long orElse) {
    if (sizeGuess == null) {
      return orElse;
    }
    return sizeGuess;
  }

  long getSizeGuessOrElse(Supplier<Long> orElse) {
    if (sizeGuess == null) {
      return orElse.get();
    }
    return sizeGuess;
  }

  public static class Builder<T extends Builder> {
    final List<String> includeColumnNames = new ArrayList<>();
    private Long sizeGuess = null;

    /**
     * Set a guess of how many bytes of memory the resulting data will take. This is totally
     * optional and only used for estimating the memory usage of loading the data. Most users
     * will not be able to use this accurately. This is a difficult task, but if you have a rough
     * idea of the row count and schema then calling
     * {@link ai.rapids.cudf.Schema#guessTableSize guessTableSize} will help.
     * @param guess a guess at the number of bytes needed to store the output table.
     * @return this for chaining.
     */
    public T withOutputSizeGuess(long guess) {
      sizeGuess = guess;
      return (T) this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    public T includeColumn(String... names) {
      for (String name : names) {
        includeColumnNames.add(name);
      }
      return (T) this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    public T includeColumn(Collection<String> names) {
      includeColumnNames.addAll(names);
      return (T) this;
    }
  }
}
