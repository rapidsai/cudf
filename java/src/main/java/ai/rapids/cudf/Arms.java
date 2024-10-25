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

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Function;

/**
 * This class contains utility methods for automatic resource management.
 */
public class Arms {
    /**
     * This method close the resource if an exception is thrown while executing the function.
     */
    public static <R extends AutoCloseable, T> T closeIfException(R resource, Function<R, T> function) {
        try {
            return function.apply(resource);
        } catch (Exception e) {
            if (resource != null) {
                try {
                    resource.close();
                } catch (Exception inner) {
                    e.addSuppressed(inner);
                }
            }
            throw e;
        }
    }

    /**
     * This method safes closes the resources.
     */
    public static <R extends AutoCloseable> void close(Iterator<R> resources) {
        Throwable t = null;
        while (resources.hasNext()) {
            try {
                resources.next().close();
            } catch (Exception e) {
                if (t == null) {
                    t = e;
                } else {
                    t.addSuppressed(e);
                }
            }
        }
    }

    /**
     * This method safes closes the resources.
     */
    public static <R extends AutoCloseable> void close(R... resources) {
        close(Arrays.asList(resources));
    }

    /**
     * This method safes closes the resources.
     */
    public static <R extends AutoCloseable> void close(Iterable<R> resources) {
        close(resources.iterator());
    }
}
