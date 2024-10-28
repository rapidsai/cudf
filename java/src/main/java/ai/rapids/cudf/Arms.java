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
import java.util.Collection;
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
     * This method safely closes all the resources.
     * <p>
     * This method will iterate through all the resources and closes them. If any exception happened during the
     * traversal, exception will be captured and rethrown after all resources closed.
     * </p>
     */
    public static <R extends AutoCloseable> void closeAll(Iterator<R> resources) {
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

        if (t != null) throw new RuntimeException(t);
    }


    /**
     * This method safely closes all the resources. See {@link #closeAll(Iterator)} for more details.
     */
    public static <R extends AutoCloseable> void closeAll(R... resources) {
        closeAll(Arrays.asList(resources));
    }

    /**
     * This method safely closes the resources. See {@link #closeAll(Iterator)} for more details.
     */
    public static <R extends AutoCloseable> void closeAll(Collection<R> resources) {
        closeAll(resources.iterator());
    }
}
