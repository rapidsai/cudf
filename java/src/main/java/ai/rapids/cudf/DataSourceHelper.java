/*
 *
 *  Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * This is here because we need some JNI methods to work with a DataSource, but
 * we also want to cache callback methods at startup for performance reasons. If
 * we put both in the same class we will get a deadlock because of how we load
 * the JNI. We have a static block that blocks loading the class until the JNI
 * library is loaded and the JNI library cannot load until the class is loaded
 * and cached. This breaks the loop.
 */
class DataSourceHelper {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    static long createWrapperDataSource(DataSource ds) {
        return createWrapperDataSource(ds, ds.size(), ds.supportsDeviceRead(),
                ds.getDeviceReadCutoff());
    }

    private static native long createWrapperDataSource(DataSource ds, long size,
                                                       boolean deviceReadSupport,
                                                       long deviceReadCutoff);

    static native void destroyWrapperDataSource(long handle);
}
