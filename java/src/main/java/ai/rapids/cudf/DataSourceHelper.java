/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
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
