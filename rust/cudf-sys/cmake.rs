/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use cmake_package::{Error as CmakeError, Version, VersionError, find_package};

const CUDF_C_COMPONENT: &str = "c_api";
const CUDF_C_API_TARGET: &str = "cudf::c_api";
const CUDF_C_CMAKE_INSPECTION_FAILED: &str = "CMake failed while inspecting cudf_c. Check the build environment for missing tools such as ninja/make, C/C++ compilers, or CUDA dependencies.";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PRINT_LIBCUDF_C_PACKAGE_DIR: &str = r#"
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("libcudf_c")
if spec is None or spec.submodule_search_locations is None:
    raise ModuleNotFoundError("libcudf_c")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
"#;

pub(crate) struct CudfCMetadata {
    pub(crate) include_dir: PathBuf,
    #[cfg(feature = "generate-bindings")]
    pub(crate) bindgen_include_dirs: Vec<PathBuf>,
    pub(crate) lib_dir: PathBuf,
}

fn cmake_unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "CMake is not installed or does not satisfy this build's requirements. Install the required CMake version and try again."
    )
}

fn cudf_c_package_not_found_error() -> anyhow::Error {
    anyhow::anyhow!(
        "Could not find a cudf_c CMake package compatible with cudf-sys {PACKAGE_VERSION}.\n\n\
         Install cudf_c via one of:\n\
         - conda: conda install -c rapidsai libcudf\n\
         - pip:   pip install libcudf-c-cu<CUDA_VERSION> and set LIBCUDF_USE_PYTHON=1\n\
         Or set CMAKE_PREFIX_PATH to point to your cudf_c build/install directory."
    )
}

fn cudf_c_incompatible_version_error(
    required_version: &Version,
    candidates: &[cmake_package::Considered],
) -> anyhow::Error {
    let considered = candidates
        .iter()
        .map(|candidate| format!("- {} (version: {})", candidate.config, candidate.version))
        .collect::<Vec<_>>()
        .join("\n");

    anyhow::anyhow!(
        "Found cudf_c CMake package candidates, but none are compatible with cudf-sys {PACKAGE_VERSION}.\n\n\
         Required compatibility: same major/minor as {required_version} and not older than {required_version}.\n\n\
         Considered candidates:\n{considered}"
    )
}

fn cmake_package_error(error: CmakeError, required_version: &Version) -> anyhow::Error {
    match error {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        CmakeError::PackageNotFound => cudf_c_package_not_found_error(),
        CmakeError::Version(VersionError::VersionIncompatible(candidates)) => {
            cudf_c_incompatible_version_error(required_version, &candidates)
        }
        CmakeError::Version(VersionError::InvalidVersion) => anyhow::anyhow!(
            "{CUDF_C_CMAKE_INSPECTION_FAILED}\n\nCMake reported an invalid cudf_c package version."
        ),
        CmakeError::IO(error) => {
            anyhow::anyhow!("{CUDF_C_CMAKE_INSPECTION_FAILED}\n\nUnderlying error: {error}")
        }
        CmakeError::Internal => anyhow::anyhow!("{CUDF_C_CMAKE_INSPECTION_FAILED}"),
    }
}

fn find_target(
    package: &cmake_package::CMakePackage,
    target_name: &str,
) -> Result<cmake_package::CMakeTarget> {
    package.target(target_name).with_context(|| {
        format!("Found CMake package {}, but target {target_name} was not exported.", package.name)
    })
}

fn find_cudf_c_package(
    required_version: &Version,
    python_package_dir: Option<&Path>,
) -> Result<cmake_package::CMakePackage> {
    let mut package = find_package("cudf_c")
        .version(*required_version)
        .components([CUDF_C_COMPONENT.to_owned()])
        .define("CMAKE_FIND_PACKAGE_PREFER_CONFIG", "TRUE");

    if let Some(python_package_dir) = python_package_dir {
        package = package.prefix_paths(vec![python_package_dir.to_path_buf()]);
    }

    package.find().map_err(|error| cmake_package_error(error, required_version))
}

/// Run CMake `find_package(cudf_c <version>)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
pub(crate) fn try_find_cudf_c_package(
    required_version: &Version,
    python_package_dir: Option<&Path>,
) -> Result<CudfCMetadata> {
    let package = find_cudf_c_package(required_version, python_package_dir)?;
    let target = find_target(&package, CUDF_C_API_TARGET)?;

    let include_dir = target
        .include_directories
        .first()
        .map(PathBuf::from)
        .context("cudf_c CMake target did not export any include directories")?;

    #[cfg(feature = "generate-bindings")]
    let bindgen_include_dirs: Vec<_> = {
        target
            .include_directories
            .iter()
            .skip(1)
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir())
            .filter(|dir| dir != &include_dir)
            .collect()
    };

    let lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .map(Path::to_path_buf)
        .or_else(|| target.link_directories.first().map(PathBuf::from))
        .context("cudf_c CMake target did not export a library location or link directory")?;

    target.link();

    Ok(CudfCMetadata {
        include_dir,
        #[cfg(feature = "generate-bindings")]
        bindgen_include_dirs,
        lib_dir,
    })
}

fn find_python_cudf_c_package_dir() -> Result<PathBuf> {
    let python =
        Path::new(if std::env::var_os("VIRTUAL_ENV").is_some() { "python" } else { "python3" });
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_PRINT_LIBCUDF_C_PACKAGE_DIR)
        .output()
        .with_context(|| format!("LIBCUDF_USE_PYTHON is set, but failed to run {:?}.", python))?;

    anyhow::ensure!(
        output.status.success(),
        "LIBCUDF_USE_PYTHON is set, but {:?} could not locate the Python libcudf_c package.\n\n\
             Install the libcudf_c wheel in that Python environment, or unset LIBCUDF_USE_PYTHON.\n\n\
             {}",
        python,
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let package_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let cmake_dir = package_dir.join("lib64/cmake/cudf_c");
    anyhow::ensure!(
        cmake_dir.is_dir(),
        "LIBCUDF_USE_PYTHON is set, but the Python libcudf_c package at {} does not contain a cudf_c CMake package under {}.",
        package_dir.display(),
        cmake_dir.display(),
    );

    Ok(package_dir)
}

/// Locate cudf_c either from standard CMake search paths or, when explicitly
/// requested, from the active Python environment.
pub(crate) fn locate_cudf_c() -> Result<CudfCMetadata> {
    let required_version: Version = PACKAGE_VERSION
        .try_into()
        .expect("workspace package version must be a valid semantic version");

    let python_package_dir = std::env::var_os("LIBCUDF_USE_PYTHON")
        .map(|_| find_python_cudf_c_package_dir())
        .transpose()?;

    try_find_cudf_c_package(&required_version, python_package_dir.as_deref())
}
