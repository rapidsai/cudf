# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise cuDF's frozen, per-platform Maven candidate packaging."""

import json
import os
import platform
import subprocess  # nosec B404
import tarfile
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
PACKAGER = REPOSITORY / "ci/package_candidate_java_maven.sh"
SOURCE_VERSION = (REPOSITORY / "VERSION").read_text().strip().split(".")
CANDIDATE_VERSION = (
    f"{SOURCE_VERSION[0]}.{SOURCE_VERSION[1]}.{int(SOURCE_VERSION[2])}"
)


def _candidate_workspace(
    tmp_path: Path, version: str = CANDIDATE_VERSION
) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / "repo"
    (workspace / "ci").mkdir(parents=True)
    (workspace / "ci/package_candidate_java_maven.sh").write_bytes(
        PACKAGER.read_bytes()
    )
    (workspace / "ci/package_candidate_java_maven.sh").chmod(0o755)
    (workspace / "java/ci").mkdir(parents=True)
    (workspace / "java/pom.xml").write_text("<project/>\n")
    (workspace / "java/ci/java_classifier.sh").write_bytes(
        (REPOSITORY / "java/ci/java_classifier.sh").read_bytes()
    )
    classifier = (
        "cuda12-arm64"
        if platform.machine() in {"aarch64", "arm64"}
        else "cuda12"
    )
    classifier_dir = workspace / f"output_jars/{classifier}"
    classifier_dir.mkdir(parents=True)
    (classifier_dir / f"cudf-{version}-{classifier}.jar").write_bytes(
        b"classifier-jar"
    )
    (classifier_dir / f"cudf-{version}.pom").write_text("<project/>\n")
    (classifier_dir / f"cudf-{version}-sources.jar").write_bytes(b"sources")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    mvn = bindir / "mvn"
    mvn.write_text("#!/bin/sh\nprintf '%s\\n' '" + version + "'\n")
    mvn.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "RAPIDS_RELEASE_CANDIDATE": "true",
        "RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS": json.dumps(
            {"unit_package_versions": {"maven:cudf-java": version}}
        ),
        "RAPIDS_CUDA_VERSION": "12.9.2",
        "RAPIDS_CONDA_ARCH": "amd64",
    }
    return workspace, environment


def test_packages_classifier_as_maven_archive_for_exact_platform(tmp_path):
    workspace, environment = _candidate_workspace(tmp_path)
    classifier = (
        "cuda12-arm64"
        if platform.machine() in {"aarch64", "arm64"}
        else "cuda12"
    )
    subprocess.run(  # nosec B603 - fixed checked-in script and isolated fixture workspace.
        ["bash", str(workspace / "ci/package_candidate_java_maven.sh")],
        cwd=workspace,
        env=environment,
        check=True,
    )

    output = workspace / ".release-candidate-artifacts/maven"
    identity = json.loads((output / "cudf-java.identity.json").read_text())
    assert identity == {
        "ecosystem": "maven",
        "name": "ai.rapids:cudf",
        "platform": "amd64-cuda12",
        "version": CANDIDATE_VERSION,
    }
    with tarfile.open(output / "cudf-java.tar.gz", "r:gz") as archive:
        names = {
            name.removeprefix("./")
            for name in archive.getnames()
            if not any(part.startswith("._") for part in name.split("/"))
        }
    prefix = f"ai/rapids/cudf/{CANDIDATE_VERSION}/"
    assert prefix + f"cudf-{CANDIDATE_VERSION}-{classifier}.jar" in names
    assert prefix + f"cudf-{CANDIDATE_VERSION}.jar" in names
    assert prefix + f"cudf-{CANDIDATE_VERSION}.pom" in names
    assert prefix + f"cudf-{CANDIDATE_VERSION}-sources.jar" in names


def test_rejects_build_that_does_not_match_frozen_maven_version(tmp_path):
    workspace, environment = _candidate_workspace(
        tmp_path, version=f"{CANDIDATE_VERSION}-wrong"
    )
    environment["RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS"] = json.dumps(
        {"unit_package_versions": {"maven:cudf-java": CANDIDATE_VERSION}}
    )
    result = subprocess.run(  # nosec B603 - fixed checked-in script and isolated fixture workspace.
        ["bash", str(workspace / "ci/package_candidate_java_maven.sh")],
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not match frozen candidate version" in result.stderr
    assert not (
        workspace / ".release-candidate-artifacts/maven/cudf-java.tar.gz"
    ).exists()
