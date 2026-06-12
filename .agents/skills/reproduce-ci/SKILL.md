---
name: reproduce-ci
description: Reproduce cudf CI failures locally by running the same container images and scripts used in GitHub Actions. Use when debugging CI failures from a pull request. Arguments - job name and PR number.
---

# Reproducing cudf CI Failures Locally

For full background, see the [RAPIDS docs on reproducing CI](https://docs.rapids.ai/resources/reproducing-ci/).

## Usage

```bash
.agents/skills/reproduce-ci/run.sh <container-image> <ci-script> <pr-number> [--gpu] [--timeout <minutes>]
```

Examples:
```bash
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:$(head -1 VERSION | cut -d. -f1,2)-latest ci/test_cmake.sh 22538
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:$(head -1 VERSION | cut -d. -f1,2)-latest ci/test_java.sh 22538 --gpu
.agents/skills/reproduce-ci/run.sh rapidsai/citestwheel:$(head -1 VERSION | cut -d. -f1,2)-latest "ci/cudf_pandas_scripts/pandas-tests/run.sh pr" 22538 --gpu
```

The container image tag corresponds to the current RAPIDS version. Derive it from the `VERSION` file at the repo root:
```bash
RAPIDS_VERSION=$(head -1 VERSION | cut -d. -f1,2)  # e.g., "26.08"
```

The script launches a detached container, runs the CI script, and leaves the container running for inspection.
After `--timeout` minutes of idle (default: 30), the container is automatically removed.
```bash
docker exec -it cudf-ci-repro bash   # inspect interactively
docker rm -f cudf-ci-repro           # clean up manually before timeout
```

## Finding the Container Image, Script, and GPU Requirement

All CI job definitions live in `.github/workflows/pr.yaml`. Each job specifies:
- `container_image`: the Docker image to use
- `script`: the CI script to run
- `node_type`: determines whether a GPU is needed (`gpu-*` → pass `--gpu`)

```bash
grep -A 10 'job-name-here:' .github/workflows/pr.yaml
```

Alternatively, inspect the "Initialize Containers" step in the GitHub Actions job log for the exact image.
