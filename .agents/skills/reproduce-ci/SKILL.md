---
name: reproduce-ci
description: Reproduce cudf CI failures locally by running the same container images and scripts used in GitHub Actions. Use when debugging CI failures from a pull request. Arguments - job name and PR number.
---

# Reproducing cudf CI Failures Locally

## Quick Start

Run the reproduction script with the container image, CI script, and PR number:

```bash
.agents/skills/reproduce-ci/run.sh <container-image> <ci-script> <pr-number> [--gpu]
```

Examples:
```bash
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:26.08-latest ci/test_cmake.sh 22538
.agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:26.08-latest ci/test_java.sh 22538 --gpu
.agents/skills/reproduce-ci/run.sh rapidsai/citestwheel:26.08-latest "ci/cudf_pandas_scripts/pandas-tests/run.sh pr" 22538 --gpu
```

The script will:
1. Pull the specified container image
2. Launch it in detached mode with the repo volume-mounted
3. Set all required environment variables (`RAPIDS_BUILD_TYPE`, `RAPIDS_REPOSITORY`, `RAPIDS_REF_NAME`, `GH_TOKEN`)
4. Run the CI script non-interactively
5. Leave the container running so you can inspect state

To join the container interactively from another terminal:
```bash
docker exec -it cudf-ci-repro bash
```

To clean up:
```bash
docker rm -f cudf-ci-repro
```

## How to Find the Container Image, Script, and GPU Requirement

All CI job definitions live in `.github/workflows/pr.yaml`. Each job specifies:
- `container_image`: the Docker image to use
- `script`: the CI script to run
- `node_type`: determines whether a GPU is needed

### Step 1: Find the job in the workflow YAML

```bash
grep -A 10 'job-name-here:' .github/workflows/pr.yaml
```

Look for the `container_image` and `script` fields in the `with:` block.

### Step 2: Determine if GPU is needed

- Jobs with `node_type: "gpu-*"` require a GPU — pass `--gpu`
- Jobs with `node_type: "cpu*"` or no `node_type` do not require a GPU
- Build jobs generally don't need GPUs; test jobs generally do

### Step 3: Check the GitHub Actions log (alternative)

If you can't find the job in the workflow YAML, inspect the "Initialize Containers" step in the GitHub Actions job log. It shows the exact `docker pull` command with the image name.

### Container image conventions

- `rapidsai/ci-conda:*` — conda-based build and test jobs
- `rapidsai/ci-wheel:*` — wheel build jobs
- `rapidsai/citestwheel:*` — wheel-based test jobs (pandas-tests, wheel-tests-*)

## Prerequisites

- Docker installed with access to pull from `rapidsai/` on Docker Hub
- `gh` CLI authenticated (for `GH_TOKEN` — needed by test jobs that download build artifacts)
- GPU drivers + `--gpus` support if reproducing GPU test jobs
- Current directory is the cudf repo with the PR branch checked out

## How It Works

cudf CI jobs are shell scripts run inside [rapidsai/ci-imgs](https://github.com/rapidsai/ci-imgs) containers. The script:

1. Pulls the specified container image
2. Launches the container with `tail -f /dev/null` (stays running)
3. Runs the CI script via `docker exec`
4. Reports exit status but keeps the container alive for debugging

## Environment Variables

The script sets these automatically:

| Variable | Value | Purpose |
|----------|-------|---------|
| `RAPIDS_BUILD_TYPE` | `pull-request` | Tells CI scripts this is a PR build |
| `RAPIDS_REPOSITORY` | `rapidsai/cudf` | Repository name for artifact downloads |
| `RAPIDS_REF_NAME` | `pull-request/<PR_NUMBER>` | Branch ref for artifact downloads |
| `GH_TOKEN` | From `gh auth token` | Authentication for GitHub API/artifact store |

## Running a Full Local Build + Test (No Artifact Downloads)

If you want to skip downloading pre-built artifacts and instead build everything locally inside the container:

```bash
# Inside the container:
sed -ri '/rapids-download-conda-from-github/ s/_CHANNEL=.*/_CHANNEL=${RAPIDS_CONDA_BLD_OUTPUT_DIR}/' ci/*.sh

./ci/build_cpp.sh
./ci/build_python.sh
./ci/test_cpp.sh
./ci/test_python_cudf.sh
```

## Common Issues

### "GIT_DESCRIBE_NUMBER is undefined"
Fetch tags before launching: `git fetch git@github.com:rapidsai/cudf.git --tags`

### Interactive GitHub login prompt
Ensure `GH_TOKEN` is set. The script uses `gh auth token` automatically, but if `gh` isn't authenticated, set `GH_TOKEN` manually.

### Maven 429 errors (Java tests)
Transient rate limiting from Maven Central. Retry after a few minutes.

### Artifact download prompts
Should not happen — the script sets all required env vars. If they do appear, check that the PR number is correct and that CI build jobs have completed for that PR.
