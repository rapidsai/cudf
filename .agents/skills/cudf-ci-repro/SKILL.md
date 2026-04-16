---
name: cudf-ci-repro
description: Reproduce a cudf CI job locally from a GitHub Actions job URL. Use when the user provides a link to a CI job on GitHub Actions for rapidsai/cudf and wants to run it in a local Docker container.
---

Use this skill when the user provides a GitHub Actions job URL such as:
`https://github.com/rapidsai/cudf/actions/runs/<run_id>/job/<job_id>?pr=<pr_number>`

## Reference Material

Always read the **Reproducing CI** guide first: https://docs.rapids.ai/resources/reproducing-ci/

## Prerequisites

Verify before proceeding. Stop and report if any are missing.

1. **`docker` CLI** available and daemon running (`docker info`).
2. **`gh` CLI** authenticated — run `gh auth status`. If not, guide the user to run `gh auth login`. Token needs `repo` scope. Do not run `gh auth token` from within the agent.
3. **Working directory** is a `cudf` checkout.
4. Read the docker-repro script before invoking it to ensure it's still consistent with the guide:
`.agents/skills/cudf-ci-repro/scripts/docker-repro.sh`

## Step 1: Parse the Job URL

Run the helper script to extract run ID, job ID, and optional PR number:

```bash
python3 .agents/skills/cudf-ci-repro/scripts/parse-job-url.py "<URL>"
```

This outputs `RUN_ID`, `JOB_ID`, and `PR_NUMBER` (if present).

If the script is unavailable, extract manually:
- **Run ID**: path segment after `/runs/`
- **Job ID**: path segment after `/job/`
- **PR number**: `pr` query parameter (may be absent)

## Step 2: Fetch Job Metadata and Logs

```bash
gh api repos/rapidsai/cudf/actions/runs/$RUN_ID/jobs \
  --jq ".jobs[] | select(.id == $JOB_ID)"
```

From the job JSON, note:
- `name` — the job name (e.g. `conda-cpp-build / 12.9.1, 3.11, amd64, rockylinux8`)
- `steps[].name` and `steps[].conclusion` — which steps ran and their outcome

Download the full job log:

```bash
gh api repos/rapidsai/cudf/actions/jobs/$JOB_ID/logs > /tmp/ci_job_log.txt
```

Read through the log to identify:
1. The **container image** (look for `Initialize Containers` step or docker pull lines — typically `rapidsai/ci-conda:<tag>`).
2. The **CI scripts** executed (e.g. `./ci/build_cpp.sh`, `./ci/test_python.sh`).
3. The **final outcome** — whether the job passed or failed.
4. If failed, the **exact failure** — error messages, test names, assertion text.

## Step 3: Determine Build Type and Environment Variables

Classify the CI run:

| Signal | `--build-type` | `--ref-name` |
|--------|---------------|--------------|
| URL has `?pr=N` or job name contains `pull-request` | `pull-request` | `pull-request/<N>` |
| Branch build (e.g. `main`, `release/xx.yy`) | `branch` | the branch name |
| Nightly build | `nightly` | the branch name |

Also determine:
- Whether the job needs a GPU (test jobs do, build-only jobs don't).
- The nightly date if applicable.

## Step 4: Run the Repro Script

Invoke `docker-repro.sh` with the parameters gathered in Steps 1-3. The script handles checkout, tag fetching, docker launch, and CI script execution.

```bash
bash .agents/skills/cudf-ci-repro/scripts/docker-repro.sh \
  --image "<CONTAINER_IMAGE>" \
  --build-type "<BUILD_TYPE>" \
  --ref-name "<REF_NAME>" \
  --scripts "<SCRIPT1>,<SCRIPT2>,..." \
  [--pr <PR_NUMBER>] \
  [--nightly-date <YYYY-MM-DD>] \
  [--no-gpu] \
  [--local-build] \
  [--dry-run]
```

Use `--dry-run` first to show the user the docker command before executing.

Key flags:
- `--pr` triggers `gh pr checkout` before running.
- `--no-gpu` omits `--gpus all` (for build-only jobs).
- `--local-build` rewrites artifact channels to use local conda build output (for full local build+test without CI artifacts).
- `--gh-token` overrides the `GH_TOKEN` env var if needed.

## Step 5: Analyze Output

After the script completes, analyze the local output against the CI outcome from Step 2:

1. **Compare results** — did the local run match the CI outcome (both pass, both fail, or diverge)?
2. **Note discrepancies** — if local and CI results differ, call out likely causes (GPU driver version, environment differences, timing, flaky tests).
3. **Summarize** — give the user a clear summary: what ran, what the outcome was, and whether it matched CI.

### If the job failed (in CI or locally or both)

4. Summarize the **root cause** based on the error output.
5. **Ask the user** if they would like help investigating a fix.
6. If yes, launch a sub-agent (Task tool) with the failure log, the relevant CI script paths, the error context from Step 2, and any source files implicated in the failure to investigate and propose a fix.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GIT_DESCRIBE_NUMBER is undefined` | `git fetch https://github.com/rapidsai/cudf.git --tags` |
| Interactive GitHub auth prompt inside container | Ensure `GH_TOKEN` is set (docker-repro.sh passes it through) |
| GPU driver mismatch causing test differences | Note driver version from CI log; compare with local `nvidia-smi` |
| Log download returns empty/403 | Verify `gh auth status` has `repo` scope; re-auth if needed |
