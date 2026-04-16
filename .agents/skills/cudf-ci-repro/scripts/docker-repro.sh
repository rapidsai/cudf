#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: repro.sh [OPTIONS]

Checkout the correct cudf commit and launch a CI-replica Docker container
that runs specified CI scripts inside it.

Required:
  --image IMAGE           CI container image (e.g. rapidsai/ci-conda:cuda12.9.1-rockylinux8-py3.11)
  --build-type TYPE       One of: pull-request, branch, nightly
  --ref-name REF          e.g. pull-request/22144, main, release/26.06
  --scripts SCRIPT,...    Comma-separated CI scripts to run (e.g. ./ci/build_cpp.sh,./ci/test_cpp.sh)

Optional:
  --pr PR_NUMBER          PR number — triggers `gh pr checkout`
  --nightly-date DATE     Required when --build-type=nightly (YYYY-MM-DD)
  --no-gpu                Omit --gpus flag (for build-only jobs)
  --local-build           Rewrite artifact channels to use local conda build output
  --gh-token TOKEN        GitHub token; defaults to GH_TOKEN env var
  --dry-run               Print the docker command without executing
  -h, --help              Show this help
EOF
  exit "${1:-0}"
}

IMAGE=""
BUILD_TYPE=""
REF_NAME=""
SCRIPTS=""
PR_NUMBER=""
NIGHTLY_DATE=""
USE_GPU=1
LOCAL_BUILD=0
TOKEN="${GH_TOKEN:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)        IMAGE="$2";        shift 2 ;;
    --build-type)   BUILD_TYPE="$2";   shift 2 ;;
    --ref-name)     REF_NAME="$2";     shift 2 ;;
    --scripts)      SCRIPTS="$2";      shift 2 ;;
    --pr)           PR_NUMBER="$2";    shift 2 ;;
    --nightly-date) NIGHTLY_DATE="$2"; shift 2 ;;
    --no-gpu)       USE_GPU=0;         shift   ;;
    --local-build)  LOCAL_BUILD=1;     shift   ;;
    --gh-token)     TOKEN="$2";        shift 2 ;;
    --dry-run)      DRY_RUN=1;         shift   ;;
    -h|--help)      usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

# --- Validate required args ---
missing=()
[[ -z "$IMAGE" ]]      && missing+=("--image")
[[ -z "$BUILD_TYPE" ]] && missing+=("--build-type")
[[ -z "$REF_NAME" ]]   && missing+=("--ref-name")
[[ -z "$SCRIPTS" ]]    && missing+=("--scripts")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Error: missing required arguments: ${missing[*]}" >&2
  usage 1
fi

if [[ "$BUILD_TYPE" == "nightly" && -z "$NIGHTLY_DATE" ]]; then
  echo "Error: --nightly-date is required when --build-type=nightly" >&2
  exit 1
fi

# --- Step 1: Checkout ---
if [[ -n "$PR_NUMBER" ]]; then
  echo ">>> Checking out PR #${PR_NUMBER}..."
  gh pr checkout "$PR_NUMBER" --repo rapidsai/cudf
fi

echo ">>> Fetching tags..."
git fetch https://github.com/rapidsai/cudf.git --tags 2>/dev/null || true

# --- Step 2: Prepare local-build sed patch if requested ---
if [[ "$LOCAL_BUILD" -eq 1 ]]; then
  echo ">>> Patching ci/*.sh to use local conda build output..."
  sed -ri '/rapids-download-conda-from-github/ s/_CHANNEL=.*/_CHANNEL=${RAPIDS_CONDA_BLD_OUTPUT_DIR}/' ci/*.sh
fi

# --- Step 3: Build docker command ---
IFS=',' read -ra SCRIPT_ARRAY <<< "$SCRIPTS"
SCRIPT_CMD=""
for s in "${SCRIPT_ARRAY[@]}"; do
  s="$(echo "$s" | xargs)"
  SCRIPT_CMD+="${s} && "
done
SCRIPT_CMD="${SCRIPT_CMD% && }"

docker_cmd=(
  docker run
  --rm
  --pull=always
  --volume "$PWD:/repo"
  --workdir /repo
)

[[ "$USE_GPU" -eq 1 ]] && docker_cmd+=(--gpus all)

docker_cmd+=(
  --env "RAPIDS_BUILD_TYPE=$BUILD_TYPE"
  --env "RAPIDS_REPOSITORY=rapidsai/cudf"
  --env "RAPIDS_REF_NAME=$REF_NAME"
)

[[ -n "$TOKEN" ]]        && docker_cmd+=(--env "GH_TOKEN=$TOKEN")
[[ -n "$NIGHTLY_DATE" ]] && docker_cmd+=(--env "RAPIDS_NIGHTLY_DATE=$NIGHTLY_DATE")

docker_cmd+=("$IMAGE" bash -c "$SCRIPT_CMD")

echo ""
echo ">>> Docker command:"
echo "  ${docker_cmd[*]}"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "(dry-run — not executing)"
  exit 0
fi

echo ">>> Launching container and running CI scripts..."
exec "${docker_cmd[@]}"
