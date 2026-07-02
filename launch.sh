#!/bin/bash
set -e

WORKSPACE_FOLDER="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${WORKSPACE_FOLDER}/.devcontainer/cuda12.9-conda/devcontainer.json"

if [ "$1" = "--stop" ]; then
    echo "Stopping devcontainer..."
    docker stop "$(docker ps -q --filter "label=devcontainer.local_folder=${WORKSPACE_FOLDER}")" 2>/dev/null || echo "No running container found"
    exit 0
fi

# Start the devcontainer (reuses existing container if already running)
devcontainer up \
    --workspace-folder "${WORKSPACE_FOLDER}" \
    --config "${CONFIG_FILE}" \
    --mount "type=bind,source=${HOME}/.claude,target=/home/coder/.claude" \
    --mount "type=bind,source=${HOME}/.claude.json,target=/home/coder/.claude.json"

# Exec into the container
if [ "$1" = "--claude" ]; then
    shift
    echo "Launching Claude Code in devcontainer..."
    devcontainer exec \
        --workspace-folder "${WORKSPACE_FOLDER}" \
        --config "${CONFIG_FILE}" \
        claude "$@"
else
    devcontainer exec \
        --workspace-folder "${WORKSPACE_FOLDER}" \
        --config "${CONFIG_FILE}" \
        bash
fi
