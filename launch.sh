#!/bin/bash
set -e

WORKSPACE=/home/knataraj/cudf
CONFIG=$WORKSPACE/.devcontainer/cuda12.9-conda/devcontainer.json
START=false
STOP=false
for arg in "$@"; do
  [ "$arg" = "--start" ] && START=true
  [ "$arg" = "--stop" ] && STOP=true
done

RUNNING=$(docker ps -q --filter "label=devcontainer.local_folder=$WORKSPACE")

if [ "$STOP" = true ]; then
  [ -n "$RUNNING" ] && docker stop $RUNNING
  exit 0
fi

if [ -n "$RUNNING" ] && [ "$START" = false ]; then
  devcontainer exec --workspace-folder "$WORKSPACE" --config "$CONFIG" bash
  exit 0
fi

[ -n "$RUNNING" ] && docker stop $RUNNING

devcontainer up --workspace-folder "$WORKSPACE" --config "$CONFIG"
devcontainer exec --workspace-folder "$WORKSPACE" --config "$CONFIG" bash
