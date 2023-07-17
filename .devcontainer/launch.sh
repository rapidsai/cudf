#! /usr/bin/env bash

launch_devcontainer() {

    # Ensure we're in the repo root
    cd "$( cd "$( dirname "$(realpath -m "${BASH_SOURCE[0]}")" )" && pwd )/..";

    local mode="${1:-single}";
    local pkgs="${2:-conda}";

    case "${pkgs}" in
        pip   ) ;;
        conda ) ;;
        *     ) pkgs="conda";;
    esac

    case "${mode}" in
        single  ) ;;
        unified ) ;;
        isolated) ;;
        *      ) mode="single";;
    esac

    local flavor="${pkgs}/${mode}";
    local workspace="$(basename "$(pwd)")";
    local tmpdir="$(mktemp -d)/${workspace}";
    local path="$(pwd)/.devcontainer/${flavor}";

    mkdir -p "${tmpdir}";
    cp -arL "$path/.devcontainer" "${tmpdir}/";
    sed -i "s@\${localWorkspaceFolder}@$(pwd)@g" "${tmpdir}/.devcontainer/devcontainer.json";
    path="${tmpdir}";

    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')";
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder";

    echo "devcontainer URL: ${url}";

    local launch="";
    if type open >/dev/null 2>&1; then
        launch="open";
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open";
    fi

    if [ -n "${launch}" ]; then
        code --new-window "${tmpdir}";
        exec "${launch}" "${url}" >/dev/null 2>&1;
    fi
}

launch_devcontainer "$@";
