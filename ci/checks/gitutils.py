# Copyright (c) 2019-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import subprocess
import os
import re


def isFileEmpty(f):
    return os.stat(f).st_size == 0


def __git(*opts):
    """Runs a git command and returns its output"""
    cmd = "git " + " ".join(list(opts))
    ret = subprocess.check_output(cmd, shell=True)
    return ret.decode("UTF-8").rstrip("\n")


def __gitdiff(*opts):
    """Runs a git diff command with no pager set"""
    return __git("--no-pager", "diff", *opts)


def branch():
    """Returns the name of the current branch"""
    name = __git("rev-parse", "--abbrev-ref", "HEAD")
    name = name.rstrip()
    return name


def repo_version():
    """
    Determines the version of the repo by using `git describe`

    Returns
    -------
    str
        The full version of the repo in the format 'v#.#.#{a|b|rc}'
    """
    return __git("describe", "--tags", "--abbrev=0")


def repo_version_major_minor():
    """
    Determines the version of the repo using `git describe` and returns only
    the major and minor portion

    Returns
    -------
    str
        The partial version of the repo in the format '{major}.{minor}'
    """

    full_repo_version = repo_version()

    match = re.match(r"^v?(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?",
                     full_repo_version)

    if (match is None):
        print("   [DEBUG] Could not determine repo major minor version. "
              f"Full repo version: {full_repo_version}.")
        return None

    out_version = match.group("major")

    if (match.group("minor")):
        out_version += "." + match.group("minor")

    return out_version


def determine_merge_commit(current_branch="HEAD"):
    """
    When running outside of CI, this will estimate the target merge commit hash
    of `current_branch` by finding a common ancester with the remote branch
    'branch-{major}.{minor}' where {major} and {minor} are determined from the
    repo version.

    Parameters
    ----------
    current_branch : str, optional
        Which branch to consider as the current branch, by default "HEAD"

    Returns
    -------
    str
        The common commit hash ID
    """

    try:
        # Try to determine the target branch from the most recent tag
        head_branch = __git("describe",
                            "--all",
                            "--tags",
                            "--match='branch-*'",
                            "--abbrev=0")
    except subprocess.CalledProcessError:
        print("   [DEBUG] Could not determine target branch from most recent "
              "tag. Falling back to 'branch-{major}.{minor}.")
        head_branch = None

    if (head_branch is not None):
        # Convert from head to branch name
        head_branch = __git("name-rev", "--name-only", head_branch)
    else:
        # Try and guess the target branch as "branch-<major>.<minor>"
        version = repo_version_major_minor()

        if (version is None):
            return None

        head_branch = "branch-{}".format(version)

    try:
        # Now get the remote tracking branch
        remote_branch = __git("rev-parse",
                              "--abbrev-ref",
                              "--symbolic-full-name",
                              head_branch + "@{upstream}")
    except subprocess.CalledProcessError:
        print("   [DEBUG] Could not remote tracking reference for "
              f"branch {head_branch}.")
        remote_branch = None

    if (remote_branch is None):
        return None

    print(f"   [DEBUG] Determined TARGET_BRANCH as: '{remote_branch}'. "
          "Finding common ancestor.")

    common_commit = __git("merge-base", remote_branch, current_branch)

    return common_commit


def uncommittedFiles():
    """
    Returns a list of all changed files that are not yet committed. This
    means both untracked/unstaged as well as uncommitted files too.
    """
    files = __git("status", "-u", "-s")
    ret = []
    for f in files.splitlines():
        f = f.strip(" ")
        f = re.sub("\s+", " ", f)  # noqa: W605
        tmp = f.split(" ", 1)
        # only consider staged files or uncommitted files
        # in other words, ignore untracked files
        if tmp[0] == "M" or tmp[0] == "A":
            ret.append(tmp[1])
    return ret


def changedFilesBetween(baseName, branchName, commitHash):
    """
    Returns a list of files changed between branches baseName and latest commit
    of branchName.
    """
    current = branch()
    # checkout "base" branch
    __git("checkout", "--force", baseName)
    # checkout branch for comparing
    __git("checkout", "--force", branchName)
    # checkout latest commit from branch
    __git("checkout", "-fq", commitHash)

    files = __gitdiff("--name-only",
                      "--ignore-submodules",
                      f"{baseName}..{branchName}")

    # restore the original branch
    __git("checkout", "--force", current)
    return files.splitlines()


def changesInFileBetween(file, b1, b2, filter=None):
    """Filters the changed lines to a file between the branches b1 and b2"""
    current = branch()
    __git("checkout", "--quiet", b1)
    __git("checkout", "--quiet", b2)
    diffs = __gitdiff("--ignore-submodules",
                      "-w",
                      "--minimal",
                      "-U0",
                      "%s...%s" % (b1, b2),
                      "--",
                      file)
    __git("checkout", "--quiet", current)
    lines = []
    for line in diffs.splitlines():
        if filter is None or filter(line):
            lines.append(line)
    return lines


def modifiedFiles(pathFilter=None):
    """
    If inside a CI-env (ie. TARGET_BRANCH and COMMIT_HASH are defined, and
    current branch is "current-pr-branch"), then lists out all files modified
    between these 2 branches. Locally, TARGET_BRANCH will try to be determined
    from the current repo version and finding a coresponding branch named
    'branch-{major}.{minor}'. If this fails, this functino will list out all
    the uncommitted files in the current branch.

    Such utility function is helpful while putting checker scripts as part of
    cmake, as well as CI process. This way, during development, only the files
    touched (but not yet committed) by devs can be checked. But, during the CI
    process ALL files modified by the dev, as submiited in the PR, will be
    checked. This happens, all the while using the same script.
    """
    targetBranch = os.environ.get("TARGET_BRANCH")
    commitHash = os.environ.get("COMMIT_HASH")
    currentBranch = branch()
    print(
        f"   [DEBUG] TARGET_BRANCH={targetBranch}, COMMIT_HASH={commitHash}, "
        f"currentBranch={currentBranch}")

    if targetBranch and commitHash and (currentBranch == "current-pr-branch"):
        print("   [DEBUG] Assuming a CI environment.")
        allFiles = changedFilesBetween(targetBranch, currentBranch, commitHash)
    else:
        print("   [DEBUG] Did not detect CI environment. "
              "Determining TARGET_BRANCH locally.")

        common_commit = determine_merge_commit(currentBranch)

        if (common_commit is not None):

            # Now get the diff. Use --staged to get both diff between
            # common_commit..HEAD and any locally staged files
            allFiles = __gitdiff("--name-only",
                                 "--ignore-submodules",
                                 "--staged",
                                 f"{common_commit}").splitlines()
        else:
            # Fallback to just uncommitted files
            allFiles = uncommittedFiles()

    files = []
    for f in allFiles:
        if pathFilter is None or pathFilter(f):
            files.append(f)

    filesToCheckString = "\n\t".join(files) if files else "<None>"
    print(f"   [DEBUG] Found files to check:\n\t{filesToCheckString}\n")
    return files


def listAllFilesInDir(folder):
    """Utility function to list all files/subdirs in the input folder"""
    allFiles = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            allFiles.append(os.path.join(root, name))
    return allFiles


def listFilesToCheck(filesDirs, filter=None):
    """
    Utility function to filter the input list of files/dirs based on the input
    filter method and returns all the files that need to be checked
    """
    allFiles = []
    for f in filesDirs:
        if os.path.isfile(f):
            if filter is None or filter(f):
                allFiles.append(f)
        elif os.path.isdir(f):
            files = listAllFilesInDir(f)
            for f_ in files:
                if filter is None or filter(f_):
                    allFiles.append(f_)
    return allFiles
