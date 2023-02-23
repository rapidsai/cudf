# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import argparse
import datetime
import os
import re
import sys

import git

FilesToCheck = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$"),
]
ExemptFiles = [
    re.compile(r"cpp/include/cudf_test/cxxopts.hpp"),
]

# this will break starting at year 10000, which is probably OK :)
CheckSimple = re.compile(
    r"Copyright *(?:\(c\))? *(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"
)
CheckDouble = re.compile(
    r"Copyright *(?:\(c\))? *(\d{4})-(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"  # noqa: E501
)


def checkThisFile(f):
    if isinstance(f, git.Diff):
        if f.deleted_file or f.b_blob.size == 0:
            return False
        f = f.b_path
    elif not os.path.exists(f) or os.stat(f).st_size == 0:
        # This check covers things like symlinks which point to files that DNE
        return False
    for exempt in ExemptFiles:
        if exempt.search(f):
            return False
    for checker in FilesToCheck:
        if checker.search(f):
            return True
    return False


def modifiedFiles():
    """Get a set of all modified files, as Diff objects.

    The files returned have been modified in git since the merge base of HEAD
    and the upstream of the target branch. We return the Diff objects so that
    we can read only the staged changes.
    """
    repo = git.Repo()
    # Use the environment variable TARGET_BRANCH or RAPIDS_BASE_BRANCH (defined in CI) if possible
    target_branch = os.environ.get("TARGET_BRANCH", os.environ.get("RAPIDS_BASE_BRANCH"))
    if target_branch is None:
        # Fall back to the closest branch if not on CI
        target_branch = repo.git.describe(
            all=True, tags=True, match="branch-*", abbrev=0
        ).lstrip("heads/")

    upstream_target_branch = None
    if target_branch in repo.heads:
        # Use the tracking branch of the local reference if it exists. This
        # returns None if no tracking branch is set.
        upstream_target_branch = repo.heads[target_branch].tracking_branch()
    if upstream_target_branch is None:
        # Fall back to the remote with the newest target_branch. This code
        # path is used on CI because the only local branch reference is
        # current-pr-branch, and thus target_branch is not in repo.heads.
        # This also happens if no tracking branch is defined for the local
        # target_branch. We use the remote with the latest commit if
        # multiple remotes are defined.
        candidate_branches = [
            remote.refs[target_branch] for remote in repo.remotes
            if target_branch in remote.refs
        ]
        if len(candidate_branches) > 0:
            upstream_target_branch = sorted(
                candidate_branches,
                key=lambda branch: branch.commit.committed_datetime,
            )[-1]
        else:
            # If no remotes are defined, try to use the local version of the
            # target_branch. If this fails, the repo configuration must be very
            # strange and we can fix this script on a case-by-case basis.
            upstream_target_branch = repo.heads[target_branch]
    merge_base = repo.merge_base("HEAD", upstream_target_branch.commit)[0]
    diff = merge_base.diff()
    changed_files = {f for f in diff if f.b_path is not None}
    return changed_files


def getCopyrightYears(line):
    res = CheckSimple.search(line)
    if res:
        return int(res.group(1)), int(res.group(1))
    res = CheckDouble.search(line)
    if res:
        return int(res.group(1)), int(res.group(2))
    return None, None


def replaceCurrentYear(line, start, end):
    # first turn a simple regex into double (if applicable). then update years
    res = CheckSimple.sub(r"Copyright (c) \1-\1, NVIDIA CORPORATION", line)
    res = CheckDouble.sub(
        rf"Copyright (c) {start:04d}-{end:04d}, NVIDIA CORPORATION",
        res,
    )
    return res


def checkCopyright(f, update_current_year):
    """Checks for copyright headers and their years."""
    errs = []
    thisYear = datetime.datetime.now().year
    lineNum = 0
    crFound = False
    yearMatched = False

    if isinstance(f, git.Diff):
        path = f.b_path
        lines = f.b_blob.data_stream.read().decode().splitlines(keepends=True)
    else:
        path = f
        with open(f, encoding="utf-8") as fp:
            lines = fp.readlines()

    for line in lines:
        lineNum += 1
        start, end = getCopyrightYears(line)
        if start is None:
            continue
        crFound = True
        if start > end:
            e = [
                path,
                lineNum,
                "First year after second year in the copyright "
                "header (manual fix required)",
                None,
            ]
            errs.append(e)
        elif thisYear < start or thisYear > end:
            e = [
                path,
                lineNum,
                "Current year not included in the copyright header",
                None,
            ]
            if thisYear < start:
                e[-1] = replaceCurrentYear(line, thisYear, end)
            if thisYear > end:
                e[-1] = replaceCurrentYear(line, start, thisYear)
            errs.append(e)
        else:
            yearMatched = True
    # copyright header itself not found
    if not crFound:
        e = [
            path,
            0,
            "Copyright header missing or formatted incorrectly "
            "(manual fix required)",
            None,
        ]
        errs.append(e)
    # even if the year matches a copyright header, make the check pass
    if yearMatched:
        errs = []

    if update_current_year:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            lines_changed = ", ".join(str(x[1]) for x in errs_update)
            print(f"File: {path}. Changing line(s) {lines_changed}")
            for _, lineNum, __, replacement in errs_update:
                lines[lineNum - 1] = replacement
            with open(path, "w", encoding="utf-8") as out_file:
                out_file.writelines(lines)

    return errs


def getAllFilesUnderDir(root, pathFilter=None):
    retList = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            filePath = os.path.join(dirpath, fn)
            if pathFilter(filePath):
                retList.append(filePath)
    return retList


def checkCopyright_main():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    retVal = 0

    argparser = argparse.ArgumentParser(
        "Checks for a consistent copyright header in git's modified files"
    )
    argparser.add_argument(
        "--update-current-year",
        dest="update_current_year",
        action="store_true",
        required=False,
        help="If set, "
        "update the current year if a header is already "
        "present and well formatted.",
    )
    argparser.add_argument(
        "--git-modified-only",
        dest="git_modified_only",
        action="store_true",
        required=False,
        help="If set, "
        "only files seen as modified by git will be "
        "processed.",
    )

    args, dirs = argparser.parse_known_args()

    if args.git_modified_only:
        files = [f for f in modifiedFiles() if checkThisFile(f)]
    else:
        files = []
        for d in [os.path.abspath(d) for d in dirs]:
            if not os.path.isdir(d):
                raise ValueError(f"{d} is not a directory.")
            files += getAllFilesUnderDir(d, pathFilter=checkThisFile)

    errors = []
    for f in files:
        errors += checkCopyright(f, args.update_current_year)

    if len(errors) > 0:
        if any(e[-1] is None for e in errors):
            print("Copyright headers incomplete in some of the files!")
        for e in errors:
            print("  %s:%d Issue: %s" % (e[0], e[1], e[2]))
        print("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        path_parts = os.path.abspath(__file__).split(os.sep)
        file_from_repo = os.sep.join(path_parts[path_parts.index("ci") :])
        if n_fixable > 0 and not args.update_current_year:
            print(
                f"You can run `python {file_from_repo} --git-modified-only "
                "--update-current-year` and stage the results in git to "
                f"fix {n_fixable} of these errors.\n"
            )
        retVal = 1

    return retVal


if __name__ == "__main__":
    sys.exit(checkCopyright_main())
