#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################
# cuDF CHANGELOG Tester #
#########################

# Checkout main for comparison
git checkout --force --quiet main

# Switch back to tip of PR branch
git checkout --force --quiet current-pr-branch

# Ignore errors during searching
set +e

# Get list of modified files between main and PR branch
CHANGELOG=`git diff --name-only main...current-pr-branch | grep CHANGELOG.md`
# Check if CHANGELOG has PR ID
PRNUM=`cat CHANGELOG.md | grep "$PR_ID"`
RETVAL=0

# Return status of check result
if [ "$CHANGELOG" != "" -a "$PRNUM" != "" ] ; then
  echo -e "\n\n>>>> PASSED: CHANGELOG.md has been updated with current PR information.\n\nPlease ensure the update meets the following criteria.\n"
else
  echo -e "\n\n>>>> FAILED: CHANGELOG.md has not been updated!\n\nPlease add a line describing this PR to CHANGELOG.md in the repository root directory. The line should meet the following criteria.\n"
  RETVAL=1
fi

cat << EOF
  It should be placed under the section for the appropriate release.
  It should be placed under "New Features", "Improvements", or "Bug Fixes" as appropriate.
  It should be formatted as '- PR #<PR number> <Concise human-readable description of the PR's new feature, improvement, or bug fix>'
    Example format for #491 '- PR #491 Add CI test script to check for updates to CHANGELOG.md in PRs'


EOF

exit $RETVAL
