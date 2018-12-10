#!/bin/bash
#########################
# cuDF CHANGELOG Tester #
#########################

# Checkout master for comparison
git checkout master

# Switch back to tip of PR branch
git checkout current-pr-branch

# Ignore errors during searching
set +e

# Get list of modified files between matster and PR branch
CHANGELOG=`git diff --name-only master...current-pr-branch | grep CHANGELOG.md`
# Check if CHANGELOG has PR ID
PRNUM=`cat CHANGELOG.md | grep "$PR_ID"`
RETVAL=0

# Return status of check result
if [ "$CHANGELOG" != "" -a "$PRNUM" != "" ] ; then
  echo -e "\n\n>>>> PASSED: CHANGELOG.md has been updated with current PR information.\n\n"
else
  echo -e "\n\n>>>> FAILED: CHANGELOG.md has not been updated! Update with PR information to pass this test.\n\n"
  RETVAL=1
fi

exit $RETVAL
