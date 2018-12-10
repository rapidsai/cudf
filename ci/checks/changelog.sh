#!/bin/bash
set -e
git checkout master
git checkout current-pr-branch
CHANGELOG=`git diff --name-only master...current-pr-branch | grep CHANGELOG.md`
PRNUM=`cat CHANGELOG.md | grep "$PR_ID"`
RETVAL=0

if [ "$CHANGELOG" != "" -a "$PRNUM" != "" ] ; then
  echo -e "\n\n>>>> PASSED: CHANGELOG.md has been updated with current PR information.\n\n"
else
  echo -e "\n\n>>>> FAILED: CHANGELOG.md has not been updated! Update with PR information to pass this test.\n\n"
  RETVAL=1
fi

exit $RETVAL
