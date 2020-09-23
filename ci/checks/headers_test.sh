#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#####################################
# conda existence test for headers  #
#####################################

RETVAL=0
LIBNAME=cudf
DIRNAMES="cudf cudf_test"

# existence tests for lib${LIBNAME}
for DIRNAME in ${DIRNAMES[@]}; do
    HEADERS=`cd cpp && find include/${DIRNAME}/ -type f \( -iname "*.h" -o  -iname "*.hpp" \) -printf "    - test -f \\\$PREFIX/%p\n" | sort`
    META_TESTS=`grep -E "test -f .*/include/${DIRNAME}/.*\.h(pp)?" conda/recipes/lib${LIBNAME}/meta.yaml | sort`
    HEADER_DIFF=`diff <(echo "$HEADERS") <(echo "$META_TESTS")`
    LIB_RETVAL=$?

    if [ "$LIB_RETVAL" != "0" ]; then
        echo -e "\n\n>>>> FAILED: lib${LIBNAME} header existence conda/recipes/lib${LIBNAME}/meta.yaml check; begin output\n\n"
        echo -e "$HEADER_DIFF"
        echo -e "\n\n>>>> FAILED: lib${LIBNAME} header existence conda/recipes/lib${LIBNAME}/meta.yaml check; end output\n\n"
        RETVAL=1
    else
        echo -e "\n\n>>>> PASSED: lib${LIBNAME} header existence conda/recipes/lib${LIBNAME}/meta.yaml check\n\n"
    fi
done

exit $RETVAL
