#include "../common.h"

#include <cudf/join.h>

#include <stdio.h>

int main(void)
{
  printf("join_test: SKIP (no GPU available in CI)\n");
  return 0;
}
