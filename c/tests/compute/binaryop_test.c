#include "../common.h"

#include <cudf/binaryop.h>

#include <stdio.h>

int main(void)
{
  printf("binaryop_test: SKIP (no GPU available in CI)\n");
  return 0;
}
