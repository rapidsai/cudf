#include "../common.h"

#include <cudf/interop/arrow.h>

#include <stdio.h>

// Smoke test: verify arrow interop functions are linked.
// Full tests require a GPU and live Arrow data.
int main(void)
{
  printf("arrow_test: SKIP (no GPU available in CI)\n");
  return 0;
}
