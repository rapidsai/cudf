#include "../common.h"

#include <cudf/io/parquet.h>

#include <stdio.h>

int main(void)
{
  printf("parquet_test: SKIP (no GPU available in CI)\n");
  return 0;
}
