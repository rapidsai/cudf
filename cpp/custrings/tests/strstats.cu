#include <cuda_runtime.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "nvstrings/NVStrings.h"
#include "nvstrings/StringsStatistics.h"

//
// cd ../cpp/build
// nvcc -w -std=c++11 --expt-extended-lambda -gencode arch=compute_70,code=sm_70
// ../tests/strstats.cu -L. -lNVStrings -o strstats --linker-options -rpath,.:
//
// 36634-rows.csv has the following header line
// policyID,statecode,county,eq_site_limit,hu_site_limit,fl_site_limit,fr_site_limit,tiv_2011,tiv_2012,eq_site_deductible,hu_site_deductible,fl_site_deductible,fr_site_deductible,point_latitude,point_longitude,line,construction,point_granularity
//

int main(int argc, char** argv)
{
  setlocale(LC_NUMERIC, "");
  // std::string csvfile = "../../data/36634-rows.csv";
  std::string csvfile = "../../data/utf8.csv";
  NVStrings* dstrs    = NVStrings::create_from_csv(csvfile.c_str(), 16);
  printf("First 10 strings:\n");
  dstrs->print(0, 10);
  printf("\n");

  StringsStatistics stats;
  dstrs->compute_statistics(stats);

  printf("Totals:\n------\n");
  printf("  Bytes      = %'ld\n", stats.total_bytes);
  printf("  Characters = %'ld\n", stats.total_chars);
  printf("  Device memory = %'ld bytes\n", stats.total_memory);
  printf("Strings:\n-------\n");
  printf(
    "  Bytes:       avg = %'ld (%'ld - %'ld)\n", stats.bytes_avg, stats.bytes_min, stats.bytes_max);
  printf(
    "  Characters:  avg = %'ld (%'ld - %'ld)\n", stats.chars_avg, stats.chars_min, stats.bytes_max);
  printf("  Memory:      avg = %'ld (%'ld - %'ld)\n", stats.mem_avg, stats.mem_min, stats.mem_max);
  printf("  Count = %'ld\n", stats.total_strings);
  printf("  Nulls = %'ld, Empties = %'ld\n", stats.total_nulls, stats.total_empty);
  printf("  Unique = %'ld\n", stats.unique_strings);
  printf("Characters:\n----------\n");
  printf("  Whitespace = %'ld\n", stats.whitespace_count);
  printf("  Digits = %'ld\n", stats.digits_count);
  printf("  Uppercase = %'ld\n", stats.uppercase_count);
  printf("  Lowercase = %'ld\n", stats.lowercase_count);

  printf("  Character histogram size = %'ld\n", stats.char_counts.size());
  for (int idx = 0; idx < (int)stats.char_counts.size(); ++idx) {
    unsigned int chr     = stats.char_counts[idx].first;
    unsigned int num     = stats.char_counts[idx].second;
    unsigned char out[5] = {0, 0, 0, 0, 0};
    unsigned char* ptr   = out + ((chr & 0xF0000000) == 0xF0000000) +
                         ((chr & 0xFFE00000) == 0x00E00000) + ((chr & 0xFFFFC000) == 0x0000C000);
    // printf("%p,%p,%x,%d\n",out,ptr,(chr & 0xFFFF),(int)((chr & 0xFFFFC000)==0x0000C00000));
    unsigned int cvt = chr;
    while (cvt > 0) {
      *ptr-- = (unsigned char)(cvt & 255);
      cvt    = cvt >> 8;
    }
    printf("    [%s] 0x%04x = %u\n", out, chr, num);
  }

  NVStrings::destroy(dstrs);
  return 0;
}