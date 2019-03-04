#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "profiler.h"

#define KiB 1024
#define MiB (KiB*KiB)
#define GiB (KiB*KiB*KiB)
#define GHz (1000L*1000L*1000L)
#define CLOCK_SPEED (3.65L * GHz)
#define NITERS 1000L
#define CACHE_LINE_LENGTH 64L
#define STRIDE 5L
#define ALLOCATION (4L*GiB)

int main() {

  struct Profile profile;

  // Initialise
  char* P = (char*)malloc(ALLOCATION);
  for(size_t i = 0; i < ALLOCATION; ++i) {
    P[i] = 0;
  }
  printf("Allocating %lu MiB\n", ALLOCATION/MiB);

  // Open files
  FILE* nfp = fopen("/dev/null", "a");
  FILE* fp = fopen("lat.csv", "a");

  // Begin loop
  for(size_t as = KiB; as <= ALLOCATION; as *= 2L) {

    struct ProfileEntry* pe = &profile.profiler_entries[0];
    pe->time = 0.0;

    const size_t ncache_lines = as/CACHE_LINE_LENGTH;

    // Create a ring of pointers at the cache line granularity
    for(size_t i = 0; i < ncache_lines; ++i) {
      *(char**)&P[(i*CACHE_LINE_LENGTH)] = &P[((i+STRIDE)*CACHE_LINE_LENGTH)%as];
    }

    char** p = (char**)P;

    START_PROFILING(&profile);
    for(size_t i = 0; i < NITERS; ++i) {
#pragma unroll(64)
      for(size_t n = 0; n < ncache_lines; ++n) {
        p = (char**)*p;
      }
    }
    STOP_PROFILING(&profile, "p");

    const size_t load_s = (int64_t)((double)NITERS*ncache_lines / pe->time);
    const double cycles_load = (CLOCK_SPEED/(double)load_s);
    printf("array size %.3fMB stride %d cache_lines %d time %.12fs\n", (double)as/MiB, STRIDE, ncache_lines, pe->time);
    printf("load / s = %lu\n", load_s);
    printf("cycle / load = %.4f\n", cycles_load);

    fprintf(nfp, "result: %p\n", p);
    fprintf(fp, "%d,%lu,%.4f\n", STRIDE, as, cycles_load);

    pe->time = 0.0;
  }

  fclose(nfp);
  fclose(fp);
  free(P);

  return 0;
}
