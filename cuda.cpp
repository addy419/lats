#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "profiler.h"

#define KiB 1024
#define MiB (KiB*KiB)
#define GiB (KiB*KiB*KiB)
#define GHz (1000L*1000L*1000L)
#define NOUTER_ITERS 1L
#define NINNER_ITERS 50L
#define CACHE_LINE_LENGTH 128L
#define STRIDE_START 5L
#define STRIDE_END 5L
#define ALLOCATION_START (512L)
#define ALLOCATION_END (512L * MiB)
#define SIMD_SIZE 16

#define MEM_LD_LATENCY
//#define INST_LATENCY


__inline__ __device__ long long int warpReduceMin(long long int val) {
    // Full warp mask

    // Perform reduction within a warp

    return val;
}


__global__ void lat(const size_t ncache_lines, char* P, char* dummy, long long int* cycles)
{
  const size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid > warpSize) {
    return;
  }


#if defined(MEM_LD_LATENCY)

  char** p0 = (char**)&P[gid*8];

  // Warmup
  for(size_t n = 0; n < ncache_lines; ++n) {
    p0 = (char**)*p0;
  }

  long long int t0 = clock64();

  char** p1 = (char**)&P[gid*8];

#pragma unroll 64
  for(size_t n = 0; n < ncache_lines*NINNER_ITERS; ++n) {
    p1 = (char**)*p1;
  }

  *dummy = *(char*)p0 + *(char*)p1;

#elif defined(INST_LATENCY)

  long long int t0 = clock64();

  float a = 0.9999f;
  for(int n = 0; n < NINNER_ITERS; ++n) {
#if 0
    a += 0.99991f;
    a *= 0.9991f;
    a += a * 0.999991f;
    a = sqrtf(a);
    a /= 0.999991f;
#endif // if 0
  }

#if 0
  printf("%.7f\n", a);
#endif // if 0

  *dummy = (char)a;

#endif

  long long int t1 = clock64()-t0;

  unsigned mask = 0xFFFFFFFF;
  for (int offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
      t1 = min(t1, __shfl_down_sync(mask, t1, offset));
  }

  if(gid == 0) {
    *cycles += t1;
  }
}

__global__ void make_ring(const size_t ncache_lines, const size_t as, const size_t st, char* P)
{
  const size_t gid = blockDim.x*blockIdx.x+threadIdx.x;
  if(gid > warpSize) {
    return;
  }

  // Create a ring of pointers at the cache line granularity
  for(size_t i = 0; i < ncache_lines; ++i) {
    *(char**)&P[(i*CACHE_LINE_LENGTH)+(gid*8)] = &P[(((i+st)*CACHE_LINE_LENGTH)+(gid*8))%as];
  }
}

int main() {

  struct Profile profile;
  struct ProfileEntry* pe = &profile.profiler_entries[0];
  pe->time = 0.0;

  // Initialise
  char* P;
  char* dummy;
  cudaMalloc((void**)&P, ALLOCATION_END);
  cudaMalloc((void**)&dummy, 1);
  printf("Allocating %lu MiB\n", ALLOCATION_END/MiB);

  // Open files
  FILE* nfp = fopen("/dev/null", "a");
  FILE* fp = fopen("lat.csv", "a");

  long long int* d_cycles;
  long long int* d_cycles_dummy;
  cudaMalloc(&d_cycles, sizeof(long long int));
  cudaMalloc(&d_cycles_dummy, sizeof(long long int));

  for(size_t st = STRIDE_START; st <= STRIDE_END; ++st) {
    for(size_t as = ALLOCATION_START; as <= ALLOCATION_END; as *= 2L) {

      const size_t ncache_lines = as/CACHE_LINE_LENGTH;

#if defined(MEM_LD_LATENCY)
      make_ring<<<1,SIMD_SIZE>>>(ncache_lines, as, st, P);
#endif

      // Zero the cycles
      long long int h_cycles = 0;
      cudaMemcpy(d_cycles, &h_cycles, sizeof(long long int), cudaMemcpyHostToDevice);

      // Perform the test
      START_PROFILING(&profile);
      for(size_t i = 0; i < NOUTER_ITERS; ++i) {
        lat<<<1,SIMD_SIZE>>>(ncache_lines, P, dummy, d_cycles);
      }
      cudaDeviceSynchronize();
      STOP_PROFILING(&profile, "p");

      // Bring the cycle count back from the device
      cudaMemcpy(&h_cycles, d_cycles, sizeof(long long int), cudaMemcpyDeviceToHost);

      printf("Elapsed Clock Cycles %lu\n", h_cycles);

#if defined(MEM_LD_LATENCY)

      double loads = (double)NOUTER_ITERS*ncache_lines*NINNER_ITERS;
      double cycles_load = ((double)h_cycles/loads);
      printf("Array Size %.3fMB Stride %d Cache Lines %d Time %.12fs\n", 
          (double)as/MiB, st, ncache_lines, pe->time);
      double loads_s = loads / pe->time;
      double cycles_s = 1.48*GHz;
      double cycles_load2 = (double)(cycles_s / loads_s);
      printf("Loads = %lu\n", loads);
      printf("Cycles / Load = %.4f\n", cycles_load);
      //printf("backup = %.4f\n", cycles_load2);
      fprintf(fp, "%d,%lu,%.4f\n", st, as, cycles_load);

#elif defined(INST_LATENCY)

      size_t ops = NOUTER_ITERS*NINNER_ITERS;
      printf("Ops %lu\n", ops);
      printf("Cycles / Op %.4f\n", h_cycles/(double)ops);

#endif

      h_cycles = 0;
      cudaMemcpy(d_cycles, &h_cycles, sizeof(long long int), cudaMemcpyHostToDevice);

      pe->time = 0.0;
    }
  }

  fclose(nfp);
  fclose(fp);
  cudaFree(P);

  return 0;
}
