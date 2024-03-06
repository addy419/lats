#include <iostream>
#include <sycl/sycl.hpp>
#include "profiler.h"

#define KiB 1024
#define MiB (KiB * KiB)
#define GiB (KiB * KiB * KiB)
#define GHz (1000L * 1000L * 1000L)
#define NOUTER_ITERS 1L
#define NINNER_ITERS 50L
#define CACHE_LINE_LENGTH 128L
#define STRIDE_START 5L
#define STRIDE_END 5L
#define ALLOCATION_START (512L)
#define ALLOCATION_END (512L * MiB)

#define MEM_LD_LATENCY
// #define INST_LATENCY

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL ulong __attribute__((overloadable))
intel_get_cycle_counter(void);
#endif

void lat(const size_t ncache_lines, char *P, char *dummy, long long int *cycles,
         sycl::id<1> tid) {
  const size_t gid = tid[0];
  if (gid > 0) {
    return;
  }

#if defined(MEM_LD_LATENCY)

  char **p0 = (char **)P;

  // Warmup
  for (size_t n = 0; n < ncache_lines; ++n) {
    p0 = (char **)*p0;
  }

#ifdef __SYCL_DEVICE_ONLY__
  ulong t0 = intel_get_cycle_counter();
#endif

  char **p1 = (char **)P;

#pragma unroll 64
  for (size_t n = 0; n < ncache_lines * NINNER_ITERS; ++n) {
    p1 = (char **)*p1;
  }

  *dummy = *(char *)p0 + *(char *)p1;

#elif defined(INST_LATENCY)

#ifdef __SYCL_DEVICE_ONLY__
  ulong t0 = intel_get_cycle_counter();
#endif

  float a = 0.9999f;
  for (int n = 0; n < NINNER_ITERS; ++n) {
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

#ifdef __SYCL_DEVICE_ONLY__
  *cycles += intel_get_cycle_counter() - t0;
#endif
}

void make_ring(const size_t ncache_lines, const size_t as, const size_t st,
               char *P, sycl::id<1> tid) {
  const size_t gid = tid[0];

  if (gid > 0) {
    return;
  }

  // Create a ring of pointers at the cache line granularity
  for (size_t i = 0; i < ncache_lines; ++i) {
    *(char **)&P[(i * CACHE_LINE_LENGTH)] =
        &P[((i + st) * CACHE_LINE_LENGTH) % as];
  }
}

int main() {

  struct Profile profile;
  struct ProfileEntry *pe = &profile.profiler_entries[0];
  pe->time = 0.0;

  try {
    sycl::queue gpuQueue{sycl::gpu_selector_v};

    // Print device
    std::cout << "Running on "
              << gpuQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Initialise
    char *P;
    char *dummy;
    P = sycl::malloc_device<char>(ALLOCATION_END, gpuQueue);
    dummy = sycl::malloc_device<char>(1, gpuQueue);
    std::cout << "Allocating " << ALLOCATION_END / MiB << " MiB\n";

    // Open files
    FILE *nfp = fopen("/dev/null", "a");
    FILE *fp = fopen("lat.csv", "a");

    long long int *d_cycles;
    long long int *d_cycles_dummy;
    d_cycles =
        sycl::malloc_device<long long int>(sizeof(long long int), gpuQueue);
    d_cycles_dummy =
        sycl::malloc_device<long long int>(sizeof(long long int), gpuQueue);

    for (size_t st = STRIDE_START; st <= STRIDE_END; ++st) {
      for (size_t as = ALLOCATION_START; as <= ALLOCATION_END; as *= 2L) {

        const size_t ncache_lines = as / CACHE_LINE_LENGTH;

#if defined(MEM_LD_LATENCY)
        gpuQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task([=]() {
            sycl::id<1> tid = sycl::id<1>(0);
            make_ring(ncache_lines, as, st, P, tid);
          });
        }).wait();
#endif

        // Zero the cycles
        long long int h_cycles = 0;
        gpuQueue.memcpy(d_cycles, &h_cycles, sizeof(long long int)).wait();

        // Perform the test
        START_PROFILING(&profile);
        for (size_t i = 0; i < NOUTER_ITERS; ++i) {
          gpuQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task([=]() {
              sycl::id<1> tid = sycl::id<1>(0);
              lat(ncache_lines, P, dummy, d_cycles, tid);
            });
          }).wait();
        }
        STOP_PROFILING(&profile, "p");

        // Bring the cycle count back from the device
        gpuQueue.memcpy(&h_cycles, d_cycles, sizeof(long long int)).wait();

        std::cout << "Elapsed Clock Cycles " << h_cycles << std::endl;

#if defined(MEM_LD_LATENCY)

        double loads = (double)NOUTER_ITERS * ncache_lines * NINNER_ITERS;
        double cycles_load = ((double)h_cycles / loads);
        printf("Array Size %.3fMB Stride %d Cache Lines %d Time %.12fs\n",
               (double)as / MiB, (int)st, (int)ncache_lines, pe->time);
        double loads_s = loads / pe->time;
        double cycles_s = 1.48 * GHz;
        double cycles_load2 = (double)(cycles_s / loads_s);
        printf("Loads = %lu\n", loads);
        printf("Cycles / Load = %.4f\n", cycles_load);
        // printf("backup = %.4f\n", cycles_load2);
        fprintf(fp, "%d,%lu,%.4f\n", (int)st, as, cycles_load);

#elif defined(INST_LATENCY)

        size_t ops = NOUTER_ITERS * NINNER_ITERS;
        printf("Ops %lu\n", ops);
        printf("Cycles / Op %.4f\n", h_cycles / (double)ops);

#endif

        h_cycles = 0;
        gpuQueue.memcpy(d_cycles, &h_cycles, sizeof(long long int)).wait();

        pe->time = 0.0;
      }
    }

    fclose(nfp);
    fclose(fp);
    sycl::free(P, gpuQueue);
  } catch (sycl::exception &e) {
    /* handle SYCL exception */
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
