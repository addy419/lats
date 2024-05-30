# lats: Memory Latency Benchmarking

The value of `STRIDE` should be chosen to defeat prefetching on the test architecture.

To run on PVC:

```sh
# Source your compilers
make sycl-usm
ONEAPI_DEVICE_SELECTOR=level_zero:0.0 ./run.sycl
# Results are in lat.csv
```
