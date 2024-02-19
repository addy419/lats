NVOPTS = --use_fast_math #-ptx

gpu:
	gcc -O3 -std=gnu99 -c profiler.c -o profiler.o
	nvcc -O3 -arch=sm_35 $(NVOPTS) -DENABLE_PROFILING -c gpu.cu -o obj_gpu.o
	nvcc -O3 -arch=sm_35 obj_gpu.o profiler.o -lrt -o run.gpu

sycl:
	icx -O3 -std=gnu99 -c profiler.c -o profiler.o
	icpx -O3 -fsycl -DENABLE_PROFILING -c sycl.cpp -o obj_sycl.o
	icpx -O3 -fsycl obj_sycl.o profiler.o -o run.sycl

sycl-acc:
	icx -O3 -std=gnu99 -c profiler.c -o profiler.o
	icpx -O3 -fsycl -DENABLE_PROFILING -c sycl-acc.cpp -o obj_sycl.o
	icpx -O3 -fsycl obj_sycl.o profiler.o -o run.sycl-acc

cpu:
	icc -O3 -std=gnu99 -c profiler.c -o profiler.o
	icc -O3 -std=gnu99 -DENABLE_PROFILING -c cpu.c -o obj_cpu.o
	icc -O3 obj_cpu.o profiler.o -o run.cpu

clean:
	rm -rf omp4.exe cuda.exe *.i *.o
