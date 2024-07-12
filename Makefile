OPTS=-Xlinker -z,noexecstack

cuda:
	nvc -O3 -std=gnu99 -c profiler.c -o profiler.o
	nvc++ -O3 -cuda -gpu=$(ARCH) -fast -DENABLE_PROFILING -c cuda.cpp -o obj_gpu.o
	nvc++ -O3 -cuda -gpu=$(ARCH) $(OPTS) obj_gpu.o profiler.o -lrt -o run.cuda

hip:
	gcc -O3 -std=gnu99 -c profiler.c -o profiler.o
	hipcc -O3 $(HIPOPTS) -DENABLE_PROFILING -c hip.cpp -o obj_gpu.o
	hipcc -O3 $(HIPOPTS) obj_gpu.o profiler.o -o run.hip

sycl-usm:
	icx -O3 -std=gnu99 -c profiler.c -o profiler.o
	icpx -O3 -fsycl -DENABLE_PROFILING -c sycl-usm.cpp -o obj_sycl.o
	icpx -O3 -fsycl obj_sycl.o profiler.o -o run.sycl

sycl-acc:
	icx -O3 -std=gnu99 -c profiler.c -o profiler.o
	icpx -O3 -fsycl -DENABLE_PROFILING -c sycl-acc.cpp -o obj_sycl.o
	icpx -O3 -fsycl obj_sycl.o profiler.o -o run.sycl

cpu:
	icc -O3 -std=gnu99 -c profiler.c -o profiler.o
	icc -O3 -std=gnu99 -DENABLE_PROFILING -c cpu.c -o obj_cpu.o
	icc -O3 obj_cpu.o profiler.o -o run.cpu

clean:
	rm -rf omp4.exe cuda.exe *.i *.o
