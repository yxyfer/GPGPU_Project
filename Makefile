build:
	mkdir build && cd build && cmake ..

cpu_test:
	cd build && make main_cpu -j && ./main_cpu ../images/base.png ../images/obj.png

gpu_test:
	cd build && make main_gpu -j && ./main_gpu ../images/base.png ../images/obj.png

bench_cpu:
	cd build && make bench_cpu -j && ./bench_cpu


bench_gpu:
	cd build && make bench_gpu -j && ./bench_gpu

clean:
	rm -rf build/ 
