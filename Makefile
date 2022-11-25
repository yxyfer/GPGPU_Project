build:
	mkdir build && cd build && cmake ..

cpu_test:
	cd build && make main_cpu -j && ./main_cpu ../images/base.png ../images/obj.png

gpu_test:
	cd build && make main_gpu -j && ./main_gpu ../images/base.png ../images/obj.png

bench_cpu:
	cd build && make bench_cpu -j && ./bench_cpu

clean:
	rm -rf build/ 
