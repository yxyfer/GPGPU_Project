
cpu_test:
	cd build && make main_cpu && ./main_cpu ../images/base.png ../images/obj.png

gpu_test:
	cd build && make main_gpu && ./main_gpu ../images/base.png ../images/obj.png

bench:
	cd build && make bench && ./bench 

clean:
	rm -rf build/ 
