/*
compilation command : nvcc -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -o test.out HelloWorld.cu

__global__ : Tells to compiler that the function starts with this identifier will be executing on the device(GPU)
			 Instead of the host, so the compiler compiles the function specific to device.
<<< 1,1>>> : <<<  NumberOfBBlocks, ThreadsPerBlock >>>
 */
#include <iostream>

__global__ void kernel( void ) {

}

int main( void ) {
	kernel<<<1,1>>>();
	std::cout << "Hello world" << std::endl;
	return 0; 
}

