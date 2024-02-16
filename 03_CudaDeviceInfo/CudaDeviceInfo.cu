/********************************************************************************************
Author : swapnil-vh
Date   : 16/02/2024
Description : First cuda program to print cuda devices and properties
*********************************************************************************************/

/*******************************************************************************************
Some Imp Notes :

API website : https://docs.nvidia.com/cuda/cuda-runtime-api/

compilation command : nvcc -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -o CudaDeviceInfo.out CudaDeviceInfo.cu

__host__ __device__ cudaError_t cudaGetDeviceCount ( int* count )
__host__ cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )
********************************************************************************************/

/*Header file inclusion area */
#include <iostream>

/*
Entry point function 
*/
int main( void ) {

	int count = 0;
	
	if((cudaSuccess != cudaGetDeviceCount(&count)) || (count == 0))
	{
		std::cout << "No cuda supported device found " << std::endl;
		exit(1);
	}

	std::cout << "Number of cuda devices found are : " << count << std::endl;

	for(auto i=0; i < count; i++)
	{
		cudaDeviceProp devprop;
		auto err = cudaGetDeviceProperties(&devprop, i);
		if(err != cudaSuccess)
		{
			std::cout << "Invalid device " << i << std::endl;
			continue;
		}
		
		printf("\nProperties for device %d \n", i);
		printf("Device Name : %s \n", devprop.name);
		printf("Total global memory available [bytes] : %lu\n", devprop.totalGlobalMem);
		printf("Total shared memory / block  available [bytes] : %lu \n", devprop.sharedMemPerBlock);
		printf("Number of 32 bit register available / block : %d \n", devprop.regsPerBlock);
		printf("Warp size in thread %d \n", devprop.warpSize);
		printf("Mem pitch : %lu \n", devprop.memPitch);
		printf("Max threads / block : %d \n", devprop.maxThreadsPerBlock);
		printf("Max Threads dimention [Block size] : (x,y,z) : (%d, %d, %d)\n", devprop.maxThreadsDim[0], devprop.maxThreadsDim[1], devprop.maxThreadsDim[2]);	
		printf("Max Grid size [block array] : (x,y,z) : (%d, %d, %d)\n", devprop.maxGridSize[0], devprop.maxGridSize[1], devprop.maxGridSize[2]);
		printf("Clock rate of the device : %d kHz\n", devprop.clockRate);
		printf("Available total constant memory on the device : %lu \n", devprop.totalConstMem);
		printf("version %d.%d \n", devprop.major, devprop.minor);
		printf("Device type : %s \n", devprop.integrated ? "Integrated" : "Descrete");
	}

	return 0; 
}

