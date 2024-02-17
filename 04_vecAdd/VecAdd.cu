/********************************************************************************************
Author : swapnil-vh
Date   : 17/02/2024
Description : First parallel processing program
********************************************************************************************/

/*******************************************************************************************
Some Imp Notes :

API website : https://docs.nvidia.com/cuda/cuda-runtime-api/

compilation command : nvcc -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -o VecAdd.out VecAdd.cu

__global__ : Tells to compiler that the function starts with this identifier will be executing on the device(GPU)
			 Instead of the host, so the compiler compiles the function specific to device.
<<< 1,1 >>> : <<<  NumberOfBBlocks, ThreadsPerBlock >>>

__host__ __device__ cudaError_t cudaMalloc ( void** devPtr, size_t size ) :
			Allocates memory at device side of size and returns the reference to *devPtr

__host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) :
			Bridge function topo copy content of memory from host to device or vise versa by inputting value to the kind

 ********************************************************************************************/

/*Header file inclusion area */
#include <iostream>
#include <chrono>

static const int size = 1066777;

/*
GPU kernel function that be called from Host process and will execute on the GPU cores [ALUs].
*/
__global__ void vecAdd (int *v1, int *v2, int *result, int N) {
	
	int idx = blockIdx.x;
	if(idx < N)
		result[idx] = v1[idx] + v2[idx];
}

float goldenCompare(int *vec1, int *vec2, int size)
{
	int idx =0;
	for(auto i=0; i<size; i++)
	{
		if(vec1[i] == vec2[i]) idx++;
	}

	return ((idx * 1.0f) / size) * 100.0f;
}

std::size_t cpuVecAdd(int *vec1, int *vec2, int *result, int size)
{
	if(!vec1 || !vec2 || !result || !size)
	{
		std::cout << "Invalid input vector info to calculate CPU VecAdd" << std::endl;
		return 0;
	} 

	auto startTime = std::chrono::high_resolution_clock::now();
	
	for(auto i=0; i<size; i++)
		result[i] = vec1[i] + vec2[i];

	auto elapsedNanoSecTime = std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::high_resolution_clock::now() - startTime).count();
	
	return elapsedNanoSecTime;
}

std::size_t cudaVecAdd(int *vec1, int *vec2, int *result, int size)
{
	if(!vec1 || !vec2 || !result )
	{
		std::cout << "Invalid input vector info to calculate cuda VecAdd" << std::endl;
		return 0;
	}

	auto startTime = std::chrono::high_resolution_clock::now();

	vecAdd <<< size, 1 >>> (vec1, vec2, result, size);

	auto elapsedNanoSecTime = std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::high_resolution_clock::now() - startTime).count();
	
	return elapsedNanoSecTime;
}

/*
0Entry point function 
*/
int main( void ) {
	
	//declare and init host vectors
	float accuracy;
	int *vec1, *vec2, *result;
	std::size_t cpuTime;

	vec1 = new int[size];
	vec2 = new int[size];
	result = new int[size];

	for(auto i=0; i<size; i++)
	{
		vec1[i] = rand() % 100;
		vec2[i] = rand() % 100;
	}
	
	cpuTime = cpuVecAdd(vec1, vec2, result, size);

	//declare device vector pointers
	int *devvec1, *devvec2, *devresult;
	int *devToHostResult = new int[size];
	std::size_t gpuTime;

	auto ret = cudaMalloc(&devvec1, (sizeof(int) * size));
	if(ret != cudaError::cudaSuccess) goto deinit_blk;
	
	ret = cudaMalloc(&devvec2, sizeof(int) * size);
	if(ret != cudaError::cudaSuccess) goto deinit_blk;
	
	
	ret = cudaMalloc(&devresult, sizeof(int) * size);
	if(ret != cudaError::cudaSuccess) goto deinit_blk;

	//copy vector data from host to device
	ret = cudaMemcpy(devvec1, vec1, sizeof(int) * size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	if(ret != cudaError::cudaSuccess) goto deinit_blk;

	ret = cudaMemcpy(devvec2, vec2, sizeof(int) * size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	if(ret != cudaError::cudaSuccess) goto deinit_blk;

	gpuTime = cudaVecAdd(devvec1, devvec2, devresult, size);
	ret = cudaMemcpy(devToHostResult, devresult, sizeof(int) * size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	if(ret != cudaError::cudaSuccess) goto deinit_blk;

	accuracy = goldenCompare(result, devToHostResult, size);

	printf("\nSummary of the vector calculation of size %d: \n", size);
	printf("Time taken by CPU vector calculation is [milli sec] : %.3f\n", cpuTime/1000000.0f);
	printf("Time taken by GPU vector calculation is [milli sec] : %.3f\n", gpuTime/1000000.0f);
	printf("GPU computation accuracy                            : %.2f\n", accuracy);

deinit_blk :
	if(devToHostResult) delete devToHostResult;
	if(devresult) cudaFree(devresult);
	if(devvec2) cudaFree(devvec2);
	if(devvec1) cudaFree(devvec1);
	if(result) delete result;
	if(vec2) delete vec2;
	if(vec1) delete vec1;

	return 0; 
}

