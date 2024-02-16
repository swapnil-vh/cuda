/********************************************************************************************
Author : swapnil-vh
Date   : 16/02/2024
Description : First cuda program to understand the cuda kernel function execution
*********************************************************************************************/

/*******************************************************************************************
Some Imp Notes :

API website : https://docs.nvidia.com/cuda/cuda-runtime-api/

compilation command : nvcc -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -o SimpleMath.out SimpleMath.cu

__global__ : Tells to compiler that the function starts with this identifier will be executing on the device(GPU)
			 Instead of the host, so the compiler compiles the function specific to device.
<<< 1,1>>> : <<<  NumberOfBBlocks, ThreadsPerBlock >>>

__host__ __device__ cudaError_t cudaMalloc ( void** devPtr, size_t size ) :
			Allocates memory at device side of size and returns the reference to *devPtr

__host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) :
			Bridge function topo copy content of memory from host to device or vise versa by inputting value to the kind

 ********************************************************************************************/

/*Header file inclusion area */
#include <iostream>

/* User defined data types */
enum class MathExpression {
	ADDITION = 0,
	SUBSTRACTION,
	MULTIPLICATION,
	DIVISION,
};

/*
GPU kernel function that be called from Host process and will execute on the GPU cores [ALUs].
*/
__global__ void SimpleMath (MathExpression expr, int x, int y, int *result ) {
	
	if 		(expr == MathExpression::ADDITION) 			*result = x + y;
	else if (expr == MathExpression::SUBSTRACTION)		*result = x - y;
	else if (expr == MathExpression::MULTIPLICATION) 	*result = x * y;
	else if (expr == MathExpression::DIVISION)			*result = x / y;
	else 												*result = 0;
}

/*
Entry point function 
*/
int main( void ) {

	int hostContainer = 0;
	int *gpuMemory = nullptr;

	//allocate memory at GPU side to retrive the result of the calculation
	if(cudaError::cudaSuccess != cudaMalloc(&gpuMemory, sizeof(int)))
	{
		std::cout << "Failed to allocate memory for result at GPU side" << std::endl;
		exit (1);
	}
		
	//addition Block
	{
		//call the GPU kernel function to do calculation	
		SimpleMath <<<1,1>>> (MathExpression::ADDITION, 10, 20, gpuMemory);
	
		//copy result from GPU memory to host memory
		if(cudaError::cudaSuccess != cudaMemcpy( &hostContainer, gpuMemory, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout  << "Failed to copy addition result from GPU memory to host memory " << std::endl;
			cudaFree(gpuMemory);
			exit (1);
		}
		std::cout << "Addition of 10 + 20 returned by GPU is : " << hostContainer << std::endl;
	}

	//substraction block 
	{
		//call the GPU kernel function to do calculation	
		SimpleMath <<<1,1>>> (MathExpression::SUBSTRACTION, 10, 20, gpuMemory);
	
		//copy result from GPU memory to host memory
		if(cudaError::cudaSuccess != cudaMemcpy( &hostContainer, gpuMemory, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout  << "Failed to copy substaction result from GPU memory to host memory " << std::endl;
			cudaFree(gpuMemory);
			exit (1);
		}
		std::cout << "Substraction of 10,20 returned by GPU is : " << hostContainer << std::endl;
	}

	//multiplication block
	{
		//call the GPU kernel function to do calculation	
		SimpleMath <<<1,1>>> (MathExpression::MULTIPLICATION, 10, 20, gpuMemory);
	
		//copy result from GPU memory to host memory
		if(cudaError::cudaSuccess != cudaMemcpy( &hostContainer, gpuMemory, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout  << "Failed to copy multiplication result from GPU memory to host memory " << std::endl;
			cudaFree(gpuMemory);
			exit (1);
		}
		std::cout << "Multiplication of 10,20 returned by GPU is : " << hostContainer << std::endl;
	}


	//division block
	{
		//call the GPU kernel function to do calculation	
		SimpleMath <<<1,1>>> (MathExpression::DIVISION, 20, 10, gpuMemory);
	
		//copy result from GPU memory to host memory
		if(cudaError::cudaSuccess != cudaMemcpy( &hostContainer, gpuMemory, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost))
		{
			std::cout  << "Failed to copy division result from GPU memory to host memory " << std::endl;
			cudaFree(gpuMemory);
			exit (1);
		}
		std::cout << "Division of 20,10 returned by GPU is : " << hostContainer << std::endl;
	}

	//free GPU memory
	cudaFree(gpuMemory);

	return 0; 
}

