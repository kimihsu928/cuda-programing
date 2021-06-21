#include "parameters.h"
#include <stdio.h>
__global__ void cuda_kernel(float *B, float *A, IndexSave *dInd)
{
	// complete cuda kernel function
	int TotalThread = blockDim.x * gridDim.x;
	int stripe = SIZE / TotalThread;
	int head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
	int LoopLim = head + stripe;

	for (int i = head; i < LoopLim; i++)
	{
		dInd[i].blockInd_x = blockIdx.x;
		dInd[i].threadInd_x = threadIdx.x;
		dInd[i].head = head;
		dInd[i].stripe = stripe;

		B[i] = (B[i] - A[i]) * (B[i] - A[i]);
	}
};

float GPU_kernel(float *B, float *A, IndexSave *indsave)
{
	float *dA, *dB;
	IndexSave *dInd;

	// Creat Timing Event
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate Memory Space on Device

	// Allocate Memory Space on Device (for observation)
	cudaMalloc((void **)&dInd, sizeof(IndexSave) * SIZE);

	// Copy Data to be Calculated
	cudaMalloc((void **)&dB, sizeof(float) * SIZE);
	cudaMalloc((void **)&dA, sizeof(float) * SIZE);
	// Copy Data to be Calculated
	cudaMemcpy(dB, B, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dA, A, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	// Copy Data (indsave array) to device
	cudaMemcpy(dInd, indsave, sizeof(IndexSave) * SIZE, cudaMemcpyHostToDevice);

	// Start Timer
	cudaEventRecord(start, 0);

	// Lunch Kernel
	dim3 dimGrid(4);
	dim3 dimBlock(4);
	cuda_kernel<<<dimGrid, dimBlock>>>(dB, dA, dInd);
	// Stop Timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Copy Output back
	cudaMemcpy(B, dB, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(A, dA, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(indsave, dInd, sizeof(IndexSave) * SIZE, cudaMemcpyDeviceToHost);

	// Release Memory Space on Device
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dInd);

	// Calculate Elapsed Time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	return elapsedTime;
}
