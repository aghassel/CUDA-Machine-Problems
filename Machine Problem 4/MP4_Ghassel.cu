/*
Name:		Abdellah Ghassel
Student #:	20230384

Note: matrix multiplication from Professor's slides
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// Define tolerance for comparing CPU and GPU results
#define TOLERANCE 0.01

// Function to fill a matrix with random float values
void fillRandMatrix(float *matrix, const int N, const int M)
{
	for (int i = 0; i < (N * M); i++)
		matrix[i] = (float)(rand()) / RAND_MAX;
}

// Matrix multiplication on the CPU
void CPUMatrixMult(float *P, float *M, float *N, int width)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float sum = 0;
			for (int k = 0; k < width; k++)
			{
				sum += M[i * width + k] * N[k * width + j];
			}
			P[i * width + j] = sum;
		}
	}
}

// Tiled matrix multiplication kernel
__global__ void matrixMultTiled(float *P, float *M, float *N, int width, int tileWidth)
{
	// Shared memory for tile data
	extern __shared__ float shared_mem[];
	float *Mds = &shared_mem[0];
	float *Nds = &shared_mem[tileWidth * tileWidth];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * tileWidth + ty;
	int col = bx * tileWidth + tx;

	float p_val = 0;

	// Iterate over tiles
	for (int ph = 0; ph < (width) / tileWidth; ++ph)
	{
		Mds[ty * tileWidth + tx] = M[row * width + ph * tileWidth + tx];
		Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * width + col];
		__syncthreads();

		// Perform matrix multiplication for this thread's assigned element using the loaded tiles
		for (int k = 0; k < tileWidth; k++)
		{
			p_val += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
		}
		// Synchronize threads again before moving to the next tile
		__syncthreads();
	}
	P[row * width + col] = p_val;
}

// Function to verify if the CPU and GPU results are within the given tolerance
void verifyTolerance(float *CPU, float *GPU, int M, int N)
{
	bool testPassed = true;

	for (int i = 0; i < (M * N); i++)
	{
		if (fabs(CPU[i] - GPU[i]) > TOLERANCE)
		{
			testPassed = false;
			break;
		}
	}

	if (testPassed)
	{
		printf("TEST PASSED\n\n");
	}
	else
	{
		printf("TEST FAILED\n\n");
	}
}

// Bonus: Tiled matrix multiplication kernel for rectangular matrices
__global__ void matrixMultTiledRect(float *P, float *M, float *N, int M_height, int M_width, int N_width, int tileWidth, int tileHeight)
{
	// Shared memory for tile data
	extern __shared__ float shared_mem[];
	float *Mds = &shared_mem[0];
	float *Nds = &shared_mem[tileWidth * tileHeight];

	// Calculate block and thread indices
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Calculate row and col indices for this thread
	int row = by * tileHeight + ty;
	int col = bx * tileWidth + tx;

	float p_val = 0;

	// Iterate over tiles
	for (int ph = 0; ph < (M_width + tileWidth - 1) / tileWidth; ++ph)
	{
		// Load M and N tiles to shared memory with boundary checks
		if (row < M_height && ph * tileWidth + tx < M_width)
		{
			Mds[ty * tileWidth + tx] = M[row * M_width + ph * tileWidth + tx];
		}
		else
		{
			Mds[ty * tileWidth + tx] = 0;
		}

		if (col < N_width && ph * tileWidth + ty < M_width)
		{
			Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * N_width + col];
		}
		else
		{
			Nds[ty * tileWidth + tx] = 0;
		}

		// Synchronize threads to make sure shared memory is filled
		__syncthreads();

		// Perform matrix multiplication for this thread's assigned element using the loaded tiles
		for (int k = 0; k < tileWidth; k++)
		{
			p_val += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
		}

		// Synchronize threads again before moving to the next tile
		__syncthreads();
	}

	// Store the result in the output matrix with boundary checks
	if (row < M_height && col < N_width)
	{
		P[row * N_width + col] = p_val;
	}
}

// Bonus: CPU Multiplication for Rectangular Matrices
void CPUMatrixMultRect(float *P, float *M, float *N, int M_height, int M_width, int N_width)
{
	for (int i = 0; i < M_height; i++)
	{
		for (int j = 0; j < N_width; j++)
		{
			float sum = 0;
			for (int k = 0; k < M_width; k++)
			{
				sum += M[i * M_width + k] * N[k * N_width + j];
			}
			P[i * N_width + j] = sum;
		}
	}
}

int main()
{
	// Matrix sizes and tile widths to test
	int matrixSizes[] = {125, 250, 500, 1000, 2000};
	int tileWidths[] = {2, 5, 10, 20, 25};

	// Iterate over matrix sizes
	for (int matrixSize : matrixSizes)
	{
		printf("======= Matrix size: %d x %d =======\n", matrixSize, matrixSize);

		int size = matrixSize * matrixSize * sizeof(float);

		// Allocate memory for input and output matrices
		float *M = (float *)malloc(size);
		float *N = (float *)malloc(size);
		float *P = (float *)malloc(size);
		float *P_cpu = (float *)malloc(size);

		// Initialize input matrices with random values
		fillRandMatrix(M, matrixSize, matrixSize);
		fillRandMatrix(N, matrixSize, matrixSize);

		// Perform CPU matrix multiplication and measure the time
		clock_t cpuStart = clock();
		CPUMatrixMult(P_cpu, M, N, matrixSize);
		float cpuElapsedTime = (float)(clock() - cpuStart) * 1000.0f / CLOCKS_PER_SEC;
		printf("\tCPU time: %.2f ms\n\n", cpuElapsedTime);

		// Allocate memory on the GPU
		float *d_M, *d_N, *d_P;
		cudaMalloc((void **)&d_M, size);
		cudaMalloc((void **)&d_N, size);
		cudaMalloc((void **)&d_P, size);

		// Copy input matrices to the GPU
		cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

		// Iterate over tile widths
		for (int tileWidth : tileWidths)
		{
			// Set up block and grid sizes
			dim3 blockSize(tileWidth, tileWidth);
			dim3 gridSize((matrixSize + blockSize.x - 1) / blockSize.x, (matrixSize + blockSize.y - 1) / blockSize.y);
			size_t shared_mem_size = 2 * tileWidth * tileWidth * sizeof(float);

			// Set up CUDA events for timing
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float gpuElapsedTime;

			// Record start time, run GPU matrix multiplication, and record end time
			cudaEventRecord(start, 0);
			matrixMultTiled<<<gridSize, blockSize, shared_mem_size>>>(d_P, d_M, d_N, matrixSize, tileWidth);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&gpuElapsedTime, start, stop);

			// Copy result back to host memory
			cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

			// Print timing and correctness results
			printf("Tile width: %d\t", tileWidth);
			printf("  GPU time: %.2f ms\n", gpuElapsedTime);
			verifyTolerance(P_cpu, P, matrixSize, matrixSize);
			printf("\n");
		}
	}

	// Bonus: Rectangular matrices
	int matrixSizesBonus[][4] = {
		{350, 400, 400, 500},
		{1900, 1600, 1600, 1300},
	};

	int tileWidth = 8;
	int tileHeight = 15;

	// Iterate over matrix dimensions
	for (int i = 0; i < 2; i++)
	{
		int M_height = matrixSizesBonus[i][0];
		int M_width = matrixSizesBonus[i][1];
		int N_height = matrixSizesBonus[i][2];
		int N_width = matrixSizesBonus[i][3];

		printf("======= Bonus: Matrix size: %d x %d multiplied by %d x %d =======\n", M_height, M_width, N_height, N_width);

		// Allocate memory for input and output matrices
		float *M = (float *)malloc(M_height * M_width * sizeof(float));
		float *N = (float *)malloc(N_height * N_width * sizeof(float));
		float *P = (float *)malloc(M_height * N_width * sizeof(float));
		float *P_cpu = (float *)malloc(M_height * N_width * sizeof(float));

		// Initialize input matrices with random values
		fillRandMatrix(M, M_height, M_width);
		fillRandMatrix(N, N_height, N_width);

		// Perform CPU matrix multiplication and measure the time
		clock_t cpuStart = clock();
		CPUMatrixMultRect(P_cpu, M, N, M_height, M_width, N_width);
		float cpuElapsedTime = (float)(clock() - cpuStart) * 1000.0f / CLOCKS_PER_SEC;
		printf("\tCPU time: %.2f ms\n\n", cpuElapsedTime);

		// Allocate memory on the GPU
		float *d_M, *d_N, *d_P;
		cudaMalloc((void **)&d_M, M_height * M_width * sizeof(float));
		cudaMalloc((void **)&d_N, N_height * N_width * sizeof(float));
		cudaMalloc((void **)&d_P, M_height * N_width * sizeof(float));

		// Copy input matrices to the GPU
		cudaMemcpy(d_M, M, M_height * M_width * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_N, N, N_height * N_width * sizeof(float), cudaMemcpyHostToDevice);

		// tileWidth = tileSize[i];

		dim3 blockSize(tileWidth, tileHeight);
		dim3 gridSize((N_width + blockSize.x - 1) / blockSize.x, (M_height + blockSize.y - 1) / blockSize.y);
		size_t shared_mem_size = tileWidth * tileHeight * sizeof(float) * 2;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float gpuElapsedTime;

		cudaEventRecord(start, 0);
		matrixMultTiledRect<<<gridSize, blockSize, shared_mem_size>>>(d_P, d_M, d_N, M_height, M_width, N_width, tileWidth, tileHeight);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpuElapsedTime, start, stop);

		// Copy result back to host memory
		cudaMemcpy(P, d_P, M_height * N_width * sizeof(float), cudaMemcpyDeviceToHost);

		// Print timing and correctness results
		printf("Tile width: %d x %d\t", tileWidth, tileHeight);
		printf("  GPU time: %.2f ms\n", gpuElapsedTime);
		verifyTolerance(P_cpu, P, M_height, N_width);
		printf("\n");

		// Free GPU memory
		cudaFree(d_M);
		cudaFree(d_N);
		cudaFree(d_P);

		// Free host memory
		free(M);
		free(N);
		free(P);
		free(P_cpu);
	}

	return 0;
}
