/*
Name:		Abdellah Ghassel
Student #:	20230384

Note: matrix multiplication from Professor's slides
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono>

// Define tolerance for comparing CPU and GPU results
#define TOLERANCE 0.01

// Function to fill a matrix with random float values
void fillRandMatrix(float *matrix, const int matrixSize)
{
    for (int i = 0; i < (matrixSize * matrixSize); i++)
        matrix[i] = (float)(rand()) / RAND_MAX;
}

// Function to multiply two matrices on CPU
void CPUMatrixMult(float *M, float *N, float *P, int Width)
{
    for (int i = 0; i < Width; i++)
    {
        for (int j = 0; j < Width; j++)
        {
            for (int k = 0; k < Width; k++)
                P[i * Width + j] += M[i * Width + k] * N[k * Width + j];
        }
    }
}

// CUDA kernel for matrix multiplication on GPU
__global__ void matrixMultKernel(float *M, float *N, float *P, int Width)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * Width + col;

    if (row < Width && col < Width)
    {
        P[idx] = 0.0;
        for (int i = 0; i < Width; i++)
            P[idx] += M[row * Width + i] * N[i * Width + col];
    }
}

// Function to verify if the CPU and GPU results are within the given tolerance
void verifyTolerance(float *CPU, float *GPU, int Width)
{
    bool testPassed = true;

    for (int i = 0; i < (Width * Width); i++)
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

// Function to run the GPU matrix multiplication with a given block size
float hostFunc(float *M_CPU, float *N_CPU, float *P_CPU, int blockSize, int Width)
{

    cudaEvent_t start, end;
    float gpuElapsedTime;
    float *M_GPU, *N_GPU, *P_GPU, *gpuProduct;
    size_t size = Width * Width * sizeof(float);

    cudaMalloc((void **)&M_GPU, size);
    cudaMalloc((void **)&N_GPU, size);
    cudaMalloc((void **)&P_GPU, size);
    gpuProduct = (float *)malloc(size);
    memset(gpuProduct, 0.0, size);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaMemcpy(M_GPU, M_CPU, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_GPU, N_CPU, size, cudaMemcpyHostToDevice);

    int numBlocks = Width / blockSize;
    dim3 block(blockSize, blockSize, 1);
    dim3 grid(numBlocks, numBlocks, 1);

    cudaEventRecord(start);
    matrixMultKernel<<<grid, block>>>(M_GPU, N_GPU, P_GPU, Width);
    cudaEventRecord(end);

    cudaEventElapsedTime(&gpuElapsedTime, start, end);
    //make a copy of the result from the GPU Device to host
    cudaMemcpy(gpuProduct, P_GPU, size, cudaMemcpyDeviceToHost);
    verifyTolerance(P_CPU, gpuProduct, Width);

    cudaFree(M_GPU);
    cudaFree(N_GPU);
    cudaFree(P_GPU);
    return gpuElapsedTime;
}

// Function to run CPU and GPU matrix multiplication and compare their results
void output(int Width)
{
    float *M_CPU, *N_CPU, *P_CPU;
    size_t size = Width * Width * sizeof(float);

    // Allocate memory for matrices and fill them with random float values
    M_CPU = (float *)malloc(size);
    N_CPU = (float *)malloc(size);
    P_CPU = (float *)malloc(size);
    fillRandMatrix(M_CPU, Width);
    fillRandMatrix(N_CPU, Width);

    printf("\nMatrix Multiplication: (%dx%d)\t", Width, Width);
    // Run matrix multiplication on CPU and measure time taken
    clock_t cpuStart = clock();
    CPUMatrixMult(M_CPU, N_CPU, P_CPU, Width);
    float elapsedTime = (float)(clock() - cpuStart) * 1000.0f / CLOCKS_PER_SEC;
    printf("\tCPU time: %.2f ms\n\n", elapsedTime);

    int blockSize[] = {1, 2, 4, 10, 20, 25};
    float gpuElapsedTime;

    // Run matrix multiplication on GPU with different block sizes and measure time taken
    for (int i = 0; i < (sizeof(blockSize) / sizeof(blockSize[0])); i++)
    {
        gpuElapsedTime = hostFunc(M_CPU, N_CPU, P_CPU, blockSize[i], Width);
        printf("\tBlock size: %d\tGPU time: %f ms\t\t", blockSize[i], gpuElapsedTime);
    }

    // Free allocated memory
    free(M_CPU);
    free(N_CPU);
    free(P_CPU);
}

int main()
{

    // Array of matrix sizes to test
    int matrixSize[] = {125, 250, 500, 1000, 2000};

    for (int i = 0; i < sizeof(matrixSize) / sizeof(matrixSize[0]); i++)
    {
        output(matrixSize[i]); // Measure time taken for CPU and GPU matrix multiplication

        float *M_CPU, *N_CPU, *M_GPU, *N_GPU;
        size_t size = matrixSize[i] * matrixSize[i] * sizeof(float);
        float time = 0;
        cudaEvent_t start, end;

        // Allocate memory for matrices and fill them with random float values
        M_CPU = (float *)malloc(size);
        N_CPU = (float *)malloc(size);
        cudaMalloc((void **)&M_GPU, size);
        cudaMalloc((void **)&N_GPU, size);
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        fillRandMatrix(M_CPU, matrixSize[i]);
        fillRandMatrix(N_CPU, matrixSize[i]);

        // Measure time taken for host-to-device memory transfer
        cudaEventRecord(start);
        cudaMemcpy(M_GPU, M_CPU, size, cudaMemcpyHostToDevice);
        cudaMemcpy(N_GPU, N_CPU, size, cudaMemcpyHostToDevice);
        cudaEventRecord(end);
        cudaEventElapsedTime(&time, start, end);

        printf("\n\tHost -> Device Transfer: %f ms\n", time);

        // Measure time taken for device-to-host memory transfer
        cudaEventRecord(start);
        cudaMemcpy(M_CPU, M_GPU, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(N_CPU, N_GPU, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(end);
        cudaEventElapsedTime(&time, start, end);

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        printf("\tDevice -> Host Transfer: %f ms\n", time);

        // Free allocated memory
        cudaFree(M_GPU);
        cudaFree(N_GPU);
        free(M_CPU);
        free(N_CPU);
    }
    return 0;
}
