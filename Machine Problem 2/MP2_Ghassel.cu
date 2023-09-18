/*
Name:		Abdellah Ghassel
Student #:	20230384

Note: matrix addition from Professor's slides
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// constants
#define BLOCKDIM 16
#define TOLERANCE 0.01

// function declarations
void getMatrixElementAddition(float *d_A, float *d_B, float *d_C, int N);
void getMatrixRowAddition(float *d_A, float *d_B, float *d_C, int N);
void getMatrixColAddition(float *d_A, float *d_B, float *d_C, int N);
void hostFunc(float *A, float *B, float *C, int N, void (*addHandler)(float *, float *, float *, int));

// CUDA kernel for element-wise matrix addition
__global__ void matrixAddElements(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// CUDA kernel for row-wise matrix addition
__global__ void matrixAddRows(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N)
    {
        for (int col = 0; col < N; col++)
        {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

// CUDA kernel for column-wise matrix addition
__global__ void matrixAddCols(float *A, float *B, float *C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N)
    {
        for (int row = 0; row < N; row++)
        {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

// invoke the specified addition kernel
void getMatrixElementAddition(float *d_A, float *d_B, float *d_C, int N)
{
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), (int)ceil(N / (float)threadsPerBlock.y));
    matrixAddElements<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

void getMatrixRowAddition(float *d_A, float *d_B, float *d_C, int N)
{
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    dim3 numBlocks(1, (int)ceil(N / (float)threadsPerBlock.y));
    matrixAddRows<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

void getMatrixColAddition(float *d_A, float *d_B, float *d_C, int N)
{
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), 1);
    matrixAddCols<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

int main()
{
    float *A, *B, *C;
    int matrixSize[5] = {125, 250, 500, 1000, 2000};

    for (int i = 0; i < sizeof(matrixSize) / sizeof(matrixSize[0]); i++)
    {
        int N = matrixSize[i];
        size_t size = N * N * sizeof(float);
        A = (float *)malloc(size);
        B = (float *)malloc(size);
        C = (float *)malloc(size);

        printf("Matrix (%d, %d)\n", N, N);
        printf("\tElment Addition\n");
        hostFunc(A, B, C, N, getMatrixElementAddition);
        printf("\tRow Addition\n");
        hostFunc(A, B, C, N, getMatrixRowAddition);
        printf("\tColumn Addition\n");
        hostFunc(A, B, C, N, getMatrixColAddition);
        printf("\n\n");
        free(A);
        free(B);
        free(C);
    }
    return 0;
}

// host function to handle data transfer and kernel invocation
void hostFunc(float *A, float *B, float *C, int N, void (*addHandler)(float *, float *, float *, int))
{
    // set up matricies with random values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i * N + j;
            A[index] = (float)rand() / (float)RAND_MAX;
            B[index] = (float)rand() / (float)RAND_MAX;
            C[index] = 0.0f;
        }
    }
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // allocate memory on device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // GPU addition timing
    float time = 0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    addHandler(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("GPU time: %f ms\n", time);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // CPU timing
    time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    float *C_CPU = (float *)malloc(size);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = i + j * N;
            C_CPU[index] = A[index] + B[index];
        }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("CPU time: %f ms\n", time);

    // determine error tolerance
    float errorTolerance = 0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            int idx = i + j * N;
            errorTolerance = fmax(errorTolerance, fabs(C[idx] - C_CPU[idx]));
        }
    }

    // determine if errors occured
    if (errorTolerance < TOLERANCE)
    {
        printf("\tTEST PASSED\n\n");
    }
    else
    {
        printf("\tTEST FAILED\n\n");
    }
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
