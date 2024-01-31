
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "helper_cupti_activity.h"




// Macros
#define ARRAY_SIZE 32
#define THREADS_PER_BLOCK 32

// Kernels
__global__ void VectorAdd( const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        pC[i] = pA[i] + pB[i];
    }
}

static void MainCompute(cudaStream_t stream)
{
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    size_t size = ARRAY_SIZE * sizeof(int);
    int blocksPerGrid = 0;

    CUcontext context;

    // Allocate input vectors pHostA and pHostB in host memory.
    pHostA = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostA);

    pHostB = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostB);

    pHostC = (int *)malloc(size);
    MEMORY_ALLOCATION_CALL(pHostC);

    // Allocate vectors in device memory.
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, size));

    CUpti_ActivityPCSamplingConfig configPC;
    configPC.size = sizeof(CUpti_ActivityPCSamplingConfig);
    configPC.samplingPeriod=CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
    configPC.samplingPeriod2 = 0;
    cuCtxGetCurrent(&context);

    // Configure api needs to be called before activity enable for chips till Pascal chips.
    // Order does not matter for Volta+ chips.
    CUPTI_API_CALL(cuptiActivityConfigurePCSampling(context, &configPC));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceA, pHostA, size, cudaMemcpyHostToDevice, stream));
    RUNTIME_API_CALL(cudaMemcpyAsync(pDeviceB, pHostB, size, cudaMemcpyHostToDevice, stream));

    blocksPerGrid = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    VectorAdd <<< blocksPerGrid, THREADS_PER_BLOCK, 0, stream >>> (pDeviceA, pDeviceB, pDeviceC, ARRAY_SIZE);
    RUNTIME_API_CALL(cudaGetLastError());

    RUNTIME_API_CALL(cudaMemcpyAsync(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost, stream));

    if (stream == 0)
    {
        // Waits for stream tasks to complete.
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e
        RUNTIME_API_CALL(cudaDeviceSynchronize());
    }
    else
    {
        RUNTIME_API_CALL(cudaStreamSynchronize(stream));
    }

    // Free host memory.
    if (pHostA)
    {
        free(pHostA);
    }
    if (pHostB)
    {
        free(pHostB);
    }
    if (pHostC)
    {
        free(pHostC);
    }

    // Free device memory.
    if (pDeviceA)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceA));
    }
    if (pDeviceB)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceB));
    }
    if (pDeviceC)
    {
        RUNTIME_API_CALL(cudaFree(pDeviceC));
    }
}


int main(int argc, char *argv[]) {
    std::cout << "Welcome to XLLM!";

    int deviceNum = 0;
    cudaDeviceProp prop;

// Setup Cupti

    UserData *pUserData = (UserData *)malloc(sizeof(UserData));
    MEMORY_ALLOCATION_CALL(pUserData);

    memset(pUserData, 0, sizeof(UserData));
    pUserData->pPostProcessActivityRecords = NULL;
    pUserData->printActivityRecords        = 1;

    // Common CUPTI Initialization.
    InitCuptiTrace(pUserData, NULL, stdout);

// Print Metadata

    RUNTIME_API_CALL(cudaGetDevice(&deviceNum));
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
    printf("Device Name: %s\n", prop.name);
    printf("Device compute capability: %d.%d\n", prop.major, prop.minor);

// Main compute

    MainCompute(0);



// Clean up


    RUNTIME_API_CALL(cudaDeviceSynchronize());
    RUNTIME_API_CALL(cudaDeviceReset());

    DeInitCuptiTrace();

    exit(EXIT_SUCCESS);



    return 0;
}