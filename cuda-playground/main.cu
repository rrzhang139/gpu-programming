#include <stdio.h>

__global__ void squareKernel(int *gpuData, int length) {
  int idx = threadIdx.x;
  if (idx < length) {
      gpuData[idx] = gpuData[idx] * gpuData[idx];
  }
}

int main() {
  int N = 10;
  //////////////////////////////////
  // Phase 1: pointers and mem alloc
  //////////////////////////////////
  // A pointer is an address in memory.
  int* a = 0;
  // malloc allocates memory in the cpu and returns the first byte of the memory
  int* mem = (int*) malloc(N*sizeof(int));

  for (int i = 0; i < N; i++) {
    mem[i] = 2;
  }


  ///////////////////////////

  // Great! Now that you've allocated memory on the CPU, let's move on to allocating memory on the GPU.
  // In CUDA, you can use the cudaMalloc function to allocate memory on the GPU. The function takes two arguments:
  // 1. A pointer to a pointer, which will store the address of the allocated memory on the GPU.
  // 2. The size of the memory to allocate, in bytes (similar to malloc).

// In this example, cudaMalloc will hold the address of the memory address in cpu that is allocated in GPU memory
  int* gpuMem;
  cudaError_t err = cudaMalloc(&gpuMem, N*sizeof(int));

  ///////////////////////////

  // Excellent! Now that you've allocated memory on both the CPU and the GPU, the next step is to copy data between them.
  // To copy data from the CPU to the GPU or vice versa, you can use the cudaMemcpy function. This function takes four arguments:
  // 1. The destination pointer.
  // 2. The source pointer.
  // 3. The size of the data to copy, in bytes.
  // 4. The direction of the copy, specified as one of the following constants: 

  cudaMemcpy(gpuMem, mem, N*sizeof(int), cudaMemcpyHostToDevice);

  ///////////////////////////

  // Fantastic! Now that you've allocated memory and copied data to the GPU, the next step is to perform computations on this data using a CUDA kernel.

  // A CUDA kernel is a function that you write to run in parallel on the GPU. 
  // You define a kernel in your .cu file using the __global__ keyword before the function definition.

  // This is how you call it: func<<<blocks, threads>>>
  // Blocks and Threads: In CUDA, the execution configuration is organized into a grid of blocks, and each block contains multiple threads. 
  // Both blocks and threads are used to parallelize the computation. 
  // Threads within the same block can communicate and synchronize more easily than threads in different blocks.
  squareKernel<<<1, N>>>(gpuMem, N);

  ///////////////////////////

  // Lets return the answer back to the host or CPU
  cudaMemcpy(mem, gpuMem, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    printf("%d ", mem[i]);
  }
  printf("\n");

  cudaFree(gpuMem);
  free(mem);
}

// We basically allocated memory in cpu, allocated memory in gpu (using a pointer of a pointer), copied the cpu memory to gpu, performed a kernel execution on this memory space, copied back to cpu memory, and printed the results. Threads are lightweight execution instances responsible for single execution path, and blocks are groups of threads that allow more synchronization and coordination, leading to less overhead. 
