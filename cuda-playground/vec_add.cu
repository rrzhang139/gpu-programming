#include <stdio.h>
#include <stdlib.h>

__global__ void vectorAddition(float *v1, float *v2, float *o, int length) {
  int i = threadIdx.x;
  if (i < length) {
    o[i] = v1[i] + v2[i];
  }
}

int main() {
  int N = 10;
  float *v1 = (float*) malloc(N*sizeof(float));
  float *v2 = (float*) malloc(N*sizeof(float));
  float *o = (float*) malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    v1[i] = 1;
    v2[i] = 1;
  }  

  float* v1Gpu;
  cudaError_t err1 = cudaMalloc(&v1Gpu, N*sizeof(float));
  float* v2Gpu;
  cudaError_t err2 = cudaMalloc(&v2Gpu, N*sizeof(float));
  float* oGpu;
  cudaError_t err3 = cudaMalloc(&oGpu, N*sizeof(float));

  cudaMemcpy(v1Gpu, v1, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(v2Gpu, v2, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(oGpu, o, N*sizeof(float), cudaMemcpyHostToDevice);

  vectorAddition<<<1, N>>>(v1Gpu, v2Gpu, oGpu, N);

  cudaMemcpy(o, oGpu, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    printf("%f ", v1[i]);
  }
  printf("\n"); 
  for (int i = 0; i < N; i++) {
    printf("%f ", v2[i]);
  }
  printf("\n"); 
  for (int i = 0; i < N; i++) {
    printf("%f ", o[i]);
  }
  printf("\n"); 

  cudaFree(v1Gpu);
  cudaFree(v2Gpu);
  cudaFree(oGpu);
  free(v1);
  free(v2);
  free(o);
}