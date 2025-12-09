#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduce_sum(float* input, float* output, int n) {
    // Shared memory for this block
  __shared__ float tile[256];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load from global to shared
  tile[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; 0 < stride; stride /= 2) {
    if (tid < stride) {
      tile[tid] += tile[tid + stride];
    }

    // Wait for all threads before next iteration
    __syncthreads();
  }

  // Thread 0 writes block's result
  if (tid == 0) {
    output[blockIdx.x] = tile[0];
  }
}

// cudaMemCpy moves data across PCIe bus.
int main() {
  int n = 1'000'000;

  // Allocate host memory
  float *h_input;
  float *d_input, *d_partial, *d_output;
  h_input = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) h_input[i] = 1.0f;

  // Allocate device memory
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_partial, n * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));

  // Copy H->D
  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  // Execution configuration
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // First kernel: N elements => numBlocks partial sums
  reduce_sum<<<numBlocks, threadsPerBlock>>>(d_input, d_partial, n);

  // Second kernel: numBlocks partial sums -> 1 final sum
  reduce_sum<<<1, threadsPerBlock>>>(d_partial, d_output, numBlocks);

  // Copy D->H
  float h_output;
  cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

  // Verify results
  float cpu_sum = 0.0f;
  for (int i = 0; i < n; i++) cpu_sum += h_input[i];
  printf("CPU sum: %f, GPU sum: %f\n", cpu_sum, h_output);

  // Free memory
  free(h_input);
  cudaFree(d_input);
  cudaFree(d_partial);
  cudaFree(d_output);

  return 0;
}
