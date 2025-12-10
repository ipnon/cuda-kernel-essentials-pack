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

  // Reduce n elements -> numBlocks partial sums
  // Each block reduces 256 elements to 1 partial sum
  reduce_sum<<<numBlocks, threadsPerBlock>>>(d_input, d_partial, n);

  // Iteratively reduce until 1 element remains
  // Why do we need to iterate?
  // threadsPerBlock < numBlocks (numBlocks is 3907 when n is 1,000,000)
  // One kernel launch can only reduce by a factor of 256.
  int remaining = numBlocks;
  float *in_ptr = d_partial;
  float *out_ptr = d_output;

  while (1 < remaining) {
    int blocks = (remaining + threadsPerBlock - 1) / threadsPerBlock;
    reduce_sum<<<blocks, threadsPerBlock>>>(in_ptr, out_ptr, remaining);
    remaining = blocks;

    // Swap so that the output becomes the input for the next iteration.
    float *temp = in_ptr;
    in_ptr = out_ptr;
    out_ptr = temp;
  }

  // The result is in in_ptr, not d_output, due to the swapping.
  float h_output;
  cudaMemcpy(&h_output, in_ptr, sizeof(float), cudaMemcpyDeviceToHost);

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
