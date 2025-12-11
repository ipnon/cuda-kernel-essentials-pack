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

// Helper function to run complete reduction and return result in in_ptr
void run_reduction(float* d_input, float* d_partial, float* d_output, int n,
                   int threadsPerBlock, float** result_ptr) {
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // First pass
  reduce_sum<<<numBlocks, threadsPerBlock>>>(d_input, d_partial, n);

  // Iterative passes
  int remaining = numBlocks;
  float* in_ptr = d_partial;
  float* out_ptr = d_output;
  while (1 < remaining) {
    int blocks = (remaining + threadsPerBlock - 1) / threadsPerBlock;
    reduce_sum<<<blocks, threadsPerBlock>>>(in_ptr, out_ptr, remaining);
    remaining = blocks;

    float* temp = in_ptr;
    in_ptr = out_ptr;
    out_ptr = temp;
  }

  *result_ptr = in_ptr;  // Give caller address of result
}

int main() {
  int n = 1'000'000;

  // Allocate host memory
  float *h_input, h_output;
  h_input = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) h_input[i] = 1.0f;

  // Allocate device memory
  float *d_input, *d_partial, *d_output;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_partial, n * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));
  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  // Verify
  float cpu_sum = 0.0f;
  for (int i = 0; i < n; i++) cpu_sum += h_input[i];
  int threadsPerBlock = 256;
  float* result_ptr;
  run_reduction(d_input, d_partial, d_output, n, threadsPerBlock, &result_ptr);
  cudaMemcpy(&h_output, result_ptr, sizeof(float), cudaMemcpyDeviceToHost);
  printf("CPU sum: %f, GPU sum: %f\n", cpu_sum, h_output);

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int i = 0; i < 10; i++) {
    run_reduction(d_input, d_partial, d_output, n, threadsPerBlock,
                  &result_ptr);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    run_reduction(d_input, d_partial, d_output, n, threadsPerBlock,
                  &result_ptr);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Average time: %.4f ms\n", ms / 100);

  // Free memory
  free(h_input);
  cudaFree(d_input);
  cudaFree(d_partial);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
