#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// cudaMemCpy moves data across PCIe bus.
int main() {
  int n = 1'000'000;

  // Allocate host memory
  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;
  h_a = (float*)malloc(n * sizeof(float));
  h_b = (float*)malloc(n * sizeof(float));
  h_c = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }
  
  // Allocate device memory
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));
  
  // Copy H->D
  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

  // Execution parameters
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  // Verify general correctness before starting benchmarking
  vector_add<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
  cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < n; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) errors++;
  }
  printf("Errors: %d\n", errors);

  // Benchmarking
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  // First kernel launches are slow due to:
  // CUDA context initialization: Driver setup, memory mapping
  // JIT compilation: PTX -> SASS for your specific GPU
  // GPU clock ramping: GPU boosts clocks under load, first runs may be at lower frequency
  // Cache warming: Instruction caches, TLB entries
  for (int i = 0; i < 10; i++) {
    vector_add<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
  }
  cudaDeviceSynchronize();

  // Timed runs
  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    vector_add<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Average time: %.4f ms\n", ms / 100);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(h_a); free(h_b); free(h_c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
