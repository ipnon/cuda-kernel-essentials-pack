#include <cuda_runtime.h>
#include <stdio.h>

__global__ void relu(float* a, float* b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    b[idx] = fmaxf(0.0f, a[idx]);
  }
}

int main() {
  int n = 1'000'000;

  float *h_a, *h_b;
  float *d_a, *d_b;
  h_a = (float*)malloc(n * sizeof(float));
  h_b = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    h_a[i] = (i % 2 == 0) ? i : -i;
  }
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

  // Verify kernel
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  relu<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);
  cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float expected = fmaxf(0.0f, h_a[i]);  // Same as ReLU
    if (h_b[i] != expected) errors++;
  }
  printf("Errors: %d\n", errors);

  // Benchmarking
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; i++) {
    relu<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < 100; i++) {
    relu<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Average time: %.4f ms\n", ms / 100);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Free memory
  free(h_a);
  free(h_b);
  cudaFree(d_a);
  cudaFree(d_b);

  return 0;
}
