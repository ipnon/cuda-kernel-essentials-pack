#include <cuda_runtime.h>
#include <stdio.h>

__global__ void relu(float* a, float* b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    b[idx] = fmaxf(0.0f, a[idx]);
  }
}

// cudaMemCpy moves data across PCIe bus.
int main() {
  int n = 1'000'000;

  // Allocate host memory
  float *h_a, *h_b;
  float *d_a, *d_b;
  h_a = (float*)malloc(n * sizeof(float));
  h_b = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    h_a[i] = (i % 2 == 0) ? i : -i;
  }

  // Allocate device memory
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));

  // Copy H->D
  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  relu<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);

  // Copy D->H
  cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify results
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float expected = fmaxf(0.0f, h_a[i]);  // Same as ReLU
    if (h_b[i] != expected) errors++;
  }
  printf("Errors: %d\n", errors);

  // Free memory
  free(h_a);
  free(h_b);
  cudaFree(d_a);
  cudaFree(d_b);

  return 0;
}
