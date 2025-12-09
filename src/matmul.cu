#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul(float* A, float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main() {
  int n = 256;

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  h_a = (float*)malloc(n * n * sizeof(float));
  h_b = (float*)malloc(n * n * sizeof(float));
  h_c = (float*)malloc(n * n * sizeof(float));

  for (int i = 0; i < n * n; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 1.0f;
  }

  cudaMalloc(&d_a, n * n * sizeof(float));
  cudaMalloc(&d_b, n * n * sizeof(float));
  cudaMalloc(&d_c, n * n * sizeof(float));
  cudaMemcpy(d_a, h_a, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);                  // 16x16 = 256 threads
  dim3 numBlocks((n + 15) / 16, (n + 15) / 16);  // Enough blocks to cover NxN
  matmul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify
  int errors = 0;
  for (int i = 0; i < n * n; i++)
    if (h_c[i] != n) errors++;
  printf("Errors: %d\n", errors);

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
