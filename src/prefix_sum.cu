#include <cuda_runtime.h>
#include <stdio.h>

__global__ void prefix_sum(float* input, float* output, int n) {
  __shared__ float tile[256];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load from global to shared
  tile[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Up-sweep
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid + 1) % (stride * 2) == 0) {
      tile[tid] += tile[tid - stride];
    }
    __syncthreads();
  }

  // Down-sweep
  tile[blockDim.x - 1] =
      0;  // The sum of everything before the first element is 0.
  __syncthreads();
  for (int stride = blockDim.x / 2; 0 < stride; stride /= 2) {
    if ((tid + 1) % (stride * 2) == 0) {
      float temp = tile[tid - stride];  // Save left
      tile[tid - stride] = tile[tid];   // Push down
      tile[tid] += temp;                // Accumulate
    }
    __syncthreads();
  }

  // Write result to global memory
  if (idx < n) {
    output[idx] = tile[tid];
  }
}

int main() {
  int n = 256;

  float *h_input, *h_output;
  float *d_input, *d_output;

  h_input = (float*)malloc(n * sizeof(float));
  h_output = (float*)malloc(n * sizeof(float));

  // Initialize with natural numbers;
  for (int i = 0; i < n; i++) h_input[i] = i + 1;

  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));
  cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

  prefix_sum<<<1, 256>>>(d_input, d_output, n);

  cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Exclusive prefix sum: each element is the sum of all elements before it.
  printf("First 10: ");
  for (int i = 0; i < 10; i++) printf("%.0f, ", h_output[i]);
  printf("\nExpected: 0, 1, 3, 6, 10, 15, 21, 28, 36, 45\n");

  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
