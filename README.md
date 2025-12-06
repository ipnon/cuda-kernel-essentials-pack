### Month 2: CUDA Fundamentals

#### Objectives

- Learn GPU execution model: blocks, threads, warps.
- Learn memory types: global, shared, registers.

#### Readings

- CUDA Programming Guide (intro sections).
- CUDA by Example.
- Nsight Compute Quickstart.

#### Project: CUDA Kernel Essentials Pack

##### Requirements

Implement the following as separate kernels:

1. Vector Add
2. Elementwise Unary Function (e.g., sigmoid or ReLU)
3. Parallel Reduction (sum + max)
4. Prefix Sum (Blelloch scan)
5. Naive Matmul (no tiling)

Benchmarks:
- Compare performance against PyTorch equivalents.
- Use Nsight Compute to collect runtime, occupancy, and memory throughput.

##### Why This Matters

- These five kernels cover the primitives most ML ops are built from.
- This project is a portfolio piece showing GPU fluency.
