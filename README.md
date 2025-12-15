# Essential CUDA kernels

Five fundamental CUDA kernels implementing core GPU computing primitives, benchmarked against PyTorch on an NVIDIA A10 GPU.

## Kernels Implemented

1. **Vector Add** - Elementwise addition of two arrays
2. **ReLU** - Elementwise activation function using `fmaxf(0, x)`
3. **Parallel Reduction** - Multi-pass sum reduction with shared memory
4. **Prefix Sum** - Blelloch scan (exclusive) with up-sweep/down-sweep
5. **Naive Matmul** - Matrix multiplication without tiling

## Performance Results

### Runtime Comparison (vs PyTorch)

| Kernel | Size | Ours | PyTorch | Ratio |
|--------|------|------|---------|-------|
| Vector Add | 1M elements | 0.0240 ms | 0.0143 ms | 1.68x slower |
| ReLU | 1M elements | 0.0146 ms | 0.0144 ms | 1.01x (match) |
| Reduction | 1M elements | 0.0226 ms | 0.0174 ms | 1.30x slower |
| Prefix Sum | 256 elements | 0.0045 ms | 0.0195 ms | **4.3x faster** |
| Matmul | 1024x1024 | 1.2199 ms | 0.1372 ms | 8.9x slower |

### Nsight Compute Profiling

| Kernel | Grid Size | Occupancy | Memory Throughput |
|--------|-----------|-----------|-------------------|
| Vector Add | 3907 blocks | 79.9% | 470.4 GB/s |
| ReLU | 3907 blocks | 69.7% | 375.6 GB/s |
| Reduction (pass 1) | 3907 blocks | 87.5% | 134.6 GB/s |
| Prefix Sum | 1 block | 16.6% | 0.66 GB/s |
| Matmul | 64x64 blocks | 95.7% | 8.7 GB/s |

**A10 Peak Memory Bandwidth: ~600 GB/s**

## Analysis

### Memory-Bound Kernels (Vector Add, ReLU)
These achieve 60-80% of peak memory bandwidth. Performance is limited by memory throughput, not compute. Our naive implementations match PyTorch within 1-2x.

### Reduction
Multi-pass reduction achieves good occupancy (87.5%) on the first pass but drops significantly on later passes as work decreases. The iterative approach has kernel launch overhead vs a single optimized kernel.

### Prefix Sum (Blelloch Scan)
Single-block implementation beats PyTorch's general-purpose `cumsum`. Low occupancy (16.6%) because only 1 block runs, but the kernel is fast enough that PyTorch's dispatch overhead dominates for small arrays.

### Naive Matmul
High occupancy (95.7%) but terrible memory throughput (8.7 GB/s vs 600 GB/s peak). The strided access pattern to matrix B causes poor memory coalescing:

```cuda
B[k * N + col]  // Adjacent threads read N elements apart
```

This is why tiled matmul with shared memory is essential for production use.

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Running Benchmarks

```bash
# CUDA kernels
./vector_add
./elementwise
./reduction
./prefix_sum
./matmul

# PyTorch comparison
python3 ../benchmarks/pytorch_comparison.py
```

## Profiling

```bash
sudo ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second ./kernel_name
```

## Hardware

- **GPU**: NVIDIA A10 (Ampere, SM 8.6)
- **Memory**: 24 GB GDDR6, ~600 GB/s bandwidth
- **SMs**: 72
- **CUDA Cores**: 9216

## Key Learnings

1. **Memory-bound vs compute-bound**: Simple elementwise ops are memory-bound; naive implementations are nearly optimal.

2. **Memory coalescing matters**: Matmul's 9x slowdown comes from strided memory access, not lack of compute.

3. **Occupancy isn't everything**: Prefix sum has 16% occupancy but beats PyTorch. Small kernels can win on low overhead.

4. **Multi-pass overhead**: Reduction's iterative approach has diminishing occupancy and kernel launch costs.

## Next Steps

- Implement tiled matmul with shared memory (target: 50% of cuBLAS)
- Add Tensor Core support (WMMA)
- Implement multi-block prefix sum
