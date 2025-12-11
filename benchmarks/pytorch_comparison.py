import torch
import torch.nn.functional as F

def benchmark(fn, warmup=10, iterations=100):
  """Returns average time in milliseconds."""

  # Warmup runs to be discarded
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()

  # Timed runs
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  start.record()
  for _ in range(iterations):
    fn()
  end.record()

  torch.cuda.synchronize()
  return start.elapsed_time(end) / iterations

def bench_vector_add():
  n = 1_000_000
  a = b = torch.ones(n, device='cuda')
  def fn():
    c = a + b
  ms = benchmark(fn)
  print(f"Vector add: {ms:.4f} ms")

def bench_elementwise():
  n = 1_000_000
  a = torch.randn(n, device='cuda')
  def fn():
    b = F.relu(a)
  ms = benchmark(fn)
  print(f"Elementwise: {ms:.4f} ms")

def bench_reduction():
  n = 1_000_000
  a = torch.ones(n, device='cuda')
  def fn():
    b = a.sum()
  ms = benchmark(fn)
  print(f"Reduction: {ms:.4f} ms")

def bench_prefix_sum():
  n = 256
  a = torch.ones(n, device='cuda')
  def fn():
    b = torch.cumsum(a, dim=0)
  ms = benchmark(fn)
  print(f"Prefix sum: {ms:.4f} ms")

def bench_matmul():

  # Too small a matrix size leads to efficient caching
  n = 1024
  a = b = torch.ones(n, n, device='cuda')
  def fn():
    c = torch.matmul(a, b)
  ms = benchmark(fn)
  print(f"Matmul: {ms:.4f} ms")

if __name__ == "__main__":
  print(f"Device: {torch.cuda.get_device_name()}")
  print("-" * 40)
  bench_vector_add()
  bench_elementwise()
  bench_reduction()
  bench_prefix_sum()
  bench_matmul()
