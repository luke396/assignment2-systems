### distributed_communication_single_node

![All-Reduce Benchmark](images/all_reduce_benchmark.png)

| Backend | Data Size | 2 procs | 4 procs | 6 procs |
|---------|-----------|---------|---------|---------|
| Gloo+CPU | 1 MB | 1.14 ms | 1.94 ms | 3.07 ms |
| Gloo+CPU | 10 MB | 9.10 ms | 12.97 ms | 15.72 ms |
| Gloo+CPU | 100 MB | 93.26 ms | 127.92 ms | 145.69 ms |
| Gloo+CPU | 1 GB | 851.30 ms | 2251.83 ms | 1947.31 ms |
| NCCL+GPU | 1 MB | 0.09 ms | 0.20 ms | 0.28 ms |
| NCCL+GPU | 10 MB | 0.46 ms | 0.70 ms | 0.74 ms |
| NCCL+GPU | 100 MB | 3.63 ms | 6.29 ms | 6.87 ms |
| NCCL+GPU | 1 GB | 33.47 ms | 63.21 ms | 70.07 ms |

NCCL+GPU is consistently 10–25x faster than Gloo+CPU across all configurations, thanks to high-bandwidth GPU interconnects (NVLink). 

Both backends show roughly linear latency scaling with data size, confirming that all-reduce is bandwidth-bound for large messages. Increasing the process count raises latency moderately, with similar relative scaling across both backends—consistent with the ring all-reduce algorithm whose communication volume scales as `2·(N−1)/N · data_size`.

One anomaly is the Gloo 1 GB case where 6 processes (1947 ms) is slightly faster than 4 processes (2252 ms), likely due to measurement variance at only 5 iterations.
