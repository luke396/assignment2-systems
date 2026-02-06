### End-to-End JIT vs Eager Comparison Results

#### JIT vs Eager: xl

| Seq Len | Mode  | Avg Time (s) | Mem (GB) |
| ------- | ----- | ------------ | -------- |
| 128     | Eager | 0.5456       | 34.26    |
| 128     | JIT   | 0.5079       | 32.50    |
| 256     | Eager | 0.9759       | 41.71    |
| 256     | JIT   | 0.9120       | 37.49    |
| 512     | Eager | 1.8508       | 60.34    |
| 512     | JIT   | 1.6537       | 49.86    |
| 1024    | Eager | N/A          | N/A      |
| 1024    | JIT   | N/A          | N/A      |

#### JIT vs Eager: 2.7B

| Seq Len | Mode  | Avg Time (s) | Mem (GB) |
| ------- | ----- | ------------ | -------- |
| 128     | Eager | 0.8397       | 53.87    |
| 128     | JIT   | 0.8099       | 52.16    |
| 256     | Eager | 1.4698       | 60.86    |
| 256     | JIT   | 1.4043       | 56.75    |
