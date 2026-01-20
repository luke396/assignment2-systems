
#### Mixed Precision Analysis

##### Forward Pass Comparison (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 75.89 | 79.35 | 0.96x |
| medium | 94.93 | 103.34 | 0.92x |
| large | 115.40 | 123.42 | 0.94x |
| xl | 144.16 | 146.84 | 0.98x |
| 2.7B | 155.02 | 629.84 | 0.25x |


##### Forward+Backward Pass Comparison (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 85.71 | 105.35 | 0.81x |
| medium | 119.69 | 141.50 | 0.85x |
| large | 156.81 | 180.92 | 0.87x |


##### Attention Score Computation (Median Time in ms)

| Model | Full Precision | BF16 Mixed | Speedup |
| --- | --- | --- | --- |
| small | 0.31 | 0.23 | 1.35x |
| medium | 0.24 | 0.19 | 1.27x |
| large | 0.35 | 0.17 | 2.02x |
| xl | 0.56 | 0.18 | 3.07x |
| 2.7B | 1.68 | 0.50 | 3.34x |


##### Conclusions

1. BF16 mixed precision is slower (5-12%) on small-medium models due to autocast type conversion overhead (copy kernels dominate)
2. Attention score computation shows 1.2x-3.3x speedup with BF16 larger models benefit more from reduced precision matmul
3. As model size increases, BF16 disadvantage decreases because compute time dominates conversion overhead
