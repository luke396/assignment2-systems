#### (d) Residual Stream Activation Size (Single Precision)

For fixed batch_size = 4, d_model = 2560, FP32 = 4 bytes

| Seq Length | Calculation                | Size (MB)   |
| ---------- | -------------------------- | ----------- |
| 128        | 4 × 128 × 2560 × 4 / 1024² | **5.0 MB**  |
| 256        | 4 × 256 × 2560 × 4 / 1024² | **10.0 MB** |
| 512        | 4 × 512 × 2560 × 4 / 1024² | **20.0 MB** |

We can see that **activation size scales linearly with seq_length** — when seq doubles, activation size also doubles.
