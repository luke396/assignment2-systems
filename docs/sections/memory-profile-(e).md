#### (e) Memory Allocation Analysis (Single-Step Peak)

##### FWD seq128 - Max Single Allocation

| Category | Baseline | Autocast | Δ (MiB) |
| --- | --- | --- | --- |
| AttnScores | 8.0 MiB | 4.0 MiB | -4.0 |
| AttnOut | 8.0 MiB | 4.0 MiB | -4.0 |
| Softmax | 8.0 MiB | 8.0 MiB | 0 |
| FFN | 20.0 MiB | 10.0 MiB | -10.0 |
| Norm | 5.0 MiB | 5.0 MiB | 0 |
| Embed | 5.0 MiB | 5.0 MiB | 0 |
| Attn | 5.0 MiB | 2.5 MiB | -2.5 |
| einsum | 20.0 MiB | 50.0 MiB | +30.0 |
| Block | 5.0 MiB | 5.0 MiB | 0 |

##### FWD seq256 - Max Single Allocation

| Category | Baseline | Autocast | Δ (MiB) |
| --- | --- | --- | --- |
| AttnScores | 32.0 MiB | 16.0 MiB | -16.0 |
| AttnOut | 32.0 MiB | 16.0 MiB | -16.0 |
| Softmax | 32.0 MiB | 32.0 MiB | 0 |
| FFN | 40.0 MiB | 20.0 MiB | -20.0 |
| Norm | 10.0 MiB | 10.0 MiB | 0 |
| Embed | 10.0 MiB | 10.0 MiB | 0 |
| Attn | 10.0 MiB | 5.0 MiB | -5.0 |
| einsum | 40.0 MiB | 50.0 MiB | +10.0 |
| Block | 10.0 MiB | 10.0 MiB | 0 |

##### FWD seq512 - Max Single Allocation

| Category | Baseline | Autocast | Δ (MiB) |
| --- | --- | --- | --- |
| AttnScores | 128.0 MiB | 64.0 MiB | -64.0 |
| AttnOut | 128.0 MiB | 64.0 MiB | -64.0 |
| Softmax | 128.0 MiB | 128.0 MiB | 0 |
| FFN | 80.0 MiB | 40.0 MiB | -40.0 |
| Norm | 20.0 MiB | 20.0 MiB | 0 |
| Embed | 20.0 MiB | 20.0 MiB | 0 |
| Attn | 20.0 MiB | 10.0 MiB | -10.0 |
| einsum | 128.0 MiB | 64.0 MiB | -64.0 |
| Block | 20.0 MiB | 20.0 MiB | 0 |

##### FWD_BWD seq128 - Max Single Allocation

| Category | Baseline | Autocast | Δ (MiB) |
| --- | --- | --- | --- |
| Gradient | 100.0 MiB | 100.0 MiB | 0 |
| Opt State | 100.0 MiB | 100.0 MiB | 0 |
| AttnScores | 8.0 MiB | 4.0 MiB | -4.0 |
| AttnOut | 8.0 MiB | 4.0 MiB | -4.0 |
| Softmax | 8.0 MiB | 8.0 MiB | 0 |
| FFN | 20.0 MiB | 10.0 MiB | -10.0 |
| Norm | 5.0 MiB | 5.0 MiB | 0 |
| Embed | 5.0 MiB | 5.0 MiB | 0 |
| Attn | 5.0 MiB | 2.5 MiB | -2.5 |
| einsum | 20.0 MiB | 50.0 MiB | +30.0 |
| Block | 5.0 MiB | 5.0 MiB | 0 |

##### FWD_BWD seq256 - Max Single Allocation

| Category | Baseline | Autocast | Δ (MiB) |
| --- | --- | --- | --- |
| Gradient | 100.0 MiB | 100.0 MiB | 0 |
| Opt State | 100.0 MiB | 100.0 MiB | 0 |
| AttnScores | 32.0 MiB | 16.0 MiB | -16.0 |
| AttnOut | 32.0 MiB | 16.0 MiB | -16.0 |
| Softmax | 32.0 MiB | 32.0 MiB | 0 |
| FFN | 40.0 MiB | 20.0 MiB | -20.0 |
| Norm | 10.0 MiB | 10.0 MiB | 0 |
| Embed | 10.0 MiB | 10.0 MiB | 0 |
| Attn | 10.0 MiB | 5.0 MiB | -5.0 |
| einsum | 40.0 MiB | 50.0 MiB | +10.0 |
| Block | 10.0 MiB | 10.0 MiB | 0 |

##### Top 10 Allocations: baseline FWD_BWD seq256

| # | Size | Category | Operation | Call Chain |
| --- | --- | --- | --- | --- |
| 1 | 100.0 MiB | Other | unknown | Internal |
| 2 | 100.0 MiB | Gradient | clone_for_grad | Internal |
| 3 | 100.0 MiB | Other | unknown | Internal |
| 4 | 100.0 MiB | Gradient | clone_for_grad | Internal |
| 5 | 100.0 MiB | Other | unknown | Internal |
| 6 | 100.0 MiB | Gradient | clone_for_grad | Internal |
| 7 | 100.0 MiB | Other | unknown | Internal |
| 8 | 100.0 MiB | Gradient | clone_for_grad | Internal |
| 9 | 100.0 MiB | Other | unknown | Internal |
| 10 | 100.0 MiB | Gradient | clone_for_grad | Internal |

##### Key Findings

1. **Attention scores scale quadratically**: The Q·K^T computation
   (AttnScores) creates large activation allocations that scale with O(seq^2):
   - seq128: 8 MiB baseline (batch x heads x 128 x 128 x 4 bytes)
   - seq256: 32 MiB baseline (4x increase for 2x seq)
   - seq512: 128 MiB baseline (4x increase for 2x seq)

2. **Gradient and optimizer state are fixed-size**: The largest single
   allocations (~100 MiB each) come from:
   - Gradient accumulation (AccumulateGrad) for embedding/projection layers
   - Optimizer state initialization (zeros_like) for Adam's m and v buffers
   - These scale with model parameters, not sequence length

3. **FFN allocations scale linearly**: Feed-forward network activations
   scale with O(batch x seq x d_ff):
   - seq128: 20 MiB, seq256: 40 MiB, seq512: 80 MiB

4. **Autocast reduces most activation sizes by ~50%**: With bf16:
   - AttnScores, AttnOut, FFN, Attn all halve in size
   - Softmax, Norm, Embed, Block unchanged (forced fp32 or unaffected)

5. **einsum increases under autocast**: The einsum category shows +30 MiB
   increase at seq128, likely due to dtype conversion buffers created
   during bf16<->fp32 casts in the custom einsum implementations.

6. **Softmax stays fp32**: Custom softmax implementation forces fp32 for
   numerical stability, so its allocation size doesn't change with autocast.
