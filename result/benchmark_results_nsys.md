# Benchmark Results

| Config | Seq Len | Warmup Steps | Status FWD+BWD | Total(s) FWD+BWD | Warmup(s) FWD+BWD | Avg/Step(s) FWD+BWD | Std/Step(s) FWD+BWD |
| :----- | ------: | -----------: | :------------- | ---------------: | ----------------: | ------------------: | ------------------: |
| small | 128 | 5 | OK | 1.6827 | 0.9382 | 0.0745 | 0.00369 |
| medium | 128 | 5 | OK | 2.3811 | 1.1835 | 0.1198 | 0.012624 |
| large | 128 | 5 | OK | 3.5577 | 1.678 | 0.188 | 0.009113 |
| xl | 128 | 5 | OOM | nan | nan | nan | nan |
| 2.7B | 128 | 5 | OOM | nan | nan | nan | nan |
