"""Naive distributed data parallel training implementation."""

import os
import time
from copy import deepcopy
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.blocks import TransformerLM
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_systems.benchmark import generate_random_data


def _setup(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _data(
    global_batchsize: int,
    shape: tuple[int, ...] = (3, 64),
    seed: int = 42,
) -> torch.Tensor:
    # ignore device, move data to rank_i or cpu in ddp training loop
    # generate in cpu to keep multi device stable,
    # move to gpu in ddp training loop when necessary
    return torch.randn(
        global_batchsize,
        *shape,
        generator=torch.Generator().manual_seed(seed),
    )


def _rank_parallel_train(
    rank: int,
    world_size: int,
    backend: str,
    global_batchsize: int,
    iters: int,
    *,
    flat_grad: bool = False,
) -> None:
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    _setup(rank, world_size, backend)
    microbatch_size = global_batchsize // world_size

    seed = 42
    global_data = _data(global_batchsize=global_batchsize, seed=seed).to(device)
    micro_data = global_data[rank * microbatch_size : (rank + 1) * microbatch_size]

    dim = micro_data.shape[-1]
    base_model = torch.nn.Linear(dim, dim).to(device)

    # different model init for rank's model,
    # using broadcast to sync rank0's in actul training
    torch.manual_seed(seed + rank)
    ddp_model = deepcopy(base_model)
    for param in ddp_model.parameters():
        dist.broadcast(param.data, src=0)

    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=0.01)
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for _ in range(iters):
        base_loss = base_model(global_data).mean()
        base_loss.backward()
        base_optimizer.step()
        base_optimizer.zero_grad()

        ddp_loss = ddp_model(micro_data).mean()
        ddp_loss.backward()

        # loss mean + all reduce mean equal to global batch size mean
        if flat_grad:
            flat = _flatten_dense_tensors([p.grad for p in ddp_model.parameters()])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for p, g in zip(
                ddp_model.parameters(),
                _unflatten_dense_tensors(
                    flat, [p.grad for p in ddp_model.parameters()]
                ),
                strict=True,
            ):
                p.grad = g
        else:
            for param in ddp_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        ddp_optimizer.step()
        ddp_optimizer.zero_grad()

        if rank == 0:
            for param_base, param_ddp in zip(
                base_model.parameters(), ddp_model.parameters(), strict=True
            ):
                # the two models are close enough after training
                assert torch.allclose(param_base.data, param_ddp.data, atol=1e-6)

    dist.destroy_process_group()


def naive_ddp(batch_size: int = 32, n_procs: int = 8, backend: str = "gloo") -> None:
    """Verify naive DDP training matches single-process training."""
    assert batch_size % n_procs == 0
    iters = 10
    mp.spawn(
        _rank_parallel_train,
        args=(n_procs, backend, batch_size, iters),
        nprocs=n_procs,
        join=True,
    )


def _ddp_xl(
    rank: int,
    world_size: int,
    queue: mp.Queue,
    seq_len: int = 64,
    backend: str = "nccl",
    *,
    flat_grad: bool = False,
    d_model: int = 1600,
    d_ff: int = 6400,
    n_heads: int = 25,
    n_layers: int = 48,
) -> None:
    _setup(rank, world_size, backend=backend)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    vocab_size = 10000
    batch_size = 4
    microbatch_size = batch_size // world_size
    seed = 42
    warmup_iters = 5
    iters = 10

    data = generate_random_data(
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )[rank * microbatch_size : (rank + 1) * microbatch_size]

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        context_length=seq_len,
        n_layers=n_layers,
        device=device,
    )
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _sync_grads() -> None:
        if flat_grad:
            grads = [p.grad for p in model.parameters()]
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for p, g in zip(
                model.parameters(),
                _unflatten_dense_tensors(flat, grads),
                strict=True,
            ):
                p.grad = g
        else:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    for _i in range(warmup_iters):
        logits = model(data)
        loss = loss_fn(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()
        _sync_grads()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        logits = model(data)
        loss = loss_fn(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()

        torch.cuda.synchronize()
        training_time = time.perf_counter()

        _sync_grads()

        torch.cuda.synchronize()
        allreduce_time = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        optimizer_time = time.perf_counter()
        if rank == 0:
            queue.put(
                {
                    "seq_len": seq_len,
                    "iter": i,
                    "training_time": training_time - start,
                    "allreduce_time": allreduce_time - training_time,
                    "optimizer_time": optimizer_time - allreduce_time,
                }
            )
        torch.cuda.synchronize()
        dist.barrier()
    dist.destroy_process_group()


def benchmark_ddp_xl(
    n_procs: int = 2,
    backend: str = "nccl",
    seq_len: int = 64,
    *,
    flat_grad: bool = False,
) -> list[dict]:
    """Benchmark DDP training on xl model."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    mp.spawn(
        partial(_ddp_xl, flat_grad=flat_grad),
        args=(n_procs, queue, seq_len, backend),
        nprocs=n_procs,
        join=True,
    )
    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda x: x["iter"])
    return results


def _format_results(
    all_results: list[dict],
    n_procs: int,
    label: str = "Naive",
    oom_seq_lens: list[int] | None = None,
) -> str:
    """Format benchmark results as markdown table."""
    seq_lens = sorted({r["seq_len"] for r in all_results})
    rows = []
    for sl in seq_lens:
        group = [r for r in all_results if r["seq_len"] == sl]
        n = len(group)
        avg_t = sum(r["training_time"] for r in group) / n * 1000
        avg_c = sum(r["allreduce_time"] for r in group) / n * 1000
        avg_o = sum(r["optimizer_time"] for r in group) / n * 1000
        avg_total = avg_t + avg_c + avg_o
        pct = avg_c / avg_total * 100
        rows.append(
            f"| {sl} | {avg_t:.2f} | {avg_c:.2f} "
            f"| {avg_o:.2f} | {avg_total:.2f} | {pct:.1f}% |"
        )
    rows.extend(
        f"| {sl} | OOM | OOM | OOM | OOM | OOM |" for sl in sorted(oom_seq_lens or [])
    )
    table = "\n".join(rows)

    return (
        f"### {label} DDP Benchmark (XL, 1 Node x {n_procs} GPUs)\n\n"
        "| Seq Len | Fwd+Bwd (ms) | AllReduce (ms) "
        "| Optimizer (ms) | Total (ms) | Comm % |\n"
        "|---------|-------------|----------------|"
        "----------------|------------|--------|\n"
        f"{table}\n"
    )


class DDP(torch.nn.Module):
    """Distributed data parallel wrapper with overlapped gradient communication."""

    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        """Broadcast params from rank 0 and register async all-reduce hooks."""
        super().__init__()
        self.module = module
        self.handles: list[dist.Work] = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param: torch.Tensor) -> None:
        # grad has filled for cur layer, call all-reduce instantly
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        assert handle is not None
        self.handles.append(handle)

    def forward(self, *inputs: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped module."""
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for all async all-reduce operations to complete."""
        for h in self.handles:
            h.wait()
        self.handles.clear()


if __name__ == "__main__":
    from pathlib import Path

    naive_ddp(batch_size=32, n_procs=8, backend="gloo")

    n = 2
    seq_lens = (64, 96, 128, 256, 512, 1024)
    out_dir = Path("docs/sections")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes: list[tuple[bool, str, str]] = [
        (False, "Naive", "ddp-xl-naive.md"),
        (True, "Flat-Grad-Naive", "ddp-xl-naive-flat-grad.md"),
    ]

    for use_flat, label, filename in modes:
        all_res: list[dict] = []
        oom_seqs: list[int] = []
        for seq in seq_lens:
            print(f"[{label}] Benchmarking seq_len={seq}...")
            try:
                res = benchmark_ddp_xl(
                    n_procs=n, backend="nccl", seq_len=seq, flat_grad=use_flat
                )
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(f"[{label}] OOM at seq_len={seq}, skipping rest.")
                    oom_seqs.extend(
                        s for s in seq_lens if s >= seq and s not in oom_seqs
                    )
                    break
                raise
            all_res.extend(res)

        if not all_res and not oom_seqs:
            print(f"[{label}] No results collected, skipping output.")
            continue
        md = _format_results(all_res, n_procs=n, label=label, oom_seq_lens=oom_seqs)
        print(md)
        out = out_dir / filename
        out.write_text(md)
        print(f"Saved: {out}")
