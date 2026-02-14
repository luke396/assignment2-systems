"""Naive distributed data parallel training implementation."""

import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.blocks import TransformerLM
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.cuda import nvtx
from torch.profiler import ProfilerActivity, schedule

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


def _make_sync_fn(
    model: torch.nn.Module,
    *,
    flat_grad: bool,
) -> Callable[[], None]:
    """Create a gradient sync barrier based on the DDP mode.

    Returns a callable that ensures all ranks have identical averaged gradients
    before the optimizer step, keeping model parameters in sync across ranks.
    """
    if isinstance(model, (DDP, DDPBucketed)):
        return model.finish_gradient_synchronization
    if flat_grad:

        def _sync() -> None:
            grads = [p.grad for p in model.parameters()]
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for p, g in zip(
                model.parameters(),
                _unflatten_dense_tensors(flat, grads),
                strict=True,
            ):
                p.grad = g

        return _sync

    def _sync_naive() -> None:
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    return _sync_naive


def _wrap_ddp(
    model: torch.nn.Module,
    *,
    overlapped: bool,
    bucket_size_mb: float | None,
) -> torch.nn.Module:
    """Wrap model with the appropriate DDP strategy."""
    if overlapped:
        return DDP(model)
    if bucket_size_mb is not None:
        return DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


def _make_profiler_ctx(
    profile_dir: str,
    rank: int,
    seq_len: int,
    _warmup_iters: int,
    iters: int,
) -> torch.profiler.profile:
    """Create a torch.profiler context for tracing CUDA/CPU activity."""
    out = Path(profile_dir)
    out.mkdir(parents=True, exist_ok=True)

    return torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=iters - 1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(out / f"rank{rank}_seq{seq_len}"),
        ),
        record_shapes=True,
        with_stack=False,
    )


def _run_measured_iters(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sync_grads: Callable[[], None],
    vocab_size: int,
    iters: int,
    queue: mp.Queue,
    rank: int,
    seq_len: int,
    prof: torch.profiler.profile | None,
) -> None:
    """Run measured training iterations, optionally stepping the profiler."""
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.range("forward"):
            logits = model(data)
            loss = loss_fn(
                logits.view(-1, vocab_size),
                data.view(-1),
            )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with nvtx.range("backward"):
            loss.backward()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with nvtx.range("grad_sync"):
            sync_grads()

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        with nvtx.range("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if rank == 0:
            queue.put(
                {
                    "seq_len": seq_len,
                    "iter": i,
                    "forward_time": t1 - t0,
                    "backward_time": t2 - t1,
                    "allreduce_time": t3 - t2,
                    "optimizer_time": t4 - t3,
                    "training_time": t2 - t0,
                }
            )
        if prof is not None:
            prof.step()
        torch.cuda.synchronize()
        dist.barrier()


def _ddp_xl(
    rank: int,
    world_size: int,
    queue: mp.Queue,
    seq_len: int = 64,
    backend: str = "nccl",
    *,
    flat_grad: bool = False,
    overlapped: bool = False,
    bucket_size_mb: float | None = None,
    profile_dir: str | None = None,
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

    with nvtx.range("model_param_init"):
        model = _wrap_ddp(
            model,
            overlapped=overlapped,
            bucket_size_mb=bucket_size_mb,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    sync_grads = _make_sync_fn(model, flat_grad=flat_grad)

    for _i in range(warmup_iters):
        logits = model(data)
        loss = loss_fn(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()
        sync_grads()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()

    prof_ctx = (
        _make_profiler_ctx(
            profile_dir,
            rank,
            seq_len,
            warmup_iters,
            iters,
        )
        if profile_dir and rank == 0
        else nullcontext()
    )

    with prof_ctx as prof:
        _run_measured_iters(
            model,
            data,
            loss_fn,
            optimizer,
            sync_grads,
            vocab_size,
            iters,
            queue,
            rank,
            seq_len,
            prof,
        )
    dist.destroy_process_group()


def benchmark_ddp_xl(
    n_procs: int = 2,
    backend: str = "nccl",
    seq_len: int = 64,
    *,
    flat_grad: bool = False,
    overlapped: bool = False,
    bucket_size_mb: float | None = None,
    profile_dir: str | None = None,
) -> list[dict]:
    """Benchmark DDP training on xl model."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    mp.spawn(
        partial(
            _ddp_xl,
            flat_grad=flat_grad,
            overlapped=overlapped,
            bucket_size_mb=bucket_size_mb,
            profile_dir=profile_dir,
        ),
        args=(n_procs, queue, seq_len, backend),
        nprocs=n_procs,
        join=True,
    )
    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda x: x["iter"])
    return results


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


@dataclass
class _Bucket:
    """Gradient bucket for batched all-reduce communication."""

    params: list[torch.nn.Parameter] = field(default_factory=list)
    num_ready: int = 0
    handle: dist.Work | None = None
    flat_buffer: torch.Tensor | None = None


class DDPBucketed(torch.nn.Module):
    """DDP wrapper with bucketed gradient communication."""

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float) -> None:
        """Broadcast params from rank 0 and set up gradient buckets."""
        super().__init__()
        self.module = module
        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        cur_bucket_size = 0
        cur_bucket = _Bucket()
        self._buckets: list[_Bucket] = []

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                cur_bucket_size += param.numel() * param.element_size()
                if cur_bucket.params and cur_bucket_size >= bucket_size_bytes:
                    self._buckets.append(cur_bucket)
                    cur_bucket = _Bucket()
                    cur_bucket_size = param.numel() * param.element_size()
                cur_bucket.params.append(param)
                param.register_post_accumulate_grad_hook(
                    partial(self._bucketed_grad_hook, bucket=cur_bucket)
                )
        if cur_bucket.params:
            self._buckets.append(cur_bucket)

    def _bucketed_grad_hook(self, _param: torch.Tensor, bucket: _Bucket) -> None:
        if bucket.num_ready == len(bucket.params) - 1:
            grads = [p.grad for p in bucket.params]
            flat = _flatten_dense_tensors(grads)
            bucket.handle = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=True)
            assert bucket.handle is not None
            bucket.flat_buffer = flat
        else:
            bucket.num_ready += 1

    def forward(self, *inputs: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped module."""
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for all bucket all-reduce ops and write averaged grads back."""
        for bucket in self._buckets:
            if bucket.handle is not None:
                bucket.handle.wait()
                assert bucket.flat_buffer is not None
                for p, g in zip(
                    bucket.params,
                    _unflatten_dense_tensors(
                        bucket.flat_buffer, [p.grad for p in bucket.params]
                    ),
                    strict=True,
                ):
                    p.grad = g
            bucket.handle = None
            bucket.flat_buffer = None
            bucket.num_ready = 0
