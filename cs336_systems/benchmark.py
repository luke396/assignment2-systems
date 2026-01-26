"""Code for benchmarking_script."""

import argparse
import statistics
import timeit
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
from cs336_basics import blocks
from cs336_basics.blocks import TransformerLM
from cs336_basics.training_utility import AdamW, cross_entropy
from torch.cuda import nvtx

from cs336_systems.nvtx_model import annotated_scaled_dot_product_attention


@dataclass
class Config:
    """CLI arguments for benchmarking."""

    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    context_length: int = 512
    n_layers: int = 2
    steps: int = 15  # including warm-up
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warm_up_steps: int = 5
    forward_only: bool = False
    autocast: bool = False
    python_time: bool = False
    memory_profile: bool = False
    memory_output: str = "output/memory_snapshot.snap"


@dataclass(frozen=True)
class RunInputs:
    """Runtime inputs for a benchmark run."""

    vocab_size: int
    batch_size: int
    context_length: int
    steps: int
    warm_up_steps: int
    seed: int
    autocast: bool = False
    python_time: bool = False


@dataclass
class StepContext:
    """Context for running benchmark steps."""

    model: torch.nn.Module
    device: torch.device
    run_inputs: RunInputs
    optimizer: torch.optim.Optimizer | None
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None

    @property
    def is_training(self) -> bool:
        """Check if this is a training run (not forward-only)."""
        return self.optimizer is not None and self.loss_fn is not None

    @property
    def is_cuda(self) -> bool:
        """Check if running on CUDA device."""
        return self.device.type == "cuda"

    def nvtx_range(self, name: str) -> AbstractContextManager[None]:
        """Create an NVTX range context if running on CUDA, otherwise a no-op."""
        return nvtx.range(name) if self.is_cuda else nullcontext()


def parse_cli_args() -> Config:
    """Parse command line arguments for benchmarking script."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Benchmark TransformerLM")
    parser.add_argument("--d_model", type=int, default=defaults.d_model)
    parser.add_argument("--num_heads", type=int, default=defaults.num_heads)
    parser.add_argument("--d_ff", type=int, default=defaults.d_ff)
    parser.add_argument("--context_length", type=int, default=defaults.context_length)
    parser.add_argument("--n_layers", type=int, default=defaults.n_layers)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warm_up_steps", type=int, default=defaults.warm_up_steps)
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Enable autocasting for CUDA.",
    )
    parser.add_argument(
        "--python-time",
        action="store_true",
        help="Enable Python time mode with synchronize (generates md file).",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable memory profiling (writes snapshot file).",
    )
    parser.add_argument(
        "--memory-output",
        type=str,
        default=defaults.memory_output,
        help="Output filename for memory snapshot.",
    )
    return Config(**vars(parser.parse_args()))


def generate_random_data(
    seq_len: int, vocab_size: int, batch_size: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Generate random data for benchmarking."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
        device=device,
    )


def select_benchmark_device() -> torch.device:
    """Select a usable device, falling back to CPU if CUDA is unavailable/unusable."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    try:
        torch.empty(1, device="cuda")
    # CUDA initialization can raise a mix of runtime and driver errors.
    except (RuntimeError, OSError, AssertionError) as e:
        print(f"WARNING: CUDA initialization failed ({e}), falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda")


def _setup_training(
    model: torch.nn.Module, *, forward_only: bool, lr: float, weight_decay: float
) -> tuple[torch.optim.Optimizer | None, Callable | None]:
    """Create optimizer and loss function unless running forward-only."""
    if forward_only:
        return None, None
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay), cross_entropy


def _forward_pass(
    ctx: StepContext, data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute forward pass and optionally compute loss."""
    with ctx.nvtx_range("forward"):
        logits = ctx.model(data)

    if not ctx.is_training:
        return logits, None

    assert ctx.loss_fn is not None
    loss = ctx.loss_fn(logits.view(-1, ctx.run_inputs.vocab_size), data.view(-1))
    return logits, loss


def _backward_pass(ctx: StepContext, loss: torch.Tensor) -> None:
    """Execute backward pass and optimizer step."""
    assert ctx.optimizer is not None
    with ctx.nvtx_range("backward"):
        ctx.optimizer.zero_grad()
        loss.backward()
    with ctx.nvtx_range("optimizer_step"):
        ctx.optimizer.step()


def _run_single_step(ctx: StepContext, step: int) -> None:
    """Execute a single training step."""
    data = generate_random_data(
        seq_len=ctx.run_inputs.context_length,
        vocab_size=ctx.run_inputs.vocab_size,
        batch_size=ctx.run_inputs.batch_size,
        seed=ctx.run_inputs.seed + step,
        device=ctx.device,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if ctx.run_inputs.autocast
        else nullcontext()
    )
    # Autocast only forward and loss computation; backward uses same types automatically
    with autocast_ctx:
        _, loss = _forward_pass(ctx, data)
    if loss is not None:
        _backward_pass(ctx, loss)

    if ctx.is_cuda and ctx.run_inputs.python_time:
        torch.cuda.synchronize()


def _run_warmup_steps(ctx: StepContext) -> float:
    """Run warm-up steps and return total warm-up time."""
    start = timeit.default_timer()

    for step in range(ctx.run_inputs.warm_up_steps):
        _run_single_step(ctx, step)

    end = timeit.default_timer()
    return end - start


def _run_measurement_steps(ctx: StepContext) -> tuple[list[float], float]:
    """Run measurement steps and return step times and total time."""
    step_times: list[float] = []
    start = timeit.default_timer()

    start_step = ctx.run_inputs.warm_up_steps
    end_step = ctx.run_inputs.steps
    for step in range(start_step, end_step):
        step_start = timeit.default_timer()
        _run_single_step(ctx, step)
        step_times.append(timeit.default_timer() - step_start)

    return step_times, timeit.default_timer() - start


def run_benchmark() -> None:
    """Run the benchmark."""
    args = parse_cli_args()

    # Validate mutually exclusive modes
    if args.memory_profile and args.python_time:
        msg = "--memory-profile and --python-time are mutually exclusive"
        raise ValueError(msg)

    run_inputs = RunInputs(
        vocab_size=10000,  # fixed
        batch_size=4,  # fixed
        context_length=args.context_length,
        steps=args.steps,
        warm_up_steps=args.warm_up_steps,
        seed=args.seed,
        autocast=args.autocast,
        python_time=args.python_time,
    )
    device = select_benchmark_device()

    # Memory profiling requires CUDA
    if args.memory_profile and device.type != "cuda":
        msg = "--memory-profile requires CUDA but device is CPU"
        raise RuntimeError(msg)

    # NVTX mode: enable attention-level NVTX ranges
    # (default, not in python_time or memory_profile)
    if device.type == "cuda" and not args.python_time and not args.memory_profile:
        blocks.scaled_dot_product_attention = annotated_scaled_dot_product_attention  # type: ignore[assignment]

    model = TransformerLM(
        vocab_size=run_inputs.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        n_layers=args.n_layers,
    ).to(device)

    optimizer, loss_fn = _setup_training(
        model,
        forward_only=args.forward_only,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ctx = StepContext(
        model=model,
        device=device,
        run_inputs=run_inputs,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # Set model to training mode
    ctx.model.train()

    # Memory profiling: record history during measurement steps only
    # Using private PyTorch API - this is the official way to capture memory snapshots
    # as documented in PyTorch memory profiling guide
    memory_profiling_enabled = args.memory_profile and device.type == "cuda"
    if memory_profiling_enabled:
        torch.cuda.memory._record_memory_history(max_entries=1000000)  # type: ignore[attr-defined] # noqa: SLF001

    # Run warm-up phase
    warm_up_time = _run_warmup_steps(ctx)

    step_times, measurement_time = _run_measurement_steps(ctx)

    if memory_profiling_enabled:
        torch.cuda.memory._dump_snapshot(args.memory_output)  # type: ignore[attr-defined] # noqa: SLF001
        torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore[attr-defined] # noqa: SLF001
        print(f"Memory snapshot saved to: {args.memory_output}")

    # Calculate statistics for measurement steps
    avg_time = statistics.mean(step_times) if step_times else 0.0
    std_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
    total_time = warm_up_time + measurement_time

    if args.python_time:
        print(f"Total time for {args.steps} steps: {total_time:.4f} seconds")
        print(f"Warm-up time ({args.warm_up_steps} steps): {warm_up_time:.4f} seconds")
        print(f"Avg time per step after warm-up: {avg_time:.4f} seconds")
        print(f"Std time per step after warm-up: {std_time:.6f} seconds")
    elif args.memory_profile:
        print("Running in memory profiling mode (NVTX disabled)")
        print("Benchmark completed successfully")
    else:
        print("Running in NVTX profiling mode (synchronize disabled)")
        print("Use nsys report for accurate timing measurements")
        print("Benchmark completed successfully")


def main() -> None:
    """CLI entrypoint for running the benchmark."""
    run_benchmark()


if __name__ == "__main__":
    main()
