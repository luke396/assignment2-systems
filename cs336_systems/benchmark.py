"""Code for benchmarking_script."""

import argparse
import statistics
import timeit
from dataclasses import dataclass

import torch
from cs336_basics import blocks
from cs336_basics.blocks import TransformerLM
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
    nvtx_attn: bool = False


@dataclass(frozen=True)
class RunInputs:
    """Runtime inputs for a benchmark run."""

    vocab_size: int
    batch_size: int
    context_length: int
    steps: int
    warm_up_steps: int
    seed: int


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
        "--nvtx-attn",
        action="store_true",
        help="Enable NVTX ranges inside scaled dot-product attention.",
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
        return torch.device("cpu")
    try:
        torch.empty(1, device="cuda")
    # CUDA initialization can raise a mix of runtime and driver errors.
    except (RuntimeError, OSError, AssertionError):
        return torch.device("cpu")
    return torch.device("cuda")


def _setup_training(
    model: torch.nn.Module, *, forward_only: bool, lr: float, weight_decay: float
) -> tuple[torch.optim.Optimizer | None, torch.nn.Module | None]:
    """Create optimizer and loss function unless running forward-only."""
    if forward_only:
        return None, None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    return optimizer, loss_fn


def _run_steps(
    model: torch.nn.Module,
    *,
    device: torch.device,
    run_inputs: RunInputs,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: torch.nn.Module | None,
) -> tuple[list[float], float, float]:
    """Run benchmark steps and return timing stats."""
    step_times: list[float] = []
    start = timeit.default_timer()
    warm_up_end = start

    model.train()
    for step in range(run_inputs.steps):
        step_start = timeit.default_timer()

        data = generate_random_data(
            seq_len=run_inputs.context_length,
            vocab_size=run_inputs.vocab_size,
            batch_size=run_inputs.batch_size,
            seed=run_inputs.seed + step,
            device=device,
        )

        if device.type == "cuda":
            with nvtx.range("forward"):  # profile the forward pass
                logit = model(data)
        else:
            logit = model(data)

        if optimizer is not None and loss_fn is not None:
            if device.type == "cuda":
                with nvtx.range("backward"):  # profile the backward pass
                    optimizer.zero_grad()
                    loss = loss_fn(logit.view(-1, run_inputs.vocab_size), data.view(-1))
                    loss.backward()
                with nvtx.range("optimizer_step"):  # profile the optimizer step
                    optimizer.step()
            else:
                optimizer.zero_grad()
                loss = loss_fn(logit.view(-1, run_inputs.vocab_size), data.view(-1))
                loss.backward()
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_end = timeit.default_timer()

        # Record time after warm-up phase completes
        if step == run_inputs.warm_up_steps - 1:
            warm_up_end = step_end

        # Only record measurement steps (after warm-up)
        if step >= run_inputs.warm_up_steps:
            step_times.append(step_end - step_start)

    end = timeit.default_timer()
    total_time = end - start
    warm_up_time = warm_up_end - start
    return step_times, total_time, warm_up_time


def run_basic_benchmark_default() -> None:
    """Run the benchmark."""
    args = parse_cli_args()
    run_inputs = RunInputs(
        vocab_size=10000,  # fixed
        batch_size=4,  # fixed
        context_length=args.context_length,
        steps=args.steps,
        warm_up_steps=args.warm_up_steps,
        seed=args.seed,
    )
    device = select_benchmark_device()
    if args.nvtx_attn and device.type == "cuda":
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
    step_times, total_time, warm_up_time = _run_steps(
        model,
        device=device,
        run_inputs=run_inputs,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # Calculate statistics for measurement steps
    avg_time = statistics.mean(step_times) if step_times else 0.0
    std_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    print(f"Total time for {args.steps} steps: {total_time:.4f} seconds")
    print(f"Warm-up time ({args.warm_up_steps} steps): {warm_up_time:.4f} seconds")
    print(f"Avg time per step after warm-up: {avg_time:.4f} seconds")
    print(f"Std time per step after warm-up: {std_time:.6f} seconds")


def main() -> None:
    """CLI entrypoint for running the benchmark."""
    run_basic_benchmark_default()


if __name__ == "__main__":
    main()
