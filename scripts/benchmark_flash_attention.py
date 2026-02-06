"""Benchmark for flash attention implementations."""

import json

import torch
from triton.testing import do_bench

from cs336_systems.attention_triton import (
    AttentionTriton,
    attention_regular,
)
from cs336_systems.output_utils import get_output_dir

BATCH_SIZE = 1
SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
PRECISIONS = ["fp32", "bf16"]
IMPL_NAMES = ["regular", "triton_flash"]


def _call_impl(
    impl_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the correct implementation."""
    if impl_name == "regular":
        return attention_regular(q, k, v, is_causal=True)
    return AttentionTriton.apply(q, k, v, True)  # noqa: FBT003


def _bench_forward(
    impl_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> float:
    return do_bench(lambda: _call_impl(impl_name, q, k, v))


def _bench_backward(
    impl_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> float:
    out = _call_impl(impl_name, q, k, v)
    grad_o = torch.ones_like(out)
    return do_bench(lambda: out.backward(grad_o, retain_graph=True))


def _bench_end2end(
    impl_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> float:
    def fn() -> None:
        _call_impl(impl_name, q, k, v).sum().backward()

    return do_bench(fn)


def _make_tensors(
    seq: int,
    d_model: int,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    q = torch.randn(
        BATCH_SIZE,
        seq,
        d_model,
        dtype=dtype,
        device="cuda",
    ).requires_grad_(True)  # noqa: FBT003
    k = torch.randn(
        BATCH_SIZE,
        seq,
        d_model,
        dtype=dtype,
        device="cuda",
    ).requires_grad_(True)  # noqa: FBT003
    v = torch.randn(
        BATCH_SIZE,
        seq,
        d_model,
        dtype=dtype,
        device="cuda",
    ).requires_grad_(True)  # noqa: FBT003
    return q, k, v


def _benchmark_one(
    seq: int,
    d_model: int,
    precision: str,
    impl_name: str,
) -> dict:
    """Benchmark a single configuration."""
    q, k, v = _make_tensors(seq, d_model, precision)
    result: dict = {
        "seq_len": seq,
        "d_model": d_model,
        "precision": precision,
        "impl": impl_name,
    }
    try:
        result["fwd_ms"] = _bench_forward(impl_name, q, k, v)
        result["bwd_ms"] = _bench_backward(impl_name, q, k, v)
        result["e2e_ms"] = _bench_end2end(impl_name, q, k, v)
        result["status"] = "OK"
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        torch.cuda.empty_cache()
    fwd = _fmt(result.get("fwd_ms", "OOM"))
    bwd = _fmt(result.get("bwd_ms", "OOM"))
    e2e = _fmt(result.get("e2e_ms", "OOM"))
    print(
        f"  {impl_name:12s} seq={seq:5d} d={d_model:3d}"
        f" {precision:4s}: {fwd:>10} / {bwd:>10} / {e2e:>10}"
    )
    return result


def bench() -> list[dict]:
    """Run all benchmark configurations."""
    results: list[dict] = []
    for precision in PRECISIONS:
        for seq in SEQ_LENGTHS:
            for d_model in D_MODELS:
                for impl_name in IMPL_NAMES:
                    r = _benchmark_one(
                        seq,
                        d_model,
                        precision,
                        impl_name,
                    )
                    results.append(r)
    return results


def _get_val(
    subset: list[dict],
    seq: int,
    d_model: int,
    impl: str,
    key: str,
) -> float | str:
    """Extract a timing value from results."""
    row = next(
        (
            r
            for r in subset
            if r["seq_len"] == seq and r["d_model"] == d_model and r["impl"] == impl
        ),
        None,
    )
    if row and row["status"] == "OK":
        return row.get(key, "—")
    return "OOM"


def _fmt(val: float | str) -> str:
    """Format a timing value for table display."""
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _speedup(base: float | str, target: float | str) -> str:
    """Compute speedup string: base / target."""
    if isinstance(base, float) and isinstance(target, float) and target > 0:
        return f"{base / target:.2f}x"
    return "—"


def generate_markdown(results: list[dict]) -> str:
    """Generate markdown tables from benchmark results.

    For each (precision, d_model, phase) combination, produces a table with
    columns: seq_len | regular | triton_flash | triton vs regular
    """
    lines = ["### Flash Attention Benchmark Results\n"]
    for precision in PRECISIONS:
        subset = [r for r in results if r["precision"] == precision]
        if not subset:
            continue
        for phase, key in [
            ("Forward", "fwd_ms"),
            ("Backward", "bwd_ms"),
            ("End-to-End", "e2e_ms"),
        ]:
            lines.append(f"#### {phase} Latency — {precision} (ms)\n")
            for d in D_MODELS:
                lines.append(f"**d_model = {d}**\n")
                lines.append("| seq_len | regular | triton_flash | triton vs regular |")
                lines.append("| --- | --- | --- | --- |")
                for seq in SEQ_LENGTHS:
                    reg = _get_val(subset, seq, d, "regular", key)
                    tr = _get_val(subset, seq, d, "triton_flash", key)
                    row = f"| {seq} | {_fmt(reg)} | {_fmt(tr)} | {_speedup(reg, tr)} |"
                    lines.append(row)
                lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    all_results = bench()

    output_dir = get_output_dir()
    json_path = output_dir / "flash_attention_benchmark.json"
    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    md = generate_markdown(all_results)
    md_path = output_dir / "flash_attention_benchmark.md"
    md_path.write_text(md)
    print(f"Markdown saved: {md_path}")

    docs_path = output_dir.parent / "docs" / "sections" / "flash-attention-benchmark.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(md)
    print(f"Docs section saved: {docs_path}")
