"""Plot bucketed DDP comparison from the combined markdown table."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMBINED_MD = Path("docs/sections/ddp-xl-bucketed.md")
NAIVE_MD = Path("docs/sections/ddp-xl-naive.md")
OUT_PATH = Path("docs/images/ddp-xl-bucketed.png")


def parse_combined_md(
    path: Path,
) -> tuple[list[str], dict[str, list[tuple[int, float]]]]:
    """Parse combined markdown table into {label: [(seq_len, total_ms), ...]}."""
    lines = path.read_text().splitlines()

    # Find header row (starts with "| Seq Len |").
    labels: list[str] = []
    for line in lines:
        if line.startswith("| Seq Len"):
            cols = [c.strip() for c in line.split("|")]
            # cols[0]='', cols[1]='Seq Len', cols[2..n-1]=labels, cols[n]=''
            labels = [re.sub(r"\s*\(ms\)\s*$", "", c) for c in cols[2:] if c]
            break

    if not labels:
        msg = f"No header row found in {path}"
        raise ValueError(msg)

    data: dict[str, list[tuple[int, float]]] = {lb: [] for lb in labels}
    for line in lines:
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        cols = [c.strip() for c in line.split("|")]
        sl = int(cols[1])
        for i, lb in enumerate(labels):
            val = cols[i + 2]
            if val != "OOM":
                data[lb].append((sl, float(val)))

    return labels, data


def parse_naive_breakdown(
    path: Path,
) -> list[tuple[int, float, float, float]]:
    """Parse naive DDP table into [(seq_len, fwd_ms, bwd_ms, allreduce_ms)].

    Supports both old format (Fwd+Bwd combined) and new format (Fwd, Bwd separate).
    For old format, estimates Fwd ≈ 1/3 and Bwd ≈ 2/3 of Fwd+Bwd.
    """
    lines = path.read_text().splitlines()
    # Detect format from header: new format has "Bwd" column.
    has_bwd_col = any("Bwd" in line for line in lines if line.startswith("|"))

    rows: list[tuple[int, float, float, float]] = []
    for line in lines:
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        cols = [c.strip() for c in line.split("|")]
        if cols[2] == "OOM":
            continue
        if has_bwd_col:
            # New: | Seq | Fwd | Bwd | AllReduce | ...
            fwd, bwd, comm = float(cols[2]), float(cols[3]), float(cols[4])
        else:
            # Old: | Seq | Fwd+Bwd | AllReduce | ...
            fwd_bwd, comm = float(cols[2]), float(cols[3])
            fwd, bwd = fwd_bwd / 3, fwd_bwd * 2 / 3
        rows.append((int(cols[1]), fwd, bwd, comm))
    return rows


def _plot_speedup(
    ax: plt.Axes,
    labels: list[str],
    data: dict[str, list[tuple[int, float]]],
) -> None:
    """Plot speedup vs Naive for Overlapped and Bucketed variants."""
    naive_map: dict[int, float] = {}
    if "Naive" in data:
        naive_map = dict(data["Naive"])

    for lb in labels:
        if lb == "Naive":
            continue
        points = data[lb]
        if not points:
            continue
        seqs, speedups = [], []
        for sl, t in points:
            base = naive_map.get(sl)
            if base:
                seqs.append(sl)
                speedups.append((base - t) / base * 100)
        if not seqs:
            continue
        marker = "o" if "Bucketed" in lb else "s"
        linestyle = "--" if lb == "Overlapped" else "-"
        linewidth = 2.5 if lb == "Overlapped" else 1.5
        ax.plot(
            seqs,
            speedups,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=lb,
        )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Speedup vs Naive (%)")
    ax.set_title("Speedup over Naive DDP (higher = faster)")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)


def _plot_compute_vs_comm(
    ax: plt.Axes,
    naive_breakdown: list[tuple[int, float, float, float]],
) -> None:
    """Plot backward vs communication time from naive breakdown."""
    seq_lens = [r[0] for r in naive_breakdown]
    bwd = [r[2] for r in naive_breakdown]
    comm = [r[3] for r in naive_breakdown]
    x = np.arange(len(seq_lens))
    w = 0.35

    ax.bar(
        x - w / 2,
        bwd,
        w,
        label="Backward (ms)",
        color="#4C72B0",
    )
    ax.bar(x + w / 2, comm, w, label="AllReduce (comm)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Backward vs Communication: only backward overlaps with comm")
    ax.legend()
    ax.grid(visible=True, axis="y", alpha=0.3)

    # Find crossover: where backward estimate first exceeds comm
    cross_x = None
    for i in range(len(bwd)):
        if bwd[i] > comm[i]:
            cross_x = i
            break

    if cross_x is not None:
        ax.axvline(
            x=cross_x - 0.5,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        y_top = ax.get_ylim()[1]
        ax.text(
            cross_x - 0.7,
            y_top * 0.85,
            "Comm > Backward\nCan't hide comm\n-> Bucketed wins\n   (fewer launches)",
            fontsize=8,
            ha="right",
            color="#4C72B0",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.text(
            cross_x - 0.3,
            y_top * 0.85,
            "Backward > Comm\nComm fully hidden\n"
            "-> Overlapped wins\n   (finer overlap)",
            fontsize=8,
            ha="left",
            color="#DD8452",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )


def main() -> None:
    """Generate comparison plot from combined markdown table."""
    labels, data = parse_combined_md(COMBINED_MD)
    naive_breakdown = parse_naive_breakdown(NAIVE_MD)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    _plot_speedup(ax1, labels, data)
    _plot_compute_vs_comm(ax2, naive_breakdown)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
