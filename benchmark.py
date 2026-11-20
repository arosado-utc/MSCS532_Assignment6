"""
MSCS 532 - Assignment 6
Measurement harness: the selection sweep (Part 1).

Run:  python3 benchmark.py

Output:
    results/selection.csv   one row per (kind, size, algorithm)
    results/times.csv       median wall time per (kind, size, algorithm)
    plots/selection_comparisons.png
    plots/selection_time.png

Measurement notes:
    * The comparison and swap counts come from the algorithms'
      own counters, so one run per case is exact and machine
      noise cannot move the number.
    * Wall time is the median of 5 runs for n <= 4096 and one run
      above that, where the largest cases take long enough on
      their own to damp the jitter.
    * Each (kind, size) input is generated once from a seeded
      generator, and both algorithms select from exactly the same
      input. The randomized variant's pivot draws are re-seeded
      per run, so its numbers are reproducible.
    * Sizes run 256 to 16384. The median is selected in every
      case, because that is the rank both analyses predict best.
"""

import csv
import math
import os
import random
import statistics
import time

from selection import kth_smallest, select_random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SEED = 20261116
SEL_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
KINDS = ["random", "sorted", "reverse_sorted", "heavy_dups"]
ALGORITHMS = [
    ("deterministic", kth_smallest),
    ("randomized", select_random),
]
TIME_RUNS = 5
SMALL_SIZE = 4096


def make_input(kind, n):
    """Build the n-element input of the requested kind (seeded)."""
    rng = random.Random(DATA_SEED + n)
    if kind == "random":
        return [rng.randint(0, n * 10) for _ in range(n)]
    base = [rng.randint(0, n * 10) for _ in range(n)]
    if kind == "sorted":
        return sorted(base)
    if kind == "reverse_sorted":
        return sorted(base, reverse=True)
    # heavy_dups: only 8 distinct values, so the three-way partition
    # has a large equal block to work with
    return [rng.randint(0, 7) for _ in range(n)]


def count_case(fn, data, k):
    """Run one selection and return its exact comparison/swap counts."""
    stats = {"comparisons": 0, "swaps": 0}
    fn(data[:], k, stats=stats)
    return stats


def median_time(fn, data, k):
    """Median wall time in milliseconds over repeated runs."""
    runs = 1 if len(data) > SMALL_SIZE else TIME_RUNS
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(data[:], k)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def sweep_selection():
    """Comparison/swap counts and median wall times for the sweep."""
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    comp_rows = []
    time_rows = []
    for kind in KINDS:
        for n in SEL_SIZES:
            data = make_input(kind, n)
            k = n // 2 + 1
            for name, fn in ALGORITHMS:
                random.seed(DATA_SEED + n * 7 + KINDS.index(kind))
                ops = count_case(fn, data, k)
                comp_rows.append([kind, n, name,
                                  ops["comparisons"], ops["swaps"]])
                random.seed(DATA_SEED + n * 7 + KINDS.index(kind))
                ms = median_time(fn, data, k)
                time_rows.append([kind, n, name, ms])
                print(f"{kind:15s} n={n:5d} {name:14s} "
                      f"comparisons={ops['comparisons']:>9d} "
                      f"swaps={ops['swaps']:>7d}  median={ms:8.3f} ms")

    with open(os.path.join(HERE, "results", "selection.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "size", "algorithm", "comparisons", "swaps"])
        w.writerows(comp_rows)

    with open(os.path.join(HERE, "results", "times.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "size", "algorithm", "median_ms"])
        for row in time_rows:
            w.writerow([row[0], row[1], row[2], f"{row[3]:.4f}"])

    return comp_rows, time_rows


def make_plots(comp_rows, time_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)

    style = {
        ("deterministic", "random"): ("tab:blue", "-", "o"),
        ("deterministic", "sorted"): ("tab:blue", "--", "s"),
        ("deterministic", "reverse_sorted"): ("tab:blue", ":", "^"),
        ("deterministic", "heavy_dups"): ("tab:blue", "-.", "P"),
        ("randomized", "random"): ("tab:red", "-", "D"),
        ("randomized", "sorted"): ("tab:red", "--", "v"),
        ("randomized", "reverse_sorted"): ("tab:red", ":", "x"),
        ("randomized", "heavy_dups"): ("tab:red", "-.", "*"),
    }

    for rows, yname, title, fname in (
        (comp_rows, "comparison count",
         "Selection: comparison count by input kind",
         "selection_comparisons.png"),
        (time_rows, "median wall time (ms)",
         "Selection: median wall time by input kind",
         "selection_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        seen = set()
        for kind, n, name, val, *_ in rows:
            c, ls, mk = style[(name, kind)]
            tag = f"{name} - {kind}"
            ax.plot(n, val, color=c, linestyle=ls, marker=mk,
                    label=tag if tag not in seen else None)
            seen.add(tag)
        # 2 n ln n reference: the cost of a full comparison sort
        theory = [2 * n * math.log(n) for n in SEL_SIZES]
        ax.plot(SEL_SIZES, theory, "k:", label="2 n ln n (full sort)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("input size n")
        ax.set_ylabel(yname)
        ax.set_title(title)
        ax.legend(fontsize=7, title="algorithm - kind")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "plots", fname), dpi=150)
        plt.close(fig)


def main():
    comp_rows, time_rows = sweep_selection()
    try:
        make_plots(comp_rows, time_rows)
    except ImportError:
        print("matplotlib is not installed; skipping the plots")
    print("wrote results/selection.csv, times.csv, plots/")


if __name__ == "__main__":
    main()
