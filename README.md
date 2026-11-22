# Assignment 6: Medians and Order Statistics & Elementary Data Structures

For this assignment I implemented the two standard selection
algorithms in Python. The deterministic one is the median of
medians, which is O(n) in the worst case on every input order.
The randomized one is quickselect with a uniformly random pivot,
which is O(n) in expectation. Both share a three-way partition,
so arrays full of duplicates stay linear instead of degrading.
For Part 2 I implemented the elementary data structures from
scratch: a dynamic array, a matrix, an array stack, an array
queue, a singly linked list, and a rooted tree in linked form.

The measured results are in `results/` and `plots/`, and the
full analysis is in `report.md`.

## How to run

Python 3.8 or higher. matplotlib is only needed to regenerate the
plots.

```bash
python3 selection.py   # demo of both selection algorithms
python3 elements.py    # demo of each elementary structure
python3 tests.py       # test suite (21 tests)
python3 benchmark.py   # reruns the sweep, rewrites results/ and plots/
```

## Files

| File                  | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `selection.py`        | Median of medians and randomized quickselect + the demo |
| `elements.py`         | DynamicArray, Matrix, stack, queue, linked list, tree   |
| `benchmark.py`        | The selection sweep and the Part 2 experiments          |
| `tests.py`            | Test suite, including the complexity claims             |
| `report.md`           | The full report with the analysis                       |
| `results/`            | `selection.csv`, `times.csv`, `elements.csv`            |
| `plots/`              | `selection_comparisons.png`, `selection_time.png`       |

## Summary of findings

On the comparison counts, the deterministic variant is linear on
every input kind, which is the point of the median of medians.
Sorted input is often slightly cheaper than random input, because
sorted groups hand the median step exact medians. The randomized
variant is also linear on every input and is usually about half
the comparison count of the deterministic one, so it is the
faster choice in practice. The guarantee is what the
deterministic one buys: no input order can make it quadratic,
while the randomized one's quadratic tail still exists for an
unlucky pivot sequence. The three-way partition shows up on the heavy-duplicate input.
At n = 16384 the deterministic variant makes 63,268 comparisons
there against 161,059 on distinct random values, because the
large equal block ends each partition in one round instead of
peeling one element at a time.

For Part 2, the dynamic array's amortized append is measured at
just under one copy per append over 4096 appends, which is the
geometric-growth bound. The access experiment is the array
versus linked-list trade-off in one number: 500 random accesses
on 2048 elements take 0.02 ms through the array and 17.6 ms
through the linked list, about 893x. Stack, queue, and linked
list head operations all land in the same millisecond range for
40,000 operations, which is what O(1) head access looks like.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.). MIT Press.
- Blum, M., Floyd, R. W., Pratt, V., Rivest, R. L., & Tarjan, R. E. (1973).
  Time bounds for selection. *Journal of Computer and System Sciences, 7*(4),
  448-461. https://doi.org/10.1016/S0022-0000(73)80033-9
- Tarjan, R. E. (1985). Amortized computational complexity.
  *SIAM Journal on Algebraic and Discrete Methods, 6*(2), 306-318.
  https://doi.org/10.1137/0606031
