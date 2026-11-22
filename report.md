# Assignment 6: Medians and Order Statistics & Elementary Data Structures

**Alfredo Rosado** — MSCS 532, Algorithms and Data Structures
Dr. Michael Solomon — November 22, 2026

## 1. Introduction

This report covers two parts. Part 1 implements the two standard
algorithms for selecting the kth smallest element of an array, the
order statistics problem. The deterministic algorithm is the
median of medians, which runs in O(n) in the worst case on every
input order (Blum et al., 1973). The randomized algorithm is
quickselect, which runs in O(n) in expectation. Both are built on
the same three-way partition, which is what keeps duplicate-heavy
arrays linear. Part 2 implements the elementary data structures
from scratch in Python, a dynamic array, a matrix, an array
stack, an array queue, a singly linked list, and a rooted tree in
linked form, and analyzes the cost of each operation.

The source is in `selection.py` and `elements.py`, the
measurement harness in `benchmark.py`, and the test suite in
`tests.py`; the measured data lives in `results/` and `plots/`.
Like Assignment 5, the empirical section is built on the
algorithms' own comparison and swap counters, so the numbers are
exact and one run per case is enough.

## 2. Part 1: Implementation

### 2.1 The shared three-way partition

Both algorithms partition with the Dutch national flag scheme
around a pivot value, not a pivot position. Three cursors sweep
the range once. On return, `a[lo..lt-1]` is strictly below the
pivot, `a[lt..gt]` is equal to it, and `a[gt+1..hi]` is strictly
above it.

```python
def _three_way(a, lo, hi, pivot, stats):
    lt, i, gt = lo, lo, hi
    while i <= gt:
        if a[i] < pivot:
            a[i], a[lt] = a[lt], a[i]
            lt += 1
            i += 1
        elif a[i] == pivot:
            i += 1
        else:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
    return lt, gt
```

The equal block is the reason the scheme is here. A two-way
partition peels one equal element per round on a duplicate-heavy
input, which is the same degenerate behavior Assignment 5 showed
for quicksort. The three-way split removes every equal element in
one round, so a list of all equal values finishes in a single
partition. Each element is compared against the pivot at most
twice, once for `<` and once for `==`, and every executed swap
is counted in the `stats` dict.

### 2.2 The deterministic variant

`kth_smallest(a, k)` uses the median of medians pivot choice.
Each round the range is split into groups of five, each group is
insertion-sorted in place, and each group median is compacted to
the front of the range. A recursive select then picks the median
of those medians, and the range is three-way partitioned around
it. The round recurses on the side that contains rank k, which
loses at least 30% of the elements no matter how the input is
ordered.

```python
def _median_of_medians(a, lo, hi, stats):
    m = 0
    i = lo
    while i <= hi:
        end = min(i + _GROUP - 1, hi)
        _insertion_sort(a, i, end, stats)
        mid = (i + end) // 2
        a[lo + m], a[mid] = a[mid], a[lo + m]
        m += 1
        i = end + 1
    if m == 1:
        return a[lo]
    return _select(a, lo, lo + m - 1, (m + 1) // 2, stats)
```

The median compaction keeps the algorithm in place. The m group
medians land in `a[lo..lo+m-1]` and the recursive call selects
their median in the same array, so no auxiliary storage is needed
beyond the call stack. Ranges of at most nine elements are just
sorted, which bounds the constant and stops the recursion.

### 2.3 The randomized variant

`select_random(a, k)` is the same partition with a uniformly
random pivot per round, drawn fresh from the surviving range.
Each round discards the side that does not contain rank k. The
driver is written as a loop on the surviving range rather than a
recursion, so the stack stays O(1) no matter how the pivots
land. This differs from Assignment 5 on purpose. That report
used Hoare partition with a last-element pivot and a
smaller-half recursion, while here the scheme, the pivot rule,
and the driver are all different.

```python
while True:
    if lo >= hi:
        return a[lo]
    pivot = a[random.randint(lo, hi)]
    lt, gt = _three_way(a, lo, hi, pivot, stats)
    eq_start = lt - lo + 1
    eq_end = gt - lo + 1
    if eq_start <= k <= eq_end:
        return pivot
    if k < eq_start:
        hi = lt - 1
    else:
        k -= eq_end
        lo = gt + 1
```

Both functions reorder the list in place as a side effect, the
same way quickselect does, and both return the kth smallest value
with k one-based. Both raise `ValueError` on an empty list or an
out-of-range k.

## 3. Part 1: Performance analysis

### 3.1 Why the deterministic variant is O(n) in the worst case

A group of five has a median such that at least two group
members are below it and at least two are above it. Since at
least half of the groups have their median in the upper half of
the median list, at least 3n/10 elements are below the pivot and
at least 3n/10 are above it, minus the bookkeeping. So every
round keeps at most 7n/10 + 6 elements, and the median-of-medians
step itself costs T(n/5). The recurrence is

`T(n) <= T(n/5) + T(7n/10 + 6) + c n`,

and it resolves to O(n) because 1/5 + 7/10 = 9/10 is below 1.
The key point is that the 30% discard holds for every input
order, which is exactly what a fixed pivot rule cannot give.
Assignment 5 measured the last-element pivot at 8,385,942
comparisons on sorted 4096, a pure n^2 signature. The median of
medians has no such signature on any input order, which the
measurements in Section 4 confirm.

### 3.2 Why the randomized variant is O(n) in expectation

For any fixed pivot, the probability that it is a good pivot,
one that splits the range within the middle half, is 1/2, and
the pivot draws are independent across rounds. The expected
range size therefore halves about every two rounds, which makes
the expected total work geometric and O(n). The standard bound
(Cormen et al., 2022) is at most 4n comparisons for any k, and
the measurement at n = 2048 lands at 8,122, which is 3.97n,
right on that bound. The worst case still exists for a fixed
unlucky pivot sequence, but it requires bad draws in a row and
no longer depends on the input.

### 3.3 Space and other overheads

- The deterministic variant is in place. Its call stack holds
  one frame per level of the median tree plus the surviving
  range, which is O(log n). The test suite sorts 10,000 sorted
  elements under a 300-frame recursion limit to pin this down.
- The randomized variant is in place with an O(1) stack, since
  the driver is a loop.
- Each comparison does constant work, so the comparison count
  is the true cost of the comparison-based selection.
- The insertion sort of the groups of five costs at most 10
  comparisons per group, which is absorbed in the c n term.

### 3.4 Duplicates

On an all-equal list the three-way partition finishes in one
round. The measured comparison count for 1,000 equal elements is
3,479, about 3.5n, which is the cost of building the medians.
A two-way partition on the same input would make about n^2/2
comparisons.

## 4. Part 1: Empirical analysis

### 4.1 Method

`benchmark.py` selects the median from the same seeded input
with both variants across four input kinds, random, sorted,
reverse-sorted, and heavy duplicates with only 8 distinct
values. Sizes run 256 to 16384. The comparison and swap counts
come from the counters, so one run per case is exact. Wall time
is the median of 5 runs for n <= 4096 and one run above that.
The randomized variant's pivot draws are re-seeded per run, so
its numbers are reproducible.

### 4.2 Comparison counts

From `results/selection.csv`, selecting the median:

| n     | det random | rand random | det sorted | rand sorted | det reverse | rand reverse | det dups | rand dups |
| ----- | ---------- | ----------- | ---------- | ----------- | ----------- | ------------ | -------- | --------- |
| 256   | 2,178      | 1,528       | 1,715      | 837         | 2,140       | 1,754        | 984      | 582       |
| 512   | 4,517      | 1,842       | 3,570      | 2,447       | 4,335       | 3,696        | 1,987    | 836       |
| 1024  | 9,202      | 3,788       | 3,029      | 4,091       | 8,252       | 5,128        | 3,767    | 2,910     |
| 2048  | 18,035     | 8,122       | 15,519     | 9,686       | 18,561      | 9,680        | 15,565   | 6,392     |
| 4096  | 39,456     | 13,520      | 33,060     | 15,687      | 36,719      | 24,777       | 27,377   | 14,859    |
| 8192  | 78,884     | 27,096      | 61,708     | 43,406      | 77,113      | 31,823       | 62,385   | 33,818    |
| 16384 | 161,059    | 76,031      | 60,206     | 89,158      | 153,096     | 73,757       | 63,268   | 83,608    |

### 4.3 Wall time

Representative medians from `results/times.csv`, in ms:

| kind / n   | det ms | rand ms | kind / n    | det ms | rand ms |
| ---------- | ------ | ------- | ----------- | ------ | ------- |
| random 1024  | 0.4933 | 0.1671 | sorted 16384  | 2.8746 | 4.1505 |
| random 16384 | 8.4805 | 2.9441 | reverse 1024  | 0.3443 | 0.1188 |
| dups 16384   | 3.0899 | 2.4686 | reverse 16384 | 8.7295 | 2.2618 |

The full tables and the log-log plots are in `plots/
selection_comparisons.png` and `plots/selection_time.png`,
alongside a 2 n ln n reference line, the cost of a full
comparison sort.

### 4.4 Relating the measurements to the theory

- **No input-order worst case.** The deterministic variant grows
  about 2x per doubling on every kind, 73.9x total on random
  from 256 to 16384, 71.5x on reverse-sorted, 35.1x on sorted.
  Every ratio is the linear signature. None of them approaches
  the 4x-per-doubling that Assignment 5's last-element pivot
  showed on sorted input. At n = 16384 the count is about 10n,
  while n^2/2 would be 134 million comparisons. The constant
  varies from input to input, which is why the sorted column
  wobbles between adjacent sizes, but the growth across the full
  range stays linear.
- **Sorted is often cheaper than random.** The sorted counts
  are usually below the random ones, 60,206 versus 161,059 at
  n = 16384. The reason is structural. Sorted groups of five
  hand the median step exact medians, so the first partition
  splits near the middle and the recursion keeps getting clean
  splits. Random groups give pivots that are good on average
  but imperfect on any single round.
- **The randomized variant is linear everywhere too.** Its
  counts track the same 2x-per-doubling on every kind, with the
  single-run noise a randomized method carries. At n = 2048 on
  random input the count is 8,122, which is 3.97n, right on the
  4n bound from Section 3.2.
- **The 2 n ln n line stays on top.** Every selection curve
  sits well below the full-sort reference, which is the
  separation between selection, O(n), and sorting, O(n log n).
- **Duplicates pay off.** At n = 16384 the deterministic
  variant makes 63,268 comparisons on the 8-value input
  against 161,059 on distinct random values, 2.5x fewer, and
  the all-equal case at n = 1,000 finishes in 3,479
  comparisons.

## 5. Part 2: Implementation

All structures are implemented from scratch in `elements.py`
with no list or dict shortcuts in the core operations.

| Structure      | Operation              | Worst case     | Amortized     |
| -------------- | ---------------------- | -------------- | ------------- |
| DynamicArray   | append                 | O(n) at resize | O(1)          |
| DynamicArray   | insert / delete        | O(n)           | O(n)          |
| DynamicArray   | access / set           | O(1)           | O(1)          |
| Matrix         | get / set              | O(1)           | O(1)          |
| Matrix         | insert / delete row    | O(cols) + list move | —      |
| Matrix         | insert / delete col    | O(rows)        | —             |
| ArrayStack     | push / pop / peek      | O(1)           | O(1)          |
| ArrayQueue     | enqueue / dequeue      | O(n) at compaction | O(1)   |
| LinkedList     | insert_head / delete_head | O(1)       | O(1)          |
| LinkedList     | insert_tail / delete_tail | O(n)     | O(n)          |
| LinkedList     | insert_at / delete_at / get | O(i) <= O(n) | —       |
| LinkedList     | traversal              | O(n)           | O(n)          |
| RootedTree     | add_child              | O(n) search, O(1) link | —    |
| RootedTree     | find / depth           | O(n)           | O(n)          |
| RootedTree     | preorder / size        | O(n)           | O(n)          |

A few design notes. The DynamicArray's backing buffer doubles
when full and halves when the load factor drops to 1/4, and it
counts every element it copies across a resize. The ArrayQueue
is a buffer with a moving head index instead of a fixed-size
circular array, so it never needs a capacity parameter. Dequeue
advances the head and compacts once at least 32 slots have been
abandoned, which keeps the worst-case compaction rare and the
amortized cost O(1). The RootedTree uses the first-child /
next-sibling link form, so the whole tree is built from the same
two pointers a linked list uses, and `add_child` makes the new
node the parent's eldest child.

## 6. Part 2: Empirical results

From `results/elements.csv`:

| Experiment | n    | Detail            | Value  |
| ---------- | ---- | ----------------- | ------ |
| dynarray   | 256  | resizes           | 6      |
| dynarray   | 256  | copies            | 252    |
| dynarray   | 1024 | resizes           | 8      |
| dynarray   | 1024 | copies            | 1,020  |
| dynarray   | 4096 | resizes           | 10     |
| dynarray   | 4096 | copies            | 4,092  |
| access     | 2048 | array_ms (500 random accesses) | 0.0197 |
| access     | 2048 | linked_ms (500 random accesses) | 17.5602 |
| headops    | 20000 | stack_ms (2 x 20,000 ops) | 1.2907 |
| headops    | 20000 | queue_ms (2 x 20,000 ops) | 2.5742 |
| headops    | 20000 | linked_ms (2 x 20,000 ops) | 4.4198 |

The dynamic array copied 4,092 elements across 10 resizes during
4,096 appends, 0.999 copies per append. That is the geometric
growth bound in action. The number of resizes grows logarithmically,
6, 8, and 10 across the three sizes. The access experiment is
500 random-index accesses on 2,048 elements. Through the array
they take 0.02 ms in total, and through the linked list they
take 17.56 ms, a 893x gap. Each list access walks on average
n/2 links, so the list does about 512,000 link hops against the
array's 500, which is the O(1) versus O(n) access difference
measured directly. The head-operation rows show stack, queue,
and linked list all completing 40,000 operations in the same
millisecond range, the O(1) head access the three structures
share.

## 7. Part 2: Discussion

### Arrays versus linked lists

The 893x access gap is the whole trade-off in one number.
Arrays hold their elements contiguously, so index i is a single
address computation and the hardware prefetcher keeps the cache
warm. Linked lists store one node per element, each a separate
allocation, so access i is i pointer chases through memory the
cache cannot predict. For head operations the picture reverses
in the list's favor on insertion, since linking one node costs
nothing regardless of length, while an array insertion at the
front shifts every element. In practice I would reach for an
array when the access pattern is random or the structure is
read-heavy, and for a linked list when the work is at the ends
or the elements are already distributed, as in the LRU cache of
my project, where the recency list exists precisely because
moving a node is cheaper than moving an array's contents.

### Where each structure shows up

- **Arrays and matrices** are the default for anything indexed.
  Numeric libraries, image buffers, and hash table storage all
  rely on contiguous access, and the matrix row/column
  operations are the same trick a spreadsheet engine uses for
  insert and delete.
- **Stacks** are the call stack, the undo buffer, and the
  evaluation order for expressions. Anything that processes the
  most recent thing first is a stack, and O(1) push and pop are
  what make the call stack cheap enough to grow per function
  call.
- **Queues** are the ready list of an OS scheduler, the I/O
  buffer between a producer and a consumer, and the frontier of
  a breadth-first search. The FIFO order is the semantics, and
  O(1) amortized enqueue and dequeue are what let the queue
  grow without a cap.
- **Linked lists** appear wherever nodes are inserted and
  removed in the middle of a live stream, in memory allocators,
  in the doubly linked list behind an LRU cache, and anywhere
  the data arrives one element at a time.
- **The tree form** is the DOM of a browser, the directory tree
  of a file system, and the node structure of any hierarchical
  data, and the linked first-child / next-sibling form is how
  those are stored.

The common lesson is that the right structure is a property of
the access pattern, not of the data. The same 2,048 elements
cost 0.02 ms or 17.56 ms depending on which structure holds
them, and that is the difference the analysis section predicted.

## 8. Conclusion

The implementation matches the theory on every axis it makes a
prediction on. The deterministic variant is linear on all four
input kinds with no input-order worst case, the randomized
variant is linear everywhere and lands on the 4n bound, the
three-way partition keeps duplicate-heavy inputs linear, and the
dynamic array's amortized append measures at one copy per
append. For Part 2 the measured access gap, 893x on random
access, is the concrete version of the O(1) versus O(n)
difference, and the head-operation rows confirm the O(1) end
access the three structures share.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.). MIT Press. (Chapter 9:
  Medians and order statistics, including randomized select and the
  median of medians.)
- Blum, M., Floyd, R. W., Pratt, V., Rivest, R. L., & Tarjan, R. E. (1973).
  Time bounds for selection. *Journal of Computer and System Sciences, 7*(4),
  448-461. https://doi.org/10.1016/S0022-0000(73)80033-9
- Tarjan, R. E. (1985). Amortized computational complexity.
  *SIAM Journal on Algebraic and Discrete Methods, 6*(2), 306-318.
  https://doi.org/10.1137/0606031
