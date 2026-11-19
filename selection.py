"""
MSCS 532 - Assignment 6
Medians and Order Statistics: selection algorithms.

Two in-place algorithms for the kth smallest element of a list,
both built on the same three-way partition so that a list full of
duplicates cannot force the quadratic case.

  kth_smallest      - deterministic, worst-case O(n): the pivot is
                      the median of the medians of groups of five
                      (Blum, Floyd, Pratt, Rivest & Tarjan, 1973).
  select_random     - randomized, expected O(n): the pivot is a
                      uniformly random element of the range.

k is 1-based: kth_smallest(a, 1) is the minimum and
kth_smallest(a, len(a)) is the maximum.

When a stats dict is passed, the partitions count their
comparisons and swaps into it, and the empirical analysis in the
report is built on those exact counts.
"""

import random

_GROUP = 5     # group size for the median of medians
_CUTOFF = 9    # ranges of at most this size are just sorted


def _three_way(a, lo, hi, pivot, stats):
    """Dutch-national-flag partition of a[lo..hi] around pivot.

    On return, a[lo..lt-1] is strictly below pivot, a[lt..gt] is
    equal to pivot, and a[gt+1..hi] is strictly above it. Returns
    the pair (lt, gt). The equal block is what keeps duplicate
    lists linear: a run of equal values ends in one partition
    instead of degrading into n-1 partitions of size n-1.

    Counting: each element is compared against pivot at most twice
    (once for <, once for ==), and every executed swap is counted.
    """
    lt, i, gt = lo, lo, hi
    while i <= gt:
        if stats is not None:
            stats["comparisons"] += 1
        if a[i] < pivot:
            a[i], a[lt] = a[lt], a[i]
            if stats is not None:
                stats["swaps"] += 1
            lt += 1
            i += 1
        else:
            if stats is not None:
                stats["comparisons"] += 1
            if a[i] == pivot:
                i += 1
            else:
                a[i], a[gt] = a[gt], a[i]
                if stats is not None:
                    stats["swaps"] += 1
                gt -= 1
    return lt, gt


def _insertion_sort(a, lo, hi, stats):
    """Sort the small range a[lo..hi] in place.

    Used for the groups of five and for the base case of the
    recursion. Shifts are moves, not swaps, so only comparisons
    are counted.
    """
    for i in range(lo + 1, hi + 1):
        x = a[i]
        j = i - 1
        while j >= lo:
            if stats is not None:
                stats["comparisons"] += 1
            if a[j] <= x:
                break
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


def _median_of_medians(a, lo, hi, stats):
    """Return the median of the group medians of a[lo..hi].

    Each group of _GROUP consecutive elements is insertion-sorted
    in place and its median is compacted to the front of the
    range, so the m = (hi - lo + 1) / 5 medians sit in
    a[lo..lo+m-1] and a recursive select picks their median in
    the same array. No auxiliary storage beyond the call stack.
    """
    m = 0
    i = lo
    while i <= hi:
        end = min(i + _GROUP - 1, hi)
        _insertion_sort(a, i, end, stats)
        mid = (i + end) // 2
        a[lo + m], a[mid] = a[mid], a[lo + m]
        if stats is not None:
            stats["swaps"] += 1
        m += 1
        i = end + 1
    if m == 1:
        return a[lo]
    k = (m + 1) // 2
    return _select(a, lo, lo + m - 1, k, stats)


def _select(a, lo, hi, k, stats):
    """Deterministic recursive driver. k is 1-based within the
    range a[lo..hi] (k = 1 is the smallest element of the range).
    """
    if hi - lo + 1 <= _CUTOFF:
        _insertion_sort(a, lo, hi, stats)
        return a[lo + k - 1]
    pivot = _median_of_medians(a, lo, hi, stats)
    lt, gt = _three_way(a, lo, hi, pivot, stats)
    eq_start = lt - lo + 1      # 1-based rank of the first pivot
    eq_end = gt - lo + 1        # 1-based rank of the last pivot
    if eq_start <= k <= eq_end:
        return pivot
    if k < eq_start:
        return _select(a, lo, lt - 1, k, stats)
    return _select(a, gt + 1, hi, k - eq_end, stats)


def kth_smallest(a, k, stats=None):
    """Return the kth smallest element of a (k is 1-based).

    The deterministic worst-case linear-time selection algorithm:
    every round pivots on the median of the medians of groups of
    five, which is guaranteed to discard at least 30% of the range
    no matter how the input is ordered, so no input order is a
    worst case. The list is reordered in place as a side effect
    (the same way quickselect leaves the list), but the kth
    smallest value is what the call returns.

    Parameters: a     - list of comparable elements
                k     - 1-based rank, 1 <= k <= len(a)
                stats - optional dict; filled with "comparisons"
                        and "swaps" when given
    Return:    the kth smallest value

    Complexity:
        Time   - O(n) worst case: T(n) <= T(n/5) + T(7n/10 + 6)
                 + O(n), which resolves to linear because 7/10 + 1/5
                 is below 1.
        Space  - O(log n) call stack (the median recursion) plus
                 O(1) working storage; the algorithm is in place.
    """
    if not a:
        raise ValueError("kth_smallest needs a non-empty list")
    if not 1 <= k <= len(a):
        raise ValueError(f"k={k} is out of range 1..{len(a)}")
    return _select(a, 0, len(a) - 1, k, stats)


def select_random(a, k, stats=None):
    """Return the kth smallest element of a (k is 1-based).

    Randomized quickselect: every round pivots on a uniformly
    random element of the range and discards the side that does
    not contain rank k. The expected number of comparisons is at
    most 4n for any k, so the expected time is O(n) on every input
    order; the quadratic tail still exists for a fixed unlucky
    pivot sequence, but it no longer depends on the input. The
    driver is written as a loop on the surviving range, so the
    stack stays O(1) no matter how the pivots land.

    Parameters: a     - list of comparable elements
                k     - 1-based rank, 1 <= k <= len(a)
                stats - optional dict; filled with "comparisons"
                        and "swaps" when given
    Return:    the kth smallest value

    Complexity:
        Time   - O(n) in expectation for every input and every k.
        Space  - O(1) beyond the input: the loop replaces the
                 recursion, and the partition is in place.
    """
    if not a:
        raise ValueError("select_random needs a non-empty list")
    if not 1 <= k <= len(a):
        raise ValueError(f"k={k} is out of range 1..{len(a)}")
    lo, hi = 0, len(a) - 1
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


def main():
    samples = [
        ([29, 10, 14, 23, 12, 6, 1, 21], 4),
        ([7, 7, 7, 7, 7], 3),
        ([5, 3, 5, 1, 5, 3, 5, 2], 5),
        (list(range(1, 17)), 8),
        ([-4, 2, -4, 9, -4, 0], 3),
    ]
    for arr, k in samples:
        expect = sorted(arr)[k - 1]
        print(f"k={k:2d}  input={arr}")
        print(f"  kth_smallest : {kth_smallest(arr[:], k)}")
        print(f"  select_random: {select_random(arr[:], k)}")
        print(f"  expected     : {expect}")
        print()


if __name__ == "__main__":
    main()
