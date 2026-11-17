"""
MSCS 532 - Assignment 6
Medians and Order Statistics: selection algorithms.

This module will hold two in-place algorithms for the kth
smallest element of a list, both built on the same three-way
partition so that a list full of duplicates cannot force the
quadratic case:

  kth_smallest      - deterministic, worst-case O(n)
  select_random     - randomized, expected O(n)

k is 1-based: kth_smallest(a, 1) is the minimum and
kth_smallest(a, len(a)) is the maximum.

This is the first cut of the module: the shared partition and
the small-range sorter that both algorithms will use, plus a
demo. The two public entry points land in the next commits.
"""

_GROUP = 5     # group size for the median of medians
_CUTOFF = 9    # ranges of at most this size are just sorted


def _three_way(a, lo, hi, pivot, stats):
    """Dutch-national-flag partition of a[lo..hi] around pivot.

    On return, a[lo..lt-1] is strictly below pivot, a[lt..gt] is
    equal to it, and a[gt+1..hi] is strictly above it. Returns
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


def main():
    a = [29, 10, 14, 23, 12, 6, 1, 21]
    stats = {"comparisons": 0, "swaps": 0}
    _insertion_sort(a, 0, len(a) - 1, stats)
    print("insertion-sorted:", a)
    print("comparisons:", stats["comparisons"])
    b = [7, 7, 7, 7, 7]
    lt, gt = _three_way(b, 0, len(b) - 1, 7, stats)
    print(f"all-equal three-way: equal block is [{lt}, {gt}] of {len(b)}")
    print("swaps so far:", stats["swaps"])


if __name__ == "__main__":
    main()
