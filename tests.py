"""
MSCS 532 - Assignment 6
Test suite for the selection algorithms (Part 1) and the
elementary data structures (Part 2).

Run:  python3 tests.py

Every test is a plain function; the runner at the bottom calls
them all and reports pass/fail per test. The selection tests
check correctness against the built-in sort, the edge cases,
the duplicate handling, and - using the comparison/swap
counters - the complexity claims from the report (the exact
deterministic counts, the absence of an input-order worst case,
and the linear band on random input). The structure tests
check each operation against its documented complexity and
semantics.
"""

import random
import sys

from elements import (ArrayQueue, ArrayStack, DynamicArray, LinkedList,
                      Matrix, RootedTree)
from selection import kth_smallest, select_random


def test_min_and_max_ranks():
    a = [5, 2, 8, 1, 9]
    assert kth_smallest(a[:], 1) == 1
    assert kth_smallest(a[:], 5) == 9
    assert select_random(a[:], 1) == 1
    assert select_random(a[:], 5) == 9


def test_error_cases():
    for fn in (kth_smallest, select_random):
        try:
            fn([], 1)
            assert False, "empty list must raise"
        except ValueError:
            pass
        try:
            fn([1, 2, 3], 0)
            assert False, "k=0 must raise"
        except ValueError:
            pass
        try:
            fn([1, 2, 3], 4)
            assert False, "k past the end must raise"
        except ValueError:
            pass
    assert kth_smallest([7], 1) == 7
    assert select_random([7], 1) == 7


def test_all_equal():
    a = [4] * 50
    assert kth_smallest(a[:], 25) == 4
    assert select_random(a[:], 1) == 4
    assert select_random(a[:], 50) == 4


def test_duplicates():
    a = [5, 3, 5, 1, 5, 3, 5, 2]
    for k in range(1, len(a) + 1):
        expect = sorted(a)[k - 1]
        assert kth_smallest(a[:], k) == expect
        assert select_random(a[:], k) == expect


def test_negatives_and_mixed():
    a = [-4, 2, -4, 9, -4, 0, 2, -9]
    for k in (1, 3, 6, 8):
        expect = sorted(a)[k - 1]
        assert kth_smallest(a[:], k) == expect
        assert select_random(a[:], k) == expect


def test_agrees_with_builtin_on_random_inputs():
    rng = random.Random(1234)
    for _ in range(300):
        n = rng.randint(1, 90)
        a = [rng.randint(-100, 100) for _ in range(n)]
        k = rng.randint(1, n)
        expect = sorted(a)[k - 1]
        assert kth_smallest(a[:], k) == expect
        assert select_random(a[:], k) == expect


def test_both_variants_agree_on_same_input():
    for seed in range(50):
        rng = random.Random(seed)
        n = rng.randint(20, 200)
        a = [rng.randint(0, 50) for _ in range(n)]
        k = rng.randint(1, n)
        assert kth_smallest(a[:], k) == select_random(a[:], k)


def test_list_reordered_but_multiset_preserved():
    a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    b = a[:]
    assert kth_smallest(b, 4) == 3
    assert sorted(b) == sorted(a)
    c = a[:]
    assert select_random(c, 7) == 5
    assert sorted(c) == sorted(a)


def test_deterministic_counts_are_exact_and_stable():
    # The pivot rule is fully deterministic, so these counts are
    # the same on every machine and every run. Measured once.
    s = {"comparisons": 0, "swaps": 0}
    kth_smallest(list(range(1, 65)), 32, stats=s)
    assert s == {"comparisons": 351, "swaps": 150}, s
    s = {"comparisons": 0, "swaps": 0}
    kth_smallest(list(range(1, 101)), 50, stats=s)
    assert s == {"comparisons": 613, "swaps": 269}, s


def test_deterministic_has_no_input_order_worst_case():
    # The whole point of the median of medians: sorted, reverse
    # and random input must all cost the same order, within a
    # small constant, unlike the last-pivot quicksort from
    # Assignment 5 whose sorted-input cost was quadratic.
    n = 512
    rng = random.Random(7)
    data = {
        "sorted": sorted(rng.randint(0, 10000) for _ in range(n)),
        "reverse": sorted((rng.randint(0, 10000) for _ in range(n)),
                          reverse=True),
        "random": [rng.randint(0, 10000) for _ in range(n)],
    }
    counts = {}
    for kind, arr in data.items():
        s = {"comparisons": 0, "swaps": 0}
        kth_smallest(arr, n // 2 + 1, stats=s)
        counts[kind] = s["comparisons"]
    lo, hi = min(counts.values()), max(counts.values())
    assert hi < 2.0 * lo, counts
    for c in counts.values():
        assert c < n * 20, counts  # linear, far from n^2


def test_all_equal_stays_linear():
    s = {"comparisons": 0, "swaps": 0}
    kth_smallest([7] * 1000, 500, stats=s)
    assert s["comparisons"] < 5 * 1000, s  # O(n), not O(n^2)


def test_randomized_linear_band_on_random_input():
    n = 2048
    a = [random.Random(777).randint(0, 10**6) for _ in range(n)]
    s = {"comparisons": 0, "swaps": 0}
    select_random(a, n // 2 + 1, stats=s)
    # expected comparisons are at most 4n for any k; allow headroom
    # for the single-run variance
    assert n < s["comparisons"] < 6 * n, s


def test_median_of_medians_stays_on_a_shallow_stack():
    # The recursion is only as deep as the median-of-medians tree,
    # so 10k sorted elements run under a 300-frame limit.
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(300)
    try:
        assert kth_smallest(list(range(10000)), 5000) == 4999
    finally:
        sys.setrecursionlimit(old)


def test_randomized_driver_is_iterative():
    # The randomized variant is a loop on the surviving range, so
    # even a pathological pivot sequence cannot grow the stack.
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(300)
    try:
        random.seed(3)
        assert select_random(list(range(10000)), 5000) == 4999
    finally:
        sys.setrecursionlimit(old)


def test_dynamic_array_matches_a_list():
    da = DynamicArray()
    ref = []
    for i in range(100):
        v = i * 3
        da.append(v)
        ref.append(v)
        assert list(da) == ref
    da.insert(0, -1)
    ref.insert(0, -1)
    da.insert(50, -2)
    ref.insert(50, -2)
    da.delete(25)
    ref.pop(25)
    assert list(da) == ref
    da[10] = 999
    ref[10] = 999
    assert list(da) == ref
    for bad in (-len(ref) - 1, len(ref)):
        try:
            da[bad]
            assert False, "bounds must raise"
        except IndexError:
            pass


def test_dynamic_array_amortization():
    n = 4096
    da = DynamicArray()
    for i in range(n):
        da.append(i)
    assert list(da) == list(range(n))
    # geometric growth copies at most ~2n elements in total;
    # the measurement lands just under n
    assert da.copies < 2 * n, da.copies
    assert da.resizes <= 12, da.resizes


def test_matrix_operations():
    m = Matrix(2, 3, fill=0)
    m.set(0, 1, 7)
    assert m.get(0, 1) == 7
    m.insert_row(1, [9, 9, 9])
    assert m.rows == 3 and m.cols == 3
    m.insert_col(0, [1, 2, 3])
    assert m.rows == 3 and m.cols == 4
    assert [list(r) for r in m] == [[1, 0, 7, 0],
                                    [2, 9, 9, 9],
                                    [3, 0, 0, 0]]
    assert m.delete_col(0) == [1, 2, 3]
    assert m.cols == 3
    assert m.delete_row(1) == [9, 9, 9]
    assert m.rows == 2
    try:
        m.get(0, 3)
        assert False, "bounds must raise"
    except IndexError:
        pass
    try:
        m.insert_row(0, [1, 2])
        assert False, "wrong-width row must raise"
    except ValueError:
        pass


def test_stack_lifo():
    st = ArrayStack()
    assert st.is_empty()
    for x in (1, 2, 3):
        st.push(x)
    assert st.peek() == 3
    assert st.pop() == 3
    assert st.pop() == 2
    assert len(st) == 1
    st.pop()
    for op in (st.pop, st.peek):
        try:
            op()
            assert False, "empty stack must raise"
        except IndexError:
            pass


def test_queue_fifo():
    q = ArrayQueue()
    assert q.is_empty()
    for x in range(1000):
        q.enqueue(x)
    for x in range(1000):
        assert q.dequeue() == x  # the compaction must not reorder
    assert q.is_empty()
    try:
        q.dequeue()
        assert False, "empty queue must raise"
    except IndexError:
        pass
    q.enqueue(5)
    q.enqueue(6)
    assert q.peek() == 5 and len(q) == 2


def test_linked_list_operations():
    ll = LinkedList()
    for x in (2, 4, 6):
        ll.insert_head(x)
    ll.insert_tail(8)
    ll.insert_at(1, 3)
    assert list(ll) == [6, 3, 4, 2, 8]
    assert ll.get(2) == 4
    assert ll.delete_head() == 6
    assert ll.delete_tail() == 8
    assert ll.delete_at(1) == 4
    assert list(ll) == [3, 2]
    assert ll.delete_value(2) is True
    assert ll.delete_value(99) is False
    assert list(ll) == [3]
    assert ll.find(3) == 0 and ll.find(9) == -1
    assert 3 in ll and 9 not in ll
    assert len(ll) == 1
    ll.delete_head()
    for op in (ll.delete_head, ll.delete_tail):
        try:
            op()
            assert False, "empty list must raise"
        except IndexError:
            pass
    try:
        ll.delete_at(0)
        assert False, "empty list must raise"
    except IndexError:
        pass
    try:
        ll.insert_at(5, 0)
        assert False, "bounds must raise"
    except IndexError:
        pass


def test_rooted_tree():
    t = RootedTree("root")
    t.add_child("root", "a")
    t.add_child("root", "b")
    t.add_child("a", "a1")
    t.add_child("a", "a2")
    # add_child makes the new node the eldest, so root's children
    # are b then a, and a's children are a2 then a1
    assert t.preorder() == ["root", "b", "a", "a2", "a1"]
    assert t.size() == 5
    assert t.depth("a1") == 2
    assert t.depth("root") == 0
    assert t.find("b") and not t.find("z")
    try:
        t.add_child("z", "x")
        assert False, "missing parent must raise"
    except KeyError:
        pass
    try:
        t.depth("z")
        assert False, "missing node must raise"
    except KeyError:
        pass


def main():
    tests = [name for name in globals() if name.startswith("test_")]
    failed = 0
    for name in sorted(tests):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
