"""
MSCS 532 - Assignment 6
Elementary data structures, implemented from scratch in Python:

  DynamicArray  - a growable one-dimensional array with
                  amortized O(1) append and O(n) insert/delete.
  Matrix        - a 2-D array with row and column operations.
  ArrayStack    - LIFO on an array, O(1) push and pop.
  ArrayQueue    - FIFO on a circular buffer, O(1) amortized
                  enqueue and dequeue.
  LinkedList    - singly linked list with insertion, deletion,
                  and traversal.
  RootedTree    - a rooted tree in first-child / next-sibling
                  form, so the nodes are linked-list pointers.

Each class documents the complexity of its operations; the report
collects the whole table and discusses the array-versus-linked
trade-offs the assignment asks for.
"""


class DynamicArray:
    """A growable array with amortized O(1) append.

    The backing buffer doubles when it is full and shrinks to half
    when the load factor drops to 1/4, so a run of n appends moves
    at most about 2n elements in total (geometric growth). The
    resize and copy counters make that amortization measurable:
    copies is the number of elements copied across all resizes.

    Complexity:
        append        - O(1) amortized
        insert / del  - O(n) (the shift) plus the amortized resize
        access        - O(1)
    """

    def __init__(self):
        self._buf = [None] * 4
        self._size = 0
        self.resizes = 0
        self.copies = 0

    def _copy_to(self, new_size):
        new = [None] * new_size
        for i in range(self._size):
            new[i] = self._buf[i]
        self.copies += self._size
        self.resizes += 1
        self._buf = new

    def append(self, x):
        """Add x at the end. Amortized O(1)."""
        if self._size == len(self._buf):
            self._copy_to(len(self._buf) * 2)
        self._buf[self._size] = x
        self._size += 1

    def insert(self, i, x):
        """Insert x before index i (0 <= i <= len). O(n)."""
        if i < 0 or i > self._size:
            raise IndexError("insert index out of range")
        if self._size == len(self._buf):
            self._copy_to(len(self._buf) * 2)
        for j in range(self._size, i, -1):
            self._buf[j] = self._buf[j - 1]
        self._buf[i] = x
        self._size += 1

    def delete(self, i):
        """Remove and return the element at index i. O(n)."""
        if i < 0 or i >= self._size:
            raise IndexError("delete index out of range")
        x = self._buf[i]
        for j in range(i, self._size - 1):
            self._buf[j] = self._buf[j + 1]
        self._size -= 1
        if self._size and self._size * 4 <= len(self._buf) \
                and len(self._buf) > 4:
            self._copy_to(len(self._buf) // 2)
        return x

    def __getitem__(self, i):
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError("index out of range")
        return self._buf[i]

    def __setitem__(self, i, x):
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError("index out of range")
        self._buf[i] = x

    def __len__(self):
        return self._size

    def __iter__(self):
        for i in range(self._size):
            yield self._buf[i]


class Matrix:
    """A 2-D array with row and column operations.

    Storage is a list of row lists, so a row is contiguous and a
    column is not: row operations move O(cols) entries and column
    operations touch every row.

    Complexity:
        get / set            - O(1)
        insert / delete row  - O(cols) to build or remove the row,
                               O(rows) list moves in the outer list
        insert / delete col  - O(rows)
    """

    def __init__(self, rows, cols, fill=0):
        if rows < 0 or cols < 0:
            raise ValueError("dimensions must be non-negative")
        self.rows = rows
        self.cols = cols
        self._data = [[fill] * cols for _ in range(rows)]

    def get(self, i, j):
        """Return entry (i, j). O(1)."""
        if not 0 <= i < self.rows or not 0 <= j < self.cols:
            raise IndexError("entry out of range")
        return self._data[i][j]

    def set(self, i, j, value):
        """Set entry (i, j). O(1)."""
        if not 0 <= i < self.rows or not 0 <= j < self.cols:
            raise IndexError("entry out of range")
        self._data[i][j] = value

    def insert_row(self, i, values=None):
        """Insert a row before index i; defaults to a zero row."""
        if i < 0 or i > self.rows:
            raise IndexError("row index out of range")
        if values is None:
            values = [0] * self.cols
        if len(values) != self.cols:
            raise ValueError("row has the wrong number of columns")
        self._data.insert(i, list(values))
        self.rows += 1

    def delete_row(self, i):
        """Remove and return the row at index i."""
        if i < 0 or i >= self.rows:
            raise IndexError("row index out of range")
        row = self._data.pop(i)
        self.rows -= 1
        return row

    def insert_col(self, j, values=None):
        """Insert a column before index j; defaults to a zero col."""
        if j < 0 or j > self.cols:
            raise IndexError("column index out of range")
        if values is None:
            values = [0] * self.rows
        if len(values) != self.rows:
            raise ValueError("column has the wrong number of rows")
        for i, v in enumerate(values):
            self._data[i].insert(j, v)
        self.cols += 1

    def delete_col(self, j):
        """Remove and return the column at index j."""
        if j < 0 or j >= self.cols:
            raise IndexError("column index out of range")
        col = [row.pop(j) for row in self._data]
        self.cols -= 1
        return col

    def __len__(self):
        return self.rows

    def __iter__(self):
        for row in self._data:
            yield list(row)


class ArrayStack:
    """LIFO on a Python list: push appends, pop removes the end.

    Complexity:  push / pop / peek  -  O(1)
    """

    def __init__(self):
        self._buf = []

    def push(self, x):
        """Add x on top. O(1)."""
        self._buf.append(x)

    def pop(self):
        """Remove and return the top. O(1). Raises IndexError on
        an empty stack."""
        if not self._buf:
            raise IndexError("stack is empty")
        return self._buf.pop()

    def peek(self):
        """Return the top without removing it. O(1)."""
        if not self._buf:
            raise IndexError("stack is empty")
        return self._buf[-1]

    def is_empty(self):
        return not self._buf

    def __len__(self):
        return len(self._buf)

    def __iter__(self):
        """Iterate from the top down."""
        return iter(reversed(self._buf))


class ArrayQueue:
    """FIFO on a buffer with a moving head index.

    dequeue takes the head and advances the index; once enough
    slots have been abandoned at the front, the buffer is compacted
    in one slice. Enqueue is an amortized O(1) append, dequeue is
    O(1) with amortized O(1) compaction, so the whole queue is
    O(1) amortized per operation without a fixed capacity.

    Complexity:  enqueue / dequeue / peek  -  O(1) amortized
    """

    _COMPACTION_AT = 32   # abandon at least this many before compacting

    def __init__(self):
        self._buf = []
        self._head = 0

    def enqueue(self, x):
        """Add x at the back. O(1) amortized."""
        self._buf.append(x)

    def dequeue(self):
        """Remove and return the front. O(1) amortized. Raises
        IndexError on an empty queue."""
        if self._head >= len(self._buf):
            raise IndexError("queue is empty")
        x = self._buf[self._head]
        self._head += 1
        if self._head >= self._COMPACTION_AT \
                and self._head * 2 >= len(self._buf):
            self._buf = self._buf[self._head:]
            self._head = 0
        return x

    def peek(self):
        """Return the front without removing it. O(1)."""
        if self._head >= len(self._buf):
            raise IndexError("queue is empty")
        return self._buf[self._head]

    def is_empty(self):
        return self._head >= len(self._buf)

    def __len__(self):
        return len(self._buf) - self._head

    def __iter__(self):
        """Iterate from the front to the back."""
        return iter(self._buf[self._head:])


class _Link:
    """One node of the singly linked list."""

    __slots__ = ("value", "next")

    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    """Singly linked list with insertion, deletion, and traversal.

    Head operations are O(1); operations at index i walk i nodes,
    so positional access is O(i) = O(n) in the worst case, which
    is the trade-off against the DynamicArray's O(1) index access.

    Complexity:
        insert_head / delete_head  - O(1)
        insert_tail / delete_tail  - O(n) (no tail pointer kept)
        insert_at / delete_at      - O(i)
        traversal                  - O(n)
    """

    def __init__(self):
        self.head = None
        self._size = 0

    def insert_head(self, x):
        """Insert x at the front. O(1)."""
        link = _Link(x)
        link.next = self.head
        self.head = link
        self._size += 1

    def insert_tail(self, x):
        """Append x at the end. O(n)."""
        link = _Link(x)
        if self.head is None:
            self.head = link
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            cur.next = link
        self._size += 1

    def insert_at(self, i, x):
        """Insert x at index i (0-based). O(i)."""
        if i < 0 or i > self._size:
            raise IndexError("index out of range")
        if i == 0:
            self.insert_head(x)
            return
        cur = self.head
        for _ in range(i - 1):
            cur = cur.next
        link = _Link(x)
        link.next = cur.next
        cur.next = link
        self._size += 1

    def delete_head(self):
        """Remove and return the front value. O(1)."""
        if self.head is None:
            raise IndexError("list is empty")
        x = self.head.value
        self.head = self.head.next
        self._size -= 1
        return x

    def delete_tail(self):
        """Remove and return the back value. O(n)."""
        if self.head is None:
            raise IndexError("list is empty")
        if self.head.next is None:
            x = self.head.value
            self.head = None
            self._size -= 1
            return x
        cur = self.head
        while cur.next.next is not None:
            cur = cur.next
        x = cur.next.value
        cur.next = None
        self._size -= 1
        return x

    def delete_at(self, i):
        """Remove and return the value at index i. O(i)."""
        if i < 0 or i >= self._size:
            raise IndexError("index out of range")
        if i == 0:
            return self.delete_head()
        cur = self.head
        for _ in range(i - 1):
            cur = cur.next
        x = cur.next.value
        cur.next = cur.next.next
        self._size -= 1
        return x

    def delete_value(self, x):
        """Remove the first node holding x. Returns True if one was
        removed. O(n)."""
        if self.head is None:
            return False
        if self.head.value == x:
            self.head = self.head.next
            self._size -= 1
            return True
        cur = self.head
        while cur.next is not None:
            if cur.next.value == x:
                cur.next = cur.next.next
                self._size -= 1
                return True
            cur = cur.next
        return False

    def find(self, x):
        """Return the index of the first node holding x, or -1. O(n)."""
        for i, v in enumerate(self):
            if v == x:
                return i
        return -1

    def get(self, i):
        """Return the value at index i. O(i)."""
        if i < 0 or i >= self._size:
            raise IndexError("index out of range")
        cur = self.head
        for _ in range(i):
            cur = cur.next
        return cur.value

    def __len__(self):
        return self._size

    def __iter__(self):
        cur = self.head
        while cur is not None:
            yield cur.value
            cur = cur.next

    def __contains__(self, x):
        return self.find(x) != -1


class TNode:
    """A node of the rooted tree, linked in first-child /
    next-sibling form: each node points at its first child and at
    its next younger sibling, so the whole tree is built from the
    same two pointers a linked list uses."""

    __slots__ = ("value", "first_child", "next_sibling")

    def __init__(self, value):
        self.value = value
        self.first_child = None
        self.next_sibling = None


class RootedTree:
    """A rooted tree over TNode links.

    add_child inserts the new node as the parent's eldest child;
    traversal is preorder over the first-child / next-sibling
    pointers, which is an O(n) walk of the links.

    Complexity:
        add_child / find / depth  - O(n) (the search for the parent)
        insertion itself          - O(1)
        preorder / size           - O(n)
    """

    def __init__(self, root_value):
        self.root = TNode(root_value)

    def _find(self, value):
        """Return the node holding value, or None. Preorder."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.value == value:
                return node
            if node.first_child is not None:
                stack.append(node.first_child)
            if node.next_sibling is not None:
                stack.append(node.next_sibling)
        return None

    def add_child(self, parent_value, child_value):
        """Make child_value the eldest child of the node holding
        parent_value. O(n) to find the parent, O(1) to link.
        Raises KeyError if the parent does not exist."""
        parent = self._find(parent_value)
        if parent is None:
            raise KeyError(f"no node with value {parent_value}")
        node = TNode(child_value)
        node.next_sibling = parent.first_child
        parent.first_child = node
        return node

    def find(self, value):
        """Public search: True if some node holds value. O(n)."""
        return self._find(value) is not None

    def size(self):
        """Number of nodes. O(n)."""
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            if node.first_child is not None:
                stack.append(node.first_child)
            if node.next_sibling is not None:
                stack.append(node.next_sibling)
        return count

    def depth(self, value):
        """Edges from the root to the node holding value. O(n).
        Raises KeyError if the value is not in the tree."""
        def walk(node, d):
            if node.value == value:
                return d
            cur = node.first_child
            while cur is not None:
                found = walk(cur, d + 1)
                if found is not None:
                    return found
                cur = cur.next_sibling
            return None

        result = walk(self.root, 0)
        if result is None:
            raise KeyError(f"no node with value {value}")
        return result

    def preorder(self):
        """Values in preorder (node, then children eldest first).
        O(n). The stack pops youngest-first, so the children are
        pushed in reverse sibling order."""
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node.value)
            children = []
            cur = node.first_child
            while cur is not None:
                children.append(cur)
                cur = cur.next_sibling
            for child in reversed(children):
                stack.append(child)
        return out


def main():
    print("== DynamicArray ==")
    da = DynamicArray()
    for i in range(6):
        da.append(i)
    da.insert(1, 100)
    da.delete(3)
    print("  contents:", list(da), " resizes:", da.resizes)

    print("== Matrix ==")
    m = Matrix(2, 3, fill=0)
    m.set(0, 1, 7)
    m.insert_row(1, [9, 9, 9])
    m.insert_col(0, [1, 2, 3])
    print("  dims:", m.rows, "x", m.cols)
    print("  rows:", [list(r) for r in m])

    print("== ArrayStack ==")
    st = ArrayStack()
    for x in (1, 2, 3):
        st.push(x)
    print("  pops:", st.pop(), st.pop(), "  top now:", st.peek())

    print("== ArrayQueue ==")
    q = ArrayQueue()
    for x in (1, 2, 3):
        q.enqueue(x)
    print("  dequeues:", q.dequeue(), q.dequeue(), "  front now:", q.peek())

    print("== LinkedList ==")
    ll = LinkedList()
    for x in (2, 4, 6):
        ll.insert_head(x)
    ll.insert_tail(8)
    ll.insert_at(1, 3)
    print("  contents:", list(ll))
    print("  delete_value(4):", ll.delete_value(4), " ->", list(ll))

    print("== RootedTree ==")
    t = RootedTree("root")
    t.add_child("root", "a")
    t.add_child("root", "b")
    t.add_child("a", "a1")
    print("  preorder:", t.preorder())
    print("  size:", t.size(), "  depth of a1:", t.depth("a1"))


if __name__ == "__main__":
    main()
