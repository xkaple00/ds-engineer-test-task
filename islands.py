"""Count four-connected islands from a stream of binary grid rows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence


def count_islands(rows: Iterable[Sequence[int]], columns: int | None = None) -> int:
    """Return the number of components without retaining or changing the grid."""
    if columns is not None and columns < 0:
        raise ValueError("the column count cannot be negative")

    previous = []
    active_labels = 0
    islands = 0
    for row in rows:
        if columns is None:
            columns = len(row)
        if len(row) != columns:
            raise ValueError(f"expected {columns} columns, got {len(row)}")
        if not previous:
            previous = [0] * columns

        parents = list(range(active_labels + 1))
        sizes = [1] * len(parents)
        current = [0] * columns

        def root(label: int) -> int:
            while parents[label] != label:
                parents[label] = parents[parents[label]]
                label = parents[label]
            return label

        for column, cell in enumerate(row):
            if cell == 0:
                continue
            if cell != 1:
                raise ValueError(f"expected 0 or 1, got {cell!r}")
            above = previous[column]
            left = current[column - 1] if column else 0
            if left and above:
                first, second = root(left), root(above)
                if first != second:
                    if sizes[first] < sizes[second]:
                        first, second = second, first
                    parents[second] = first
                    sizes[first] += sizes[second]
                    islands -= 1
                current[column] = first
            elif left or above:
                current[column] = left or above
            else:
                current[column] = len(parents)
                parents.append(len(parents))
                sizes.append(1)
                islands += 1

        # Only components that reach the current row can touch later rows.
        compact = {}
        for column, label in enumerate(current):
            if label:
                representative = root(label)
                current[column] = compact.setdefault(representative, len(compact) + 1)
        previous, active_labels = current, len(compact)

    return islands


def read_grid(stream: Iterable[str]) -> tuple[int, Iterable[list[int]]]:
    """Read the dimensions and return a lazy iterator over the grid rows."""
    lines = iter(stream)
    header = next(lines, "").split()
    if len(header) != 2:
        raise ValueError("the first line must contain M and N")
    height, width = map(int, header)
    if height < 0 or width < 0:
        raise ValueError("grid dimensions cannot be negative")

    def rows() -> Iterable[list[int]]:
        for row_index in range(1, height + 1):
            line = next(lines, None)
            if line is None:
                raise ValueError(f"missing row {row_index} of {height}")
            yield list(map(int, line.split()))
        if any(line.strip() for line in lines):
            raise ValueError("extra data after the declared grid")

    return width, rows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        width, rows = read_grid(sys.stdin)
        print(count_islands(rows, width))
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
