TUPLES: dict[str, list[tuple[int]]] = {
    # 4x6-tuple network by Yeh et al.
    "Yeh": [
        (0, 1, 2, 3, 4, 5),
        (4, 5, 6, 7, 8, 9),
        (0, 1, 2, 4, 5, 6),
        (4, 5, 6, 8, 9, 10),
    ],
    # 5x6-tuple network by Jaśkowski
    "Jaskowski": [
        (0, 1, 2, 3, 4, 5),
        (4, 5, 6, 7, 8, 9),
        (8, 9, 10, 11, 12, 13),
        (0, 1, 2, 4, 5, 6),
        (4, 5, 6, 8, 9, 10),
    ],
    # 8x6-tuple network by Matsuzaki
    "Matsuzaki": [
        (0, 1, 2, 4, 5, 6),
        (1, 2, 5, 6, 9, 13),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 5, 6, 7, 10),
        (0, 1, 2, 5, 9, 10),
        (0, 1, 5, 9, 13, 14),
        (0, 1, 5, 8, 9, 13),
        (0, 1, 2, 4, 6, 10),
    ],
    "4-6-mixed": [
        (0, 1, 2, 3, 4, 5),
        (4, 5, 6, 7, 8, 9),
        (8, 9, 10, 11, 12, 13),
        (0, 1, 2, 4, 5, 6),
        (4, 5, 6, 8, 9, 10),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 4, 5),
        (2, 3, 6, 7),
    ],
}  # type: ignore

# 0  1  2  3
# 4  5  6  7
# 8  9  10 11
# 12 13 14 15


# Transformation Functions ----------------------------------
def rotate90(index: int) -> int:
    """cw 90"""
    i, j = index // 4, index % 4
    return j * 4 + (3 - i)


def rotate180(index: int) -> int:
    """cw 180"""
    i, j = index // 4, index % 4
    return (3 - i) * 4 + (3 - j)


def rotate270(index: int) -> int:
    """cw 270"""
    i, j = index // 4, index % 4
    return (3 - j) * 4 + i


def mirror_horizontal(index: int) -> int:
    """horizontal mirror"""
    i, j = index // 4, index % 4
    return i * 4 + (3 - j)


def mirror_vertical(index: int) -> int:
    """vertical mirror"""
    i, j = index // 4, index % 4
    return (3 - i) * 4 + j


def mirror_diagonal(index: int) -> int:
    """main diagonal mirror"""
    i, j = index // 4, index % 4
    return j * 4 + i


def mirror_anti_diagonal(index: int) -> int:
    """anti-diagonal mirror"""
    i, j = index // 4, index % 4
    return (3 - j) * 4 + (3 - i)


TRANSFORMATIONS = [
    rotate90,
    rotate180,
    rotate270,
    mirror_horizontal,
    mirror_vertical,
    mirror_diagonal,
    mirror_anti_diagonal,
]


def get_tuples(name: str) -> list[tuple[int]]:
    """
    Get the n-tuple network for the given name.

    Parameters
    ----------
    name : str
        The name of the n-tuple network

    Returns
    -------
    list[tuple[int]]
        The n-tuple network
    """
    if name not in TUPLES:
        raise ValueError(f"Unknown n-tuple network: {name}")
    return TUPLES[name]


def get_symmetric_tuples(name: str) -> list[tuple[int]]:
    """
    Get all symmetrically sampled tuples for the given n-tuple network.

    Parameters
    ----------
    name : str
        The name of the n-tuple network

    Returns
    -------
    list[tuple[int]]
        The symmetrically sampled n-tuple network
    """
    tuples = get_tuples(name)
    symmetric_tuples = set()

    for t in tuples:
        # add original
        symmetric_tuples.add(t)

        # add all transformations
        for transform in TRANSFORMATIONS:
            transformed = tuple(transform(idx) for idx in t)
            symmetric_tuples.add(transformed)

    # return sorted list of tuples
    return sorted(symmetric_tuples, key=lambda x: x[0])


if __name__ == "__main__":
    # Example usage
    tuples = get_tuples("Yeh")
    print(len(tuples))
    print(tuples)
    print("Symmetric Tuples:")
    symmetric_tuples = get_symmetric_tuples("Yeh")
    print(len(symmetric_tuples))
    print(symmetric_tuples)
