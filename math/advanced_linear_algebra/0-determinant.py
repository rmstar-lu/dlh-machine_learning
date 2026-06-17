#!/usr/bin/env python3
""" 0-determinant module: Calculate the determinant of a matrix. """


def determinant(matrix):
    """ Calculate the determinant of a matrix. """

    if matrix == [[]]:
        return 1
    if (matrix == []
        or not (isinstance(matrix, list)
            and all(isinstance(row, list) for row in matrix))):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if any(len(row) != n
           or any(type(x) not in {int, float} for x in row)
           for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    i = 0
    return sum(((-1) ** (i + j)) * matrix[i][j] * determinant(
                [[matrix[mi][mj] for mj in range(n) if mj != j]
                    for mi in range(n) if mi != i]
           ) for j in range(n))
