#!/usr/bin/env python3
""" 2-cofactor module: Calculate the cofactor of a matrix. """


def cofactor(matrix):
    """ Calculate the cofactor of a matrix. """

    def _det(matrix):
        """ Calculate the determinant of a matrix. """

        n = len(matrix)
        if n == 0:
            return 1
        if n == 1:
            return matrix[0][0]
        i = 0
        return sum(((-1) ** (i + j)) * matrix[i][j] * _det(
                    [[matrix[mi][mj] for mj in range(n) if mj != j]
                        for mi in range(n) if mi != i]
               ) for j in range(n))

    if (matrix == []
        or not (isinstance(matrix, list)
                and all(isinstance(row, list) for row in matrix))):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if (any(len(row) != n
            or any(type(x) not in {int, float} for x in row)
            for row in matrix)):
        raise ValueError("matrix must be a non-empty square matrix")
    return [[((-1) ** (i + j)) *
             _det([[matrix[mi][mj] for mj in range(n) if mj != j]
                   for mi in range(n) if mi != i])
            for j in range(n)] for i in range(n)]
