#!/usr/bin/env python3
""" 12-bracin_the_elements module. Do element-wise operations on matrices. """


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division.
    """

    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
