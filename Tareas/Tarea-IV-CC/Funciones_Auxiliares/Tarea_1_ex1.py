# -*- coding: utf-8 -*-
"""
@author: omar.ramos
"""

import numpy as np

"""
We are looking for solving a systems of equations Ax=b, where  A is a upper/lower triangular
matrix, if:
A is upper triangular matrix we compute a succesive substitution called backward substitution
A is lower triangular matrix we compute a succesive substitution called forward substitution
"""
        
def backward_substitution(U, b):
    """
This function  computes the solution of the system Ux=b, where U is a upper triangula matrix 
the  function  works with 2 parameters
U: the upper triangular matrix (n*n)
b: the vector of constant (n*1)
 The function return the solution of the system Ux=b, the vector x.
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):  # from n-1 down to 0
        suma = 0
        for j in range(i+1, n):
            suma += U[i, j] * x[j]

        if U[i, i] == 0:
            print("No unique solution, or there isn't exists")
            return None
        x[i] = (b[i] - suma) / U[i, i]

    return x


def forward_substitution(L, b):
    """
This function  computes the solution of the system Lx=b, where L is a lower triangula matrix 
the  function  works with 2 parameters:
L: the upper triangular matrix (n*n)
b: the vector of constant (n*1)
 The function return the solution of the system Lx=b, the vector x.
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n):  # from 0 up to n-1
        suma = 0
        for j in range(i):
            suma += L[i, j] * x[j]
        if L[i, i] == 0:
            print("No unique solution, or there isn´t exists")
            return None
        x[i] = (b[i] - suma) / L[i, i]

    return x

if __name__ == "__main__":
#Funcionamiento:
    U = np.array([
    [2., -1., 0.],
    [0.,  3., 4.],
    [0.,  0., 5.]
    ])
    b = np.array([1., 11., 10.])

    x = backward_substitution(U, b)
    print("Ejemplo 1 (U upper):")
    print("U =\n", U)
    print("b =", b)
    print("x  =", x)

    L = np.array([
    [0, 0, 0.],
    [1,  3., 0],
    [2,  2, 5.]
    ])
    b = np.array([1., 11., 10.])


    print("Ejemplo 1 (L lower):")
    x = forward_substitution(L, b)
    print("L =\n", L)
    print("b =", b)
    print("x  =", x)

