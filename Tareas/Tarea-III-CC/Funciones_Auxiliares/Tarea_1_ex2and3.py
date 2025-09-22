# -*- coding: utf-8 -*-
"""
@author: omar.ramos
"""
import numpy as np
from Tarea_1_ex1 import backward_substitution, forward_substitution

def gaussian_with_pivoting(A):
    """
    Given the Algorithm 21.1. Gaussian Elimination with Partial Pivoting, this function computes
    the adequade pivoting in order to avoid problems with the factorization LU, the function needs the following:
    A: square matrix (m x m)
    And  it returns:
    P: permutation matrix
    L: lower triangular matrix with 1's on diagonal
    U: upper triangular matrix
    """
    m = A.shape[0]  # The dimension of the matrix, as is square both are equal
    U = A.copy()    # Copy of A,and will become the matrix U (Upper triangular)
    L = np.eye(m)  # np.eye gives us a squared matrix m*m, by default with 1´s on the diagonal (the identity)
    P = np.eye(m)
    
    for k in range(m - 1):
        # Select i ≥ k to maximize |u_ik|
        """
        Given  the colum k, We look for the maximum absolute value in the k-th column, 
        but only from row k to the bottom of the column
        
        U[k:, k] gives us the k-th column from row k to the end.
        np.argmax(np.abs(U[k:, k])) gives the relative index within this subarray.
        
        We add k to get the  row index because is the position in the original matrix A, no in the submatrix 
        """
        
        i_max = k + np.argmax(np.abs(U[k:, k])) #
        
        # Interchange rows in U
        U[[k, i_max], k:] = U[[i_max, k], k:]
        
        """
        For L, we only need to interchange the multipliers that have been already computed 
        When k=0, L  still being the identity matrix  so there's nothing to interchange.
        For k>0, we interchange the elements in columns 0 to k-1
        """
        if k > 0:
            L[[k, i_max], :k] = L[[i_max, k], :k]
        
        # Interchange rows in P
        P[[k, i_max], :] = P[[i_max, k], :]
        #Factorizacion LU
        for j in range(k + 1, m):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] = U[j, k:] - L[j, k] * U[k, k:]
    
    return P, L, U

def solve_system_with_pivoting_LUP(A, b):
    """
    This function solves the system Ax = b using Gaussian elimination with partial pivoting
    by imported forward and backward substitution functions
    """
    # Computes LU decomposition with pivoting
    P, L, U = gaussian_with_pivoting(A)
    
    # Apply permutation to b: Pb
    Pb = P @ b
    
    # Solve Ly = Pb using forward substitution
    y = forward_substitution(L, Pb)
    
    # Check if system can be solved 
    if y is None:
        return None, P, L, U
    
    # Solve Ux = y using backward substitution
    x = backward_substitution(U, y)
    
    if x is None:
        return None, P, L, U
    
    return x, P, L, U

if __name__ == "__main__":
    # Simulating 25, v.a U(0,1)
    Values = np.random.uniform(0, 1, 25)
    Values = Values.reshape(5, 5)  # Convert the vector into a matrix 5*5
    M_original = Values.copy()     # Copy in order to no calculate twice the matrix random
    
    # Realizar la descomposición LUP
    P, L, U = gaussian_with_pivoting(Values)
    
    print("\n" + "="*50)
    print("Excercies 3, random uniform matrix:")
    print("M =")
    print(M_original)
    print("\nP =")
    print(P)
    print("\nL =")
    print(L)
    print("\nU =")
    print(U)
    print("\nPM =")
    print(P @ M_original)  # Usar la matriz original guardada
    print("LU =")
    print(L @ U)
    print("\n" + "="*50)
    A = np.array([
    [1,0,0,0,1],
    [-1,1,0,0,1],
    [-1,-1,1,0,1],
    [-1,-1,-1,1,1],
    [-1,-1,-1,-1,1]
    ])
    P, L, U = gaussian_with_pivoting(A)
    
    print("\n" + "="*50)
    print("Excercies 3, matrix given A:")
    print("A =")
    print(A)
    print("\nP =")
    print(P)
    print("\nL =")
    print(L)
    print("\nU =")
    print(U)
    print("\nPM =")
    print(P @ A)  # Usar la matriz original guardada
    print("LU =")
    print(L @ U)
    print("\n" + "="*50)
