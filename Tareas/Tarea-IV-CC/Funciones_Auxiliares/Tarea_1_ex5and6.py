# -*- coding: utf-8 -*-
"""
@author: omar.ramos
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from Tarea_1_ex2and3 import gaussian_with_pivoting



def cholesky_factorization(A):
    """
    This funtion computes the cholesky factorization of a matrix Hermitian postive definite, 
    and it returns
    R: Upper triangular matrix such that A = R* R 
    """
    m=A.shape[1]
    # Create a copy of A for the factorization, given that the matrix is hermitian, we need to use complex number
    R = A.copy().astype(complex)  
    # for k=0, to k=m-1 equivalent to for k=1 to k=m
    for k in range(m):
        # Check if the matrix is positive definite, we need the diagonal be positive
        if R[k, k] <= 0:
            print("The matrix isnt positive definite")
            return None
        # Update the remaining rows
        for j in range(k + 1, m):
            # R_{j,j:m} = R_{j,j:m} - R_{k,j:m} * conj(R_{kj}) / R_{kk}
            R[j, j:] = R[j, j:] - R[k, j:] * np.conj(R[k, j]) / R[k, k]
        
        # Here we need R_kk be positive 
        R[k, k:] = R[k, k:] / np.sqrt(R[k, k])
    #As we work with the matrix A, and we made the factorizacion on it, we only need the upper triangular matrix
    # the lower triangular is the matrix A, and we need it be 0
    """
    numpy.triu() is a function from NumPy 
    that returns a copy of a matrix or array with the elements below a 
    specified diagonal k, if k=0 it  returns the elements on  above the mian diagonal
    """
    R = np.triu(R)
    return R


if __name__ == "__main__":
    
    #Given the dimension the mmatrix, we will compute the time that each algorith takes 
    sizes = [30, 50, 100, 150, 200,500]
    cholesky_times = []
    lup_times = []
    numpy_cholesky_times = []

    print("Comparative Analysis: Cholesky  vs LUP vs NumPy Cholesky")
    print("="*60)

    for n in sizes:
        print(f"\nTesting with matrix size {n}x{n}")

        # Generate a matrix positive definited, M is a simetric M=A^t A 
        # and also is positive defined 
        Values = np.random.uniform(0, 1, n**2)
        A = Values.reshape(n, n)
        M = A @ A.T

        # Time cholesky hand made 
        start_time = time.time()
        L = cholesky_factorization(M)
        cholesky_time = time.time() - start_time
        cholesky_times.append(cholesky_time)

        # Time LUP
        start_time = time.time()
        P, L, U = gaussian_with_pivoting(M)
        lup_time = time.time() - start_time
        lup_times.append(lup_time)

        # Time cholesky numpy
        start_time = time.time()
        L_numpy = np.linalg.cholesky(M)
        numpy_time = time.time() - start_time
        numpy_cholesky_times.append(numpy_time)

# ===================================================
#Print results
# ===================================================

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cholesky_times, 'bo-', linewidth=2, markersize=8, label='Cholesky ')
    plt.plot(sizes, lup_times, 'ro-', linewidth=2, markersize=8, label='LUP')
    plt.plot(sizes, numpy_cholesky_times, 'go-', linewidth=2, markersize=8, label='NumPy Cholesky')

    plt.xlabel('Dimension of matrix (n x n)')
    plt.ylabel('Time of ejecution (seconds)')
    plt.title('Time Comparison: Cholesky  vs LUP vs NumPy')
    plt.legend()
    plt.grid(True, alpha=0.3)

