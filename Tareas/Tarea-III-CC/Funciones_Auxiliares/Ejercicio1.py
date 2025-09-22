import numpy as np
import math


def normvec(v):
    """
    This function computes the Euclidean norm of a vector v.
    
    Parameters
    ----------
    v : numpy array
        A numerical vector of size n

    Returns
    -------
        The Euclidean norm ||v||
    """
    norm = math.sqrt(np.sum(v**2))
    return norm

def modified_gram_schmidt(A):
    """
    Computes the reduced QR factorization of a matrix using the 
    Modified Gram-Schmidt algorithm given by Trefethen 
    for matrices with full column rank.
    
    Parameters
    ----------
    A : numpy array of shape (m, n)
        Input matrix with m >= n and full column rank (rank = n)
        
    Returns
    -------
    Q : numpy array of shape (m, n)
        Matrix with orthonormal columns (Q^T Q = I)
    R : numpy array of shape (n, n)  
        Upper triangular matrix such that A = Q R

    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    V = A.copy().astype(float)
    
    for i in range(n):
        # r_ii = ||v_i||
        R[i, i] = normvec(V[:, i])        
        Q[:, i] = V[:, i] / R[i, i]
        
        for j in range(i + 1, n):
            # r_ij = q_i^* v_j   ⟨q_i, v_j⟩ 
            R[i, j] = np.dot(Q[:, i].conjugate(), V[:, j])            
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    
    return Q, R

#Example
if __name__ == "__main__":
    # Matrix
    A = np.array([[1, 1, 2],
                  [1, 2, 2],
                  [1, 3, 2],
                  [1, 4, 1]], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    # Using Gram-Schmidt 
    Q, R = modified_gram_schmidt(A)
    
    print("Matrix Q :")
    print(Q)
    print()
    
    print("Matrix R:")
    print(R)
    print()
    
    # Verifying the decomposition
    print("Q^T Q  most be the Indentiy:")
    print(Q.T @ Q)
    print()

    print("Q R (most be A):")
    print(Q @ R)
    B = np.array([[1, 1, 2],
                  [1, 2, 2],
                  [1, 3, 2],
                  [1, 4, 2]], dtype=float)
    
    print("Matrix B:")
    print(B)
    
    # Using Gram-Schmidt 
    Q, R = modified_gram_schmidt(B)
    
    print("Matrix Q :")
    print(Q)
    print()
    
    print("Matrix R:")
    print(R)
    print()
    
    # Verifying the decomposition
    print("Q^T Q  most be the Indentiy:")
    print(Q.T @ Q)
    print()

    print("Q R (most be B):")
    print(Q @ R)