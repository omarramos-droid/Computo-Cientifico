import numpy as  np
from Ejercicio1 import modified_gram_schmidt
from Tarea_1_ex1 import backward_substitution
def least_squares_qr(A, b):
    """
    Solves the least squares problem for the linear model:
    y = Xβ + ε
    
    Where:
        X is a full-rank matrix of size m × n (m > n)
        ε is a vector of random errors of size m
        y is a vector of observations of size m
        
    This function uses the QR factorization from Exercise 1 to compute
    the least squares estimator.
    
    Parameters:
    -----------
    A :  Design matrix (must have full column rank, m > n)
    b :  Vector of observations
    
    Returns:
    --------
    beta :  Least squares estimator of β and numy array
    residuals : numpy array of shape (m,)
        Residual vector ε = y - Xβ
    """
    # Step 1: QR decomposition
    Q, R = modified_gram_schmidt(A)
    
    # Step 2: Compute Q^T b
    m, n = A.shape
    qtb = np.dot(Q.T, b)
    
    # Step 3: Solve R x = Q^T b using backward substitution
    beta = backward_substitution(R, qtb)
    
    # Step 4: Compute residuals
    residuals = b - np.dot(A, beta)
    
    return beta, residuals
def least_squares_qr_compact(A, b):
    """
    Using numpy functions
    """
    Q, R = modified_gram_schmidt(A)
    qtb = np.dot(Q.T, b)
    x = np.linalg.solve(R, qtb)  # Solves R x = Q^T b
    residuals = b - np.dot(A, x)
    
    return x

if __name__ == "__main__":
    print("=== LEAST SQUARES ESTIMATOR BY QR ===\n")
    
    # 
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([6,7,8,9,10])
    
    # Matriz de diseño A para y = β₀ + β₁x
    A = np.column_stack([np.ones(len(x_data)), x_data])
    print("Desing matrix A:")
    print(A)
    print("\nVector of observations b:")
    print(y_data)
    
    # Computing the least squares estimator
    beta, residuals = least_squares_qr(A, y_data)
    print(f"hat β" )
    print(f"β₀  = {beta[0]:.4f}")
    print(f"β₁    = {beta[1]:.4f}")
    
    print(f"\nThen : y = {beta[0]:.4f} + {beta[1]:.4f}x")
    
    beta_numpy = np.linalg.lstsq(A, y_data, rcond=None)[0]
    print(f"Verifying with numpy.linalg.lstsq:")
    print(f"β₀ = {beta_numpy[0]:.4f}, β₁ = {beta_numpy[1]:.4f}")
    
 

