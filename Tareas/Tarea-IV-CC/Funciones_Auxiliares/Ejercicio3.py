import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular
from Ejercicio2 import*  

def create_vandermonde(x, p):
    """
    Computes the design matrix for polynomial regression.
    For a given vector x and degree p-1, returns matrix with columns:
    [1, x, x², x³, ..., x^(p-1)]
    """
    n = len(x)
    A = np.ones((n, p))  
    for j in range(1, p):
        A[:, j] = x ** j  
    return A

def generate_data(n):
    """
    Creates x values equally spaced between 4π/n and 4π,
    then computes y = sin(x) + random noise.
    """
    np.random.seed(12+1)  
    x = 4 * np.pi * np.arange(1, n+1) / n  # Create x values
    y_true = np.sin(x)  # True underlying function
    epsilon = np.random.normal(0, 0.11, n)  # Add some random noise
    y = y_true + epsilon  
    return x, y, y_true


if __name__ == "__main__":
# Experiment configuration - we qill test different scenarios
    cases = [2, 4, 6, 100]  
    ns = [100, 1000, 10000]  

# Storage for our results and timing measurements
    results = {}
    times_custom = np.zeros((len(cases), len(ns)))  # For our custom QR times
    times_scipy = np.zeros((len(cases), len(ns)))   # For Scipy's optimized QR times

    print("=== POLYNOMIAL REGRESSION EXPERIMENT WITH QR DECOMPOSITION ===\n")
        
# Test all combinations of polynomial degree and sample size
    for i, p in enumerate(cases):
        for j, n in enumerate(ns):
    
            # Generate data
            x, y, y_true = generate_data(n)
        
            # Create the design matrix (Vandermonde matrix)
            A = create_vandermonde(x, p)
        
            # Time our  QR implementation
            start_time = time.time()
            beta_custom, residuals_custom = least_squares_qr(A, y)
            times_custom[i, j] = time.time() - start_time
        
            # Time Scipy's optimized QR implementation for comparison
            start_time = time.time()
            Q_scipy, R_scipy = scipy_qr(A, mode='economic')
            qtb_scipy = Q_scipy.T @ y
            beta_scipy = solve_triangular(R_scipy, qtb_scipy, lower=False)
            times_scipy[i, j] = time.time() - start_time
        
            # Store all results for later analysis
            results[(p, n)] = {
                'x': x, 'y': y, 'y_true': y_true,
                'beta_custom': beta_custom,
                'beta_scipy': beta_scipy,
                'time_custom': times_custom[i, j],
                'time_scipy': times_scipy[i, j]
                }
        

    # Create visualization of the polynomial fits
    fig, axes = plt.subplots(len(cases), len(ns), figsize=(20, 20))
    fig.suptitle('Polynomial Regression ', fontsize=16)

    for i, p in enumerate(cases):
        for j, n in enumerate(ns):
            data = results[(p, n)]
            x, y, y_true = data['x'], data['y'], data['y_true']
            beta = data['beta_custom']
        
            # Create smooth curve 
            x_dense = np.linspace(x.min(), x.max(), n)
            A_dense = create_vandermonde(x_dense, p)
            y_pred_dense = A_dense @ beta
        
            ax = axes[i, j] 
            # Show the noisy data points
            ax.scatter(x, y, alpha=0.6, s=2, label=' data', color='gray')
            # Show our polynomial fit
            ax.plot(x_dense, y_pred_dense, 'r--', linewidth=2, label=f'Degree {p-1} fit')
            # Show the true underlying function
            ax.plot(x_dense, np.sin(x_dense), 'b--', linewidth=2, label=' sin(x)')
            
            ax.set_title(f'Degree: {p-1}, Samples: {n}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()


    # Repeat timing with more precision
    for i, p in enumerate(cases):
        for j, n in enumerate(ns):
            
            # Generate data
            x, y, y_true = generate_data(n)
            A = create_vandermonde(x, p)
        
            # Measure our custom QR implementation (repeat 5 times for accuracy)
            custom_times = []
            for _ in range(5):
                start_time = time.perf_counter_ns()  # High-resolution timer
                beta_custom = least_squares_qr(A, y)
                end_time = time.perf_counter_ns()
                custom_times.append((end_time - start_time) / 1e6)  # Convert to milliseconds
        
            times_custom[i, j] = np.median(custom_times)  # Use median to avoid outliers
        
            # Measure Scipy's QR implementation (repeat 5 times for accuracy)
            scipy_times = []
            for _ in range(5):
                start_time = time.perf_counter_ns()
                Q_scipy, R_scipy = scipy_qr(A, mode='economic')
                qtb_scipy = Q_scipy.T @ y
                beta_scipy = solve_triangular(R_scipy, qtb_scipy, lower=False)
                end_time = time.perf_counter_ns()
                scipy_times.append((end_time - start_time) / 1e6)  # Convert to milliseconds
        
            times_scipy[i, j] = np.median(scipy_times)

    # Create timing comparison plots
    plt.figure(figsize=(10, 4))

    # Plot 1: How time changes with polynomial degree
    plt.subplot(1, 2, 1)
    markers = ['o', 's', '^', 'D']
    colors = ['blue', 'red', 'green', 'purple']

    for j, n in enumerate(ns):
        plt.plot(cases, times_custom[:, j], marker=markers[j], 
                 linestyle='-', linewidth=2, markersize=8,
                 label=f'n={n} (Our method)', color=colors[j])
        plt.plot(cases, times_scipy[:, j], marker=markers[j], 
                 linestyle='--', linewidth=2, markersize=8,
                 label=f'n={n} (Scipy)', color=colors[j], alpha=0.7)
        
        plt.xlabel('Polynomial Degree', fontsize=12)
        plt.ylabel('Execution Time (ms)', fontsize=12)
        plt.title('Time vs Polynomial Complexity', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # Use logarithmic scale to see patterns better
        plt.yscale('log')

    # Plot 2: How time changes with sample size
    plt.subplot(1, 2, 2)
    for i, p in enumerate(cases):
        plt.plot(ns, times_custom[i, :], marker=markers[i], 
                 linestyle='-', linewidth=2, markersize=8,
                 label=f'Degree {p-1} (Our method)', color=colors[i])
        plt.plot(ns, times_scipy[i, :], marker=markers[i], 
                 linestyle='--', linewidth=2, markersize=8,
                 label=f'Degree {p-1} (Scipy)', color=colors[i], alpha=0.7)

    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Time vs Dataset Size', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('timing_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

