import numpy as np
import sys
import os
from scipy.linalg import qr as scipy_qr
sys.path.append(os.path.join(os.path.dirname(__file__), 'Funciones_Auxiliares'))
from Ejercicio1 import modified_gram_schmidt

def qr_iteration_gram_schmidt(A, tol, max_iter, verbose=False):
    """
    Iteración QR con Gram–Schmidt modificado
    """
    A_k = A.copy().astype(float)
    eigen_history = []

    for k in range(max_iter):
        Q, R = modified_gram_schmidt(A_k)
        A_k = R @ Q
        eigen_history.append(np.diag(A_k).copy())

        if verbose:
            print(f"\n[Gram-Schmidt] Iteración {k}")
            print(A_k)

        off_diag = np.sum(np.abs(A_k - np.diag(np.diag(A_k))))
        if off_diag < tol:
            break

    eigenvalues = np.diag(A_k)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]

    return eigenvalues, eigen_history, k+1

def qr_iteration_scipy(A, tol, max_iter, verbose=False):
    """
    Iteración QR con factorización SciPy
    """
    A_k = A.copy().astype(float)
    eigen_history = []

    for k in range(max_iter):
        Q, R = scipy_qr(A_k)
        A_k = R @ Q
        eigen_history.append(np.diag(A_k).copy())

        if verbose:
            print(f"\n[SciPy QR] Iteración {k}")
            print(A_k)

        off_diag = np.sum(np.abs(A_k - np.diag(np.diag(A_k))))
        if off_diag < tol:
            break

    eigenvalues = np.diag(A_k)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]

    return eigenvalues, eigen_history, k+1


def comparar_metodos_qr(epsilon_values, tol, max_iter):
    """
    Comparar ambos métodos para diferentes valores de epsilon
    usando la norma relativa L2 como métrica de error.
    """
    resultados = {}
    
    for eps in epsilon_values:
        # Construir matriz A
        A = np.array([
            [8, 1, 0],
            [1, 4, eps],
            [0, eps, 1]
        ], dtype=float)
        
        # Referencia exacta
        eig_exacto = np.linalg.eigvals(A)
        eig_exacto = np.sort(eig_exacto)[::-1]  # Orden descendente
        
        # Calcular con ambos métodos
        eig_gs, hist_gs, iter_gs = qr_iteration_gram_schmidt(A, tol, max_iter)
        eig_sp, hist_sp, iter_sp = qr_iteration_scipy(A, tol, max_iter)
        
        # Ordenar aproximaciones igual que los exactos
        eig_gs = eig_gs[np.argsort(np.abs(eig_gs))[::-1]]
        eig_sp = eig_sp[np.argsort(np.abs(eig_sp))[::-1]]
        
        # Calcular errores NORMA RELATIVA
        error_rel_gs = np.linalg.norm(eig_gs - eig_exacto) / np.linalg.norm(eig_exacto)
        error_rel_sp = np.linalg.norm(eig_sp - eig_exacto) / np.linalg.norm(eig_exacto)
        
        resultados[eps] = {
            'exacto': eig_exacto,
            'gram_schmidt': eig_gs,
            'scipy_qr': eig_sp,
            'iter_gs': iter_gs,
            'iter_sp': iter_sp,
            'error_rel_gs': error_rel_gs,
            'error_rel_sp': error_rel_sp,
            'historia_gs': hist_gs,
            'historia_sp': hist_sp
        }
    
    return resultados

def mostrar_resumen(resultados, epsilon_values):
    """Muestra un resumen compacto de todos los resultados"""
    print("\n" + "="*120)
    print("RESUMEN COMPARATIVO - ITERACIÓN QR")
    print("="*120)
    print(f"{'ε':<8} {'Exacto':<25} {'Gram-Schmidt':<25} {'Scipy QR':<25}  {'Error Rel GS':<12} {'Error Rel Sp':<12} {'Iter GS':<8} {'Iter Sp':<8}")
    print("-"*120)
    
    for eps in epsilon_values:
        res = resultados[eps]
        exacto_str = "[" + ", ".join([f"{x:.4f}" for x in res['exacto']]) + "]"
        gs_str = "[" + ", ".join([f"{x:.4f}" for x in res['gram_schmidt']]) + "]"
        sp_str = "[" + ", ".join([f"{x:.4f}" for x in res['scipy_qr']]) + "]"
        
        print(f"{eps:<8.0e} {exacto_str:<25} {gs_str:<25} {sp_str:<25} "
              f"{res['error_rel_gs']:<12.2e} {res['error_rel_sp']:<12.2e} "
              f"{res['iter_gs']:<8} {res['iter_sp']:<8}")


if __name__ == "__main__":
    
    ##Primero se mostrara para un epsilon fijo y como se construyen las matrices.
    
    # Matriz de prueba con epsilon
    # eps = 1e-1
    # A = np.array([
    #     [8, 1, 0],
    #     [1, 4, eps],
    #     [0, eps, 1]
    # ], dtype=float)

    # TOL = 1e-5
    # MAX_ITER = 5  # Si se sube imprimira demasiado

    # # Ejecutar ambos métodos con impresión
    # eig_gs, hist_gs, it_gs = qr_iteration_gram_schmidt(A, TOL, MAX_ITER, verbose=True)
    # eig_sp, hist_sp, it_sp = qr_iteration_scipy(A, TOL, MAX_ITER, verbose=True)

    # print("\n=== Resultados finales ===")
    # print("Eigenvalores Gram-Schmidt:", eig_gs)
    # print("Eigenvalores SciPy QR    :", eig_sp)
    
    ## Aqui el resto del codgo
    # Valores de epsilon a probar
    epsilon_values = [1e0, 1e-1, 1e-2, 1e-4, 1e-5]
    
    # Parámetros de convergencia
    TOL = 1e-15       # Tolerancia para convergencia
    MAX_ITER = 1500      # Máximo número de iteraciones
    
    # Ejecutar comparación 
    resultados = comparar_metodos_qr(
        epsilon_values, 
        tol=TOL, 
        max_iter=MAX_ITER
    )

    mostrar_resumen(resultados, epsilon_values)
