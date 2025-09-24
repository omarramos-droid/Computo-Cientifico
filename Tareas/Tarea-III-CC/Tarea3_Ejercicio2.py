import numpy as np
import scipy.linalg as la

from Ejercicio2 import least_squares_qr  


def generar_x_bien_condicionada(n, d):
    """
    Genera matriz X bien condicionada con entradas U(0,1)
    """
    return np.random.uniform(0, 1, (n, d))


def generar_x_mal_condicionada(n, d):
    """
    Genera matriz X mal condicionada con casi colinealidad
    """
    X = np.random.uniform(0, 1, (n, d))

    # Crear casi colinealidad en dos columnas
    X[:, 1] = X[:, 0] + np.random.normal(0, 0.07, n)
    return X


def simular_datos(X, beta_verdadero, sigma):
    """
    Simula datos: y = Xβ + ε  
    """
    n = X.shape[0]
    epsilon = np.random.normal(0, sigma, n)
    y = X @ beta_verdadero + epsilon
    return y


def Inciso_a():
    """
    Inciso (a): X bien condicionada
    """
    print("="*60)
    print("INCISO (a): X BIEN CONDICIONADA")
    print("="*60)
    # Parámetros
    n, d = 20, 5
    beta_verdadero = np.array([5, 4, 3, 2, 1])
    sigma = 0.12
    sigma_perturbacion = 0.01

    # Generar X bien condicionada
    X = generar_x_bien_condicionada(n, d)
    print(f"Número de condición de la matriz de diseño X: {np.linalg.cond(X):.6}  ")

    # Simular y
    y = simular_datos(X, beta_verdadero, sigma)

    # β_hat con QR
    beta_hat = least_squares_qr(X, y)
    
    # Perturbar X
    delta_X = np.random.normal(0, sigma_perturbacion, (n, d))
    Xp = X + delta_X
    print(f"Número de condición de la matriz perturbada  X_p: {np.linalg.cond(Xp):.6}  ")

    # β_hat con QR usando X perturbada
    beta_hat_p = least_squares_qr(Xp, y)
    
    # β_hat_c usando fórmula clásica
    XtX = Xp.T @ Xp
    beta_hat_c = la.inv(XtX) @ Xp.T @ y
    
    

    # Resultados
    print("β verdadero:", beta_verdadero)
    print("β_hat (QR):", beta_hat)
    print("β_hat_p (QR, X perturbada):", beta_hat_p)
    print("β_hat_c Scipy :", beta_hat_c)


    # Errores
    print("\nErrores relativos a β verdadero:")
    print(f"Error QR relativo: {np.linalg.norm(beta_hat - beta_verdadero)/np.linalg.norm( beta_verdadero):.6}")
    print(f"Error QR perturbado: {np.linalg.norm(beta_hat_p - beta_verdadero)/np.linalg.norm( beta_verdadero):.6}")
    print(f"Error Scipy: {np.linalg.norm(beta_hat_c - beta_verdadero)/np.linalg.norm( beta_verdadero):.6}")



def Inciso_b():
    """
    Inciso (b): X mal condicionada
    """
    print("\n" + "="*60)
    print("INCISO (b): X MAL CONDICIONADA")
    print("="*60)

    # Parámetros
    n, d = 20, 5
    beta_verdadero = np.array([5, 4, 3, 2, 1])
    sigma = 0.12
    sigma_perturbacion = 0.01

    # Generar X mal condicionada
    X = generar_x_mal_condicionada(n, d)
    print(f"Número de condición de la matriz de diseño X: {np.linalg.cond(X):.5}  ")

    # Simular y
    y = simular_datos(X, beta_verdadero, sigma)

    # β_hat con QR
    beta_hat = least_squares_qr(X, y)

    # Perturbar X
    delta_X = np.random.normal(0, sigma_perturbacion, (n, d))
    Xp = X + delta_X
    print(f"Número de condición de la matriz perturbada  X_p: {np.linalg.cond(Xp):.5}  ")

    # β_hat con QR usando X perturbada
    beta_hat_p = least_squares_qr(Xp, y)

    # β_hat_c usando fórmula clásica
    XtX = Xp.T @ Xp
    beta_hat_c = la.inv(XtX) @ Xp.T @ y

    # Resultados
    print("β verdadero:", beta_verdadero)
    print("β_hat (QR):", beta_hat)
    print("β_hat_p (QR, X perturbada):", beta_hat_p)
    print("β_hat_c (clásica con inversa):", beta_hat_c)

    # Errores
    print("\nErrores relativos a β verdadero:")
    print(f"Error QR: {np.linalg.norm(beta_hat - beta_verdadero)/np.linalg.norm( beta_verdadero):.6f}")
    print(f"Error QR perturbado: {np.linalg.norm(beta_hat_p - beta_verdadero)/np.linalg.norm( beta_verdadero):.6f}")
    print(f"Error Scipy: {np.linalg.norm(beta_hat_c - beta_verdadero)/np.linalg.norm( beta_verdadero):.6f}")



if __name__ == "__main__":
    np.random.seed(13)
    Inciso_a()
    Inciso_b()