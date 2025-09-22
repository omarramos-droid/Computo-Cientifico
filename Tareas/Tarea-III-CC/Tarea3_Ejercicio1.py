import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.linalg
import time

# Agregar la ruta de funciones auxiliares al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'Funciones_Auxiliares'))
from Tarea_1_ex5and6 import cholesky_factorization
from Ejercicio1 import modified_gram_schmidt

def generar_matrices_espectrales():
    np.random.seed(13)  
    A= np.random.uniform(0, 10, (20, 20))
    Q,R = modified_gram_schmidt(A)
    λ_buenos = np.linspace(10, 1, 20)  
    λ_malos = np.linspace(10000, 1, 20)  
    perturbacion = np.random.normal(0, 0.01, 20)
    
    B_bien = Q @ np.diag(λ_buenos) @ Q.T
    B_mal = Q @ np.diag(λ_malos) @ Q.T
    Bε_bien = Q @ np.diag(λ_buenos + perturbacion) @ Q.T
    Bε_mal = Q @ np.diag(λ_malos + perturbacion) @ Q.T
    
    return {
        'B_bien': B_bien, 'Bε_bien': Bε_bien,
        'B_mal': B_mal, 'Bε_mal': Bε_mal
    }

def analizar_estabilidad_cholesky(B, Bε, metodo="propio"):
    """
    Analiza la estabilidad del algoritmo de Cholesky.
    metodo = "propio" usa la implementacion de la Tarea 2 (Inciso a)
    metodo = "scipy" usa scipy.linalg.cholesky (Inciso b)
    """
    κ = np.linalg.cond(B)
    error_entrada = np.linalg.norm(Bε - B) / np.linalg.norm(B)

    # Selección del método
    if metodo == "propio":
        L = cholesky_factorization(B)
        Lε = cholesky_factorization(Bε)
    elif metodo == "scipy":
        L = scipy.linalg.cholesky(B, lower=True)
        Lε = scipy.linalg.cholesky(Bε, lower=True)
    else:
        raise ValueError("Método debe ser 'propio' o 'scipy'")
    
    error_salida = np.linalg.norm(Lε - L) / np.linalg.norm(L)
    error_reconstruccion_B = np.linalg.norm(L @ L.T - B) / np.linalg.norm(B)
    error_reconstruccion_Bε = np.linalg.norm(Lε @ Lε.T - Bε) / np.linalg.norm(Bε)
    diferencia_descomposicion = np.linalg.norm(Lε - L) / np.linalg.norm(L)

    return κ, error_entrada, error_salida, error_reconstruccion_B, error_reconstruccion_Bε, diferencia_descomposicion

def Incisoa():
    """
    Análisis principal de estabilidad del algoritmo de Cholesky
    """
    print("="*70)
    print("INCISO A - ALGORITMO DE CHOLESKY")
    print("Comparación de descomposiciones de Cholesky de B y Bε")
    print("="*70)
    
    # Generar matrices
    matrices = generar_matrices_espectrales()
    
    # Analizar ambos casos
    κ_bien, error_entrada_bien, error_salida_bien, error_rec_B_bien,error_rec_Bε_bien, diff_bien = analizar_estabilidad_cholesky(
        matrices['B_bien'], matrices['Bε_bien'],)
    
    κ_mal, error_entrada_mal, error_salida_mal, error_rec_B_mal, error_rec_Bε_mal, diff_mal = analizar_estabilidad_cholesky(
        matrices['B_mal'], matrices['Bε_mal'])
    
    print("\nBUEN CONDICIONAMIENTO:")
    print(f"  Número de condición: {κ_bien:.2e}")
    print(f"  Error entrada ||Bε-B||/||B||: {error_entrada_bien:.2e}")
    print(f"  Error salida ||Lε-L||/||L||: {error_salida_bien:.2e}")
    print(f"  Ratio error_salida/error_entrada: {error_salida_bien/error_entrada_bien:.2f}")
    print(f"  Error reconstrucción B: {error_rec_B_bien:.2e}")
    print(f"  Error reconstrucción Bε: {error_rec_Bε_bien:.2e}")
    print(f"  Diferencia descomposiciones: {diff_bien:.2e}")
    
    print("\nMAL CONDICIONAMIENTO:")
    print(f"  Número de condición: {κ_mal:.2e}")
    print(f"  Error entrada ||Bε-B||/||B||: {error_entrada_mal:.2e}")
    print(f"  Error salida ||Lε-L||/||L||: {error_salida_mal:.2e}")
    print(f"  Ratio error_salida/error_entrada: {error_salida_mal/error_entrada_mal:.2f}")
    print(f"  Error reconstrucción B: {error_rec_B_mal:.2e}")
    print(f"  Error reconstrucción Bε: {error_rec_Bε_mal:.2e}")
    print(f"  Diferencia descomposiciones: {diff_mal:.2e}")
    
    
def Incisob():
    print("="*70)
    print("INCISO B - COMPARACIÓN ENTRE ALGORITMOS")
    print("="*70)
    
    matrices = generar_matrices_espectrales()
    
    # Nuestro algoritmo
    κ_p, err_in_p, err_out_p, rec_B_p, rec_Bε_p, diff_p = analizar_estabilidad_cholesky(
        matrices['B_mal'], matrices['Bε_mal'], metodo="propio")
    
    # SciPy
    κ_s, err_in_s, err_out_s, rec_B_s, rec_Bε_s, diff_s = analizar_estabilidad_cholesky(
        matrices['B_mal'], matrices['Bε_mal'], metodo="scipy")
    
    print("\nMatriz mal condicionada")
    print("\nAlgoritmo Propio:")
    print(f"  Número de condición: {κ_p:.2e}")
    print(f"  Error entrada: {err_in_p:.2e}")
    print(f"  Error salida: {err_out_p:.2e}")
    print(f"  Ratio salida/entrada: {err_out_p/err_in_p:.2f}")
    print(f"  Error reconstrucción B: {rec_B_p:.2e}")
    print(f"  Error reconstrucción Bε: {rec_Bε_p:.2e}")
    print(f"  Diferencia descomposiciones: {diff_p:.2e}")

    print("\nAlgoritmo SciPy:")
    print(f"  Número de condición: {κ_s:.2e}")
    print(f"  Error entrada: {err_in_s:.2e}")
    print(f"  Error salida: {err_out_s:.2e}")
    print(f"  Ratio salida/entrada: {err_out_s/err_in_s:.2f}")
    print(f"  Error reconstrucción B: {rec_B_s:.2e}")
    print(f"  Error reconstrucción Bε: {rec_Bε_s:.2e}")
    print(f"  Diferencia descomposiciones: {diff_s:.2e}")
    


def Incisoc():
    """
    Compara el tiempo de ejecución entre la implementación propia
    de Cholesky y la de scipy
    """
    ns = [10, 50, 100, 500,1000]  
    tiempos_custom = []
    tiempos_scipy = []

    for n in ns:
        # Generar matriz simétrica definida positiva
        A = np.random.rand(n, n)
        A = np.dot(A, A.T) + n * np.eye(n)

        # Tiempo Cholesky propio
        start = time.perf_counter()
        L_custom = cholesky_factorization(A)
        tiempos_custom.append(time.perf_counter() - start)

        # Tiempo Cholesky SciPy
        start = time.perf_counter()
        L_scipy = scipy.linalg.cholesky(A, lower=True)
        tiempos_scipy.append(time.perf_counter() - start)

    # Graficar
    plt.figure(figsize=(8, 5))
    plt.plot(ns, tiempos_custom, 'ro--', label='Cholesky propio')
    plt.plot(ns, tiempos_scipy, 'bo--', label='Cholesky SciPy')
    plt.xlabel('Tamaño de matriz (n)')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Comparación de tiempos: Cholesky propio vs SciPy')
    plt.legend()
    plt.grid(True)
    plt.show()



    
if __name__ == "__main__":
    Incisoa()
    Incisob()
    Incisoc()