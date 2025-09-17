import numpy as np
import matplotlib.pyplot as plt
import time
from Ejercicio3 import generate_data, create_vandermonde, least_squares_qr

ns = [100, 500, 1000,5000] #Si le pone 10k tarda 5 minutos
tiempos = []

for n in ns:
    p = max(2, int(0.1 * n))
    x, y, y_true = generate_data(n)
    A = create_vandermonde(x, p)
    
    start = time.time()
    beta = least_squares_qr(A, y)
    end = time.time()
    
    tiempo_ms = (end - start) * 1000
    tiempos.append(tiempo_ms)
    print(f"n={n}: {tiempo_ms:.2f} ms")

plt.figure(figsize=(8, 5))
plt.plot(ns, tiempos, 'ro--')
plt.xlabel('Tamaño de muestra (n)')
plt.ylabel('Tiempo (ms)')
plt.title('Tiempo de ejecución QR (p = 0.1n)')
plt.grid(True)
plt.show()