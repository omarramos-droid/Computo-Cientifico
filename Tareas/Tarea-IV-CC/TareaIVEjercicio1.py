import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# Valor de epsilon
np.random.seed(12+1)
epsilon = np.random.uniform(-1, 1)  
# Definir la matriz
A = np.array([[8, 1, 0],
              [1, 4, epsilon],
              [0, epsilon, 1]])

# Número de filas
n = A.shape[0]
centros = np.diag(A)  # a_ii
radios = np.zeros(n)

print(f"\n=== Para ε = {epsilon:.3f} ===")
for i in range(n):
    # Radio = suma de valores absolutos de los no diagonales
    radios[i] = sum(abs(A[i, j]) for j in range(n) if j != i)
    print(f"Disco {i+1}: centro = {centros[i]}, radio = {radios[i]}, "
          f"intervalo = [{centros[i]-radios[i]:.2f}, {centros[i]+radios[i]:.2f}]")

# Eigenvalores exactos
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalores reales: {eigenvalues}")

# Graficar discos + eigenvalores
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["red", "blue", "green"]

for k in range(n):
    circle = plt.Circle((centros[k].real, centros[k].imag),
                        radios[k], color=colors[k], alpha=0.3, fill=True,
                        label=f'Radio {radios[k]:.2f}')
    ax.add_artist(circle)
    ax.plot(centros[k].real, centros[k].imag, 'o', color=colors[k])  # centro

# Eigenvalores 
ax.plot(eigenvalues.real, eigenvalues.imag, 'x', markersize=5, label="Eigenvalores")
ax.set_aspect(1)
ax.set_xlim(-1, 10)
ax.set_ylim(-2, 2)
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title(f"Discos de Gershgorin y eigenvalores (ε = {epsilon:.3f} )")
plt.legend(loc='lower right', fontsize='small', ncol=2, frameon=False)
plt.grid(True)
plt.show()
