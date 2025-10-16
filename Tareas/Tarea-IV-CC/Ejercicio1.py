import numpy as np
import matplotlib.pyplot as plt

# Valor de epsilon
epsilon = -.48  # cámbialo a lo que quieras dentro de [-1,1]

# Definir la matriz
A = np.array([[8, 1, 0],
              [1, 4, epsilon],
              [0, epsilon, 1]])

# Número de filas
n = A.shape[0]
centros = np.diag(A)  # a_ii
radios = np.zeros(n)

print(f"\n=== Para ε = {epsilon} ===")
for i in range(n):
    # Radio = suma de valores absolutos de los no diagonales
    radios[i] = sum(abs(A[i, j]) for j in range(n) if j != i)
    print(f"Disco {i+1}: centro = {centros[i]}, radio = {radios[i]}, "
          f"intervalo = [{centros[i]-radios[i]:.2f}, {centros[i]+radios[i]:.2f}]")

# Eigenvalores exactos
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalores reales: {eigenvalues}")

# Graficar discos + eigenvalores
fig, ax = plt.subplots(figsize=(7, 7))
colors = ["red", "blue", "green"]

for k in range(n):
    circle = plt.Circle((centros[k].real, centros[k].imag),
                        radios[k], color=colors[k], alpha=0.3, fill=True,
                        label=f'Disco {k+1}')
    ax.add_artist(circle)
    ax.plot(centros[k].real, centros[k].imag, 'o', color=colors[k])  # centro
    ax.plot(eigenvalues.real, eigenvalues.imag, 'X', markersize=12, color=colors[k] ,label=f'Eigenvalor {k+1}' )

ax.set_aspect(1)
ax.set_xlim(-1, 10)
ax.set_ylim(-2, 2)
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title(f"Discos de Gershgorin y eigenvalores (ε = {epsilon})")
plt.legend(loc='lower left',bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()


# # Verificar si los eigenvalores están dentro de los discos
# print("\nVerificación:")
# for i, ev in enumerate(eigenvalues):
#     for j in range(3):
#         distancia = abs(ev - centros[j])
#         dentro = distancia <= radios[j] + 1e-10  # pequeña tolerancia numérica
#         print(f"Eigenvalor {i+1} ({ev:.4f}) en Disco {j+1}: {dentro} (distancia: {distancia:.4f}, radio: {radios[j]})")