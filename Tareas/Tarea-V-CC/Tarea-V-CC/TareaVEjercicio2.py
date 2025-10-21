import numpy as np
import matplotlib.pyplot as plt

def generar_uniformes(n, semilla=12345):
    modulo = 2**31 - 1  
    a1 = 107374182
    a5 = 104420
    
    # Estado inicial: [x₀, x₁, x₂, x₃, x₄]
    estado = [((semilla * (i+1)) + 12345) % modulo for i in range(5)]
    
    resultados = []
    
    for i in range(n):
        # xᵢ = 107374182 * x₀ + 104420 * x₄
        nuevo_valor = (a1 * estado[0] + a5 * estado[4]) % modulo
        
        # x_{j-1} = x_j para j = 1, 2, 3, 4, 5
        # j = 1: x₀ = x₁
        estado[0] = estado[1]
        # j = 2: x₁ = x₂  
        estado[1] = estado[2]
        # j = 3: x₂ = x₃
        estado[2] = estado[3]
        # j = 4: x₃ = x₄
        estado[3] = estado[4]
        # j = 5: x₄ = nuevo_valor 
        estado[4] = nuevo_valor
        
        resultados.append(nuevo_valor / modulo)
    
    return resultados

def probar_uniformidad(n):
    muestra = generar_uniformes(n)
    
    # Estadísticas básicas
    media = sum(muestra) / len(muestra)
    varianza = sum((x - media)**2 for x in muestra) / len(muestra)
    
    print("=== ANÁLISIS DE UNIFORMIDAD ===")
    print(f"Muestra: {n:,} números")
    print(f"Media: {media:.6f} ")
    print(f"Varianza: {varianza:.6f} ")
    
    # Test de frecuencia en intervalos
    intervalos = [0] * 10
    for x in muestra:
        idx = int(x * 10)
        if idx == 10: idx = 9
        intervalos[idx] += 1
    
    print("\nFrecuencia por intervalos:")
    for i, freq in enumerate(intervalos):
        esperado = n / 10
        diferencia = freq - esperado
        print(f"[{i/10:.1f}-{(i+1)/10:.1f}): {freq:6d} ")
    
    # Histograma MEJORADO
    plt.figure(figsize=(12, 8))
    
    # Histograma principal
    n_bins = 20
    counts, bins, patches = plt.hist(muestra, bins=n_bins, color='lightsteelblue', 
                                    edgecolor='navy', alpha=0.7, linewidth=1.2,
                                    density=True, label='Datos generados')
    
    # Línea teórica
    x_theoretical = np.linspace(0, 1, 100)
    y_theoretical = np.ones_like(x_theoretical)  # densidad = 1 para U(0,1)
    plt.plot(x_theoretical, y_theoretical, 'r--', linewidth=2.5, 
             label='Distribución Uniforme Teórica')
    
    # Personalización
    plt.xlabel('Valor', fontsize=12, fontweight='bold')
    plt.ylabel('Densidad', fontsize=12, fontweight='bold')
    plt.title(f'Histograma de {n:,} números U(0,1) generados\n'
              f'Media: {media:.4f}, Varianza: {varianza:.4f}', 
              fontsize=14, fontweight='bold')
    
    # Cuadrícula
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Leyenda
    plt.legend(fontsize=11, framealpha=0.9)
    
    # Límites y ticks
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, max(max(counts), 1.2))
    
 
    
# Ejemplos de uso
if __name__ == "__main__":
    # Generar números, se sugiere poner n del orden de 10k


    probar_uniformidad(2000000)
