import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform, norm, gamma,multivariate_normal
from scipy.special import gamma as gamma_func
from statsmodels.graphics.tsaplots import plot_acf


def posterior(alfa, beta, n, r_1, r_2):
    """
    Distribución posterior 
    f(α,β|x) ∝ [β^(nα) / Γ(α)^n] * r_1^(α-1) * exp(-β(r_2+1)) * 1_{1≤α≤4} * 1_{β>1}
    """
    if alfa < 1 or alfa > 4 or beta <= 0:  
        return 0.0
    return (beta ** (n * alfa)) * (r_1 ** (alfa - 1)) * (np.exp(-beta * (r_2 + 1))) / (gamma_func(alfa) ** n)


def simular_gamma(alpha, beta, n=1, seed=123):
    """
    Simulamos una muestra de la distribución gamma

    Parameters
    ----------
    alpha : shape parameter 
    beta : scale inverso parameter 1/beta
    n : tamaño de muestra
    seed : semilla, para reproducibilidad

    Returns
    -------
    Muestra de tamaño de la ditribucion Gamma(alpha,beta)

    """
    np.random.seed(seed)
    muestras = []
    for i in range(n):
        #Generamos Uniforme
        U = np.random.uniform(0, 1, alpha)
        #Transformacion p
        exponenciales = (-1/beta)* np.log(U)
        #Suma es una gamma(n,beta)
        gamma_sample = np.sum(exponenciales)
        muestras.append(gamma_sample)
    
    return np.array(muestras)



def metropolis_hastings(n, r_1, r_2, num_samples=10000, sigma_1=0.15, sigma_2=0.15, seed=123):
    """
    Algoritmo de Metropolis-Hastings para muestrear la posterior conjunta de (α, β).

    Parámetros
    ----------
    n : int
        Tamaño de la muestra Gamma (número de observaciones).
    r_1 : float
        Producto de las observaciones (∏ x_i).
    r_2 : float
        Suma de las observaciones (∑ x_i).
    num_samples : int
        Número de iteraciones
    sigma_1 : float, opcional
        Desviación estándar de la propuesta para α.
    sigma_2 : float, opcional
        Desviación estándar de la propuesta para β.
    seed : int, opcional
        Semilla para reproducibilidad.

    Retorna
    -------
    samples : ndarray
        Muestras de la cadena de Markov, tamaño (num_samples, 2).
    acceptance_rate : float
        Tasa de aceptación.
    """

    np.random.seed(seed)

    # Inicialización: α,β  uniformemente en [1,4]
    alpha_init = np.random.uniform(1, 4)
    beta_init = np.random.uniform(1, 4)
    p_current = np.array([alpha_init, beta_init])

    # Matriz de covarianza diagonal
    cov = np.diag([sigma_1**2, sigma_2**2])

    samples = np.zeros((num_samples, 2))
    acceptance_count = 0

  
    for i in range(num_samples):
        # Propuesta: normal bivariada centrada en el punto actual
        p_proposed = np.random.multivariate_normal(mean=np.array([0,0]) , cov=cov)

        alpha_p, beta_p = p_proposed
        alpha_c, beta_c = p_current

        if (1 <= alpha_p <= 4) and (beta_p > 0):
            # Evaluamos la posterior
            post_current = posterior(alpha_c, beta_c, n, r_1, r_2)
            post_proposed = posterior(alpha_p, beta_p, n, r_1, r_2)

            # Propuestas simétricas 
            if post_current == 0:
                acceptance_ratio = 1
            else:
                acceptance_ratio = post_proposed / post_current
        else:
            acceptance_ratio = 0  # fuera del soporte

        rho = min(1, acceptance_ratio)
        u = np.random.uniform()

        if u < rho:
            p_current = p_proposed
            acceptance_count += 1

        samples[i, :] = p_current

    acceptance_rate = acceptance_count / num_samples
    return samples, acceptance_rate


# Parámetros
a, b = 3, 100

# Simular para ambos casos
n_casos = [5, 40]
muestras = []  
for i, n in enumerate(n_casos):
    # Simular muestras
    k = simular_gamma(a, b, n, seed=1234)
    muestras.append(k)

k_2=muestras[0].sum()
k_1= muestras[0].prod()
k_1
# Crear la malla de puntos
ngrid = 500
alfa= np.linspace(2, 5, ngrid)
beta = np.linspace(-1, 10, ngrid)

# Crear matriz para almacenar los valores de la función
f = np.zeros((ngrid, ngrid))

# Calcular los valores de la función Funcion_Objetivo() en cada punto de la malla
for i in range(ngrid):
    for j in range(ngrid):
        f[i, j] = posterior(alfa[i] , beta[j],5,k_1,k_2)



# Simular para n=5
n = 5
muestras_gamma = simular_gamma(a, b, n, seed=1234)

# Calcular r_1 y r_2 para la posterior
r_1 = np.prod(muestras_gamma)  # Producto de las muestras
r_2 = np.sum(muestras_gamma)   # Suma de las muestras


# Crear la malla de puntos para α y β
ngrid = 100  
alfa_vals = np.linspace(1, 4, ngrid)    
beta_vals = np.linspace(1, 20, ngrid)  

# Crear matriz para almacenar los valores de la posterior
f = np.zeros((ngrid, ngrid))

# Calcular los valores de la posterior en cada punto de la malla
for i in range(ngrid):
    for j in range(ngrid):
        f[i, j] = posterior(alfa_vals[i], beta_vals[j], n, r_1, r_2)



# Crear el gráfico de contorno
plt.figure(figsize=(10, 8))
contour_plot = plt.contour(alfa_vals, beta_vals, f.T, levels=20, colors='black', linewidths=0.5)



plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$\beta$', fontsize=14)
plt.title(f'Distribución Posterior - n={n}\n' + 
          f'$r_1$={r_1:.2e}, $r_2$={r_2:.3f}', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0.5,4.5)
plt.ylim(0,12)

plt.legend()
plt.tight_layout()
plt.show()



###############################
#Metropolis
###




samples, acc_rate = metropolis_hastings(
    initial_point=[0.0, 0.0],
    n_samples=20000,
    proposal_std=.1,
    seed=42
)

    
