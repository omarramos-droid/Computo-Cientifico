import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform, norm
from scipy.integrate import quad
from statsmodels.graphics.tsaplots import plot_acf

#
# MCMC CON PROPUESTA BETA
#

def posterior(p, r, n):
    """
    Distribución posterior 
    f(p) ∝ p^r (1-p)^{n-r} cos(πp) 1_{[0,1/2]}(p)
    """
    if p < 0 or p > 0.5:
        return 0.0
    return (p ** r) * ((1 - p) ** (n - r)) * np.cos(np.pi * p)


def metropolis_hastings(r, n, num_samples=10000, seed=123):
    """
    Algoritmo Metropolis-Hastings para muestrear de la posterior.
    """
    np.random.seed(seed)
    p_current = uniform.rvs(0, 1/2)
    
    alpha_prop = r + 1
    beta_prop = n - r + 1

    samples = []
    acceptance_count = 0

    for i in range(num_samples):
        # Propuesta Beta
        p_proposed = beta.rvs(alpha_prop, beta_prop)

        # Verificar si ambas están en el soporte y tienen densidad > 0
        if (0 <= p_current <= 0.5 and 0 <= p_proposed <= 0.5 and 
            posterior(p_current, r, n) > 0 and posterior(p_proposed, r, n) > 0):
            
            acceptance_ratio = np.cos(np.pi * p_proposed) / np.cos(np.pi * p_current)
        else:
            acceptance_ratio = 0
            
        rho = min(1, acceptance_ratio)
        if np.random.uniform() < rho:
            p_current = p_proposed
            acceptance_count += 1
            
        samples.append(p_current)

    acceptance_rate = acceptance_count / num_samples
    return np.array(samples), acceptance_rate


#
# MCMC CON PROPUESTA NORMAL 
#
def metropolis_hastings_normal(r, n, num_samples=10000, sigma=0.15, seed=123):
    np.random.seed(seed)
    p_current = uniform.rvs(0, 0.5)
    mu = 0.25
    
    samples = []
    acceptance_count = 0

    for i in range(num_samples):
        p_proposed = np.random.normal(mu, sigma)
      
        if 0 <= p_proposed <= 0.5:
            post_current = posterior(p_current, r, n)  # > 0 porque p_current está en soporte
            post_proposed = posterior(p_proposed, r, n)  # > 0 porque p_proposed está en soporte
            
            # Como ambos están en soporte, post_current y post_proposed > 0
            q_current = norm.pdf(p_current, mu, sigma)
            q_proposed = norm.pdf(p_proposed, mu, sigma)
            
            acceptance_ratio = (post_proposed * q_current) / (post_current * q_proposed)
        else:
            acceptance_ratio = 0  # Rechazo automático por fuera del soporte
            
        rho = min(1, acceptance_ratio)
        if np.random.uniform() < rho:
            p_current = p_proposed
            acceptance_count += 1
            
        samples.append(p_current)
        
    acceptance_rate = acceptance_count / num_samples
    return np.array(samples), acceptance_rate



# FUNCIONES DE VISUALIZACIÓN



def plot_cadena_con_posterior(r, n, label, num_samples=10_000, warmup=1_000, propuesta='beta'):
    """
    Grafica cadena, histograma y distribución posterior normalizada
    """
    # Ejecutar MCMC según la propuesta seleccionada
    if propuesta == 'beta':
        samples, acc_rate = metropolis_hastings(r, n, num_samples=num_samples)
    else: 
        samples, acc_rate = metropolis_hastings_normal(r, n, num_samples=num_samples)
 
    
    # FIGURA 1: CADENA TEMPORAL (primeras 2000 muestras)
    plt.figure(figsize=(12, 4))
    #plt.plot(samples[0:2000], 'b-', alpha=0.7, linewidth=0.8)
    plt.plot(samples, 'b-', alpha=0.7, linewidth=0.8)
    plt.axvline(x=warmup, color='red', linestyle='--', label=f'Fin de warmup ({warmup} iter)')
    plt.xlabel('Iteración')
    #plt.ylim(0,0.5)
    plt.ylabel('p')
    plt.title(f'Cadena MCMC - {label} (Propuesta: {propuesta})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # FIGURA 2: HISTOGRAMA + DISTRIBUCIÓN POSTERIOR NORMALIZADA
    plt.figure(figsize=(10, 6))
    
    # Histograma de las muestras después del warmup
    plt.hist(samples[warmup:], bins=50, density=True, alpha=0.6, 
             color='lightblue', edgecolor='black', label='Muestras MCMC (post-warmup)')
    
    # Calcular y graficar distribución posterior teórica normalizada
    p_grid = np.linspace(0.001, 0.499, 200)
    constante, error = quad(lambda p: posterior(p, r, n), 0, 0.5)
    posterior_norm = np.array([posterior(p, r, n) for p in p_grid]) / constante
    
    plt.plot(p_grid, posterior_norm, 'r-', linewidth=2, 
             label='Distribución posterior normalizada')
    
    plt.xlabel('p')
    plt.ylabel('Densidad')
    plt.title(f'Histograma y Distribución Posterior - {label}\n(Propuesta: {propuesta})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Estadísticas después del warmup
    samples_post_warmup = samples[warmup:]
    print(f"=== {label} (Propuesta: {propuesta}) ===")
    print(f"Muestras totales: {len(samples)}")
    print(f"Muestras post-warmup: {len(samples_post_warmup)}")
    print(f"Tasa de aceptación total: {acc_rate:.4f}")
    print(f"Media posterior (post-warmup): {np.mean(samples_post_warmup):.4f}")
    print(f"Desviación estándar (post-warmup): {np.std(samples_post_warmup):.4f}")
    print(f"Constante de normalización: {constante:.6e}")
    print("-" * 50)
    
    return samples, acc_rate, constante



if __name__ == "__main__":
    # Parámetros de los casos
    n = np.array([5, 50])
    r = np.array([2, 18])
    labels = ['Caso 1: Muestra pequeña', 'Caso 2: Muestra grande']
    
    warmup = 1000
    constantes_norm = []
    
    print("METROPOLIS-HASTINGS CON PROPUESTA BETA")
    
    #Ejecutar con propuesta Beta
    for n_i, r_i, label in zip(n, r, labels):
        # 1. Generar muestras y graficar
        samples, acc_rate, constante = plot_cadena_con_posterior(r_i, n_i, label, warmup=warmup, propuesta='beta')
        constantes_norm.append(constante)
        
        # 2. Graficar autocorrelación (solo post-warmup)
        plt.figure(figsize=(10, 4))
        plot_acf(samples[warmup:], lags=50, title=f'Autocorrelación (post-warmup) - {label} - Beta')
        plt.ylabel("Autocorrelation")
        plt.xlabel("Lag")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    print("METROPOLIS-HASTINGS CON PROPUESTA NORMAL")
    
    # Ejecutar con propuesta Normal
    for n_i, r_i, label in zip(n, r, labels):
        # 1. Generar muestras y graficar
        samples, acc_rate, constante = plot_cadena_con_posterior(r_i, n_i, label, warmup=warmup, propuesta='normal')
        constantes_norm.append(constante)
        
        # 2. Graficar autocorrelación (solo post-warmup)
        plt.figure(figsize=(10, 4))
        plot_acf(samples[warmup:], lags=50, title=f'Autocorrelación (post-warmup) - {label} - Normal')
        plt.ylabel("Autocorrelation")
        plt.xlabel("Lag")
        plt.grid(True, alpha=0.3)
        plt.show()
    


