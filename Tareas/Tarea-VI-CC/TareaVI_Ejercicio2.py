import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform
from scipy.integrate import quad


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

    # Propuesta Beta
    alpha_prop = r + 1
    beta_prop = n - r + 1

    samples = []
    acceptance_count = 0

    for i in range(num_samples):
        # Propuesta
        p_proposed = beta.rvs(alpha_prop, beta_prop)

        # Densidades posteriores
        post_current = posterior(p_current, r, n)
        post_proposed = posterior(p_proposed, r, n)
        #Calcular \dfrac{f(x)q(y|x)}{f(y)q(x|y)}
        if post_current > 0 and post_proposed > 0:
            posterior_ratio = post_proposed / post_current
            proposal_ratio = beta.pdf(p_current, alpha_prop, beta_prop) / beta.pdf(p_proposed, alpha_prop, beta_prop)
            acceptance_ratio = posterior_ratio * proposal_ratio
        else:
            acceptance_ratio = 0
        #rho= min(1, f(x)q(y|x)/f(y)q(x|y) )
        rho = min(1, acceptance_ratio)
        if np.random.uniform() < rho:
            p_current = p_proposed
            acceptance_count += 1

        samples.append(p_current)

    acceptance_rate = acceptance_count / num_samples
    return np.array(samples), acceptance_rate


def plot_mh_with_posterior(r, n, label, num_samples=10000):
    """
    Genera muestras con Metropolis-Hastings y compara histograma con densidad analítica.
    """
    samples, acc_rate = metropolis_hastings(r, n, num_samples=num_samples)

    # Densidad posterior normalizada
    p_values = np.linspace(0.001, 0.5, 500)
    constante, _ = quad(lambda p: posterior(p, r, n), 0, 0.5)
    posterior_norm = np.array([posterior(p, r, n) for p in p_values]) / constante

    # Gráfica comparativa
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='white', label='Muestras MH')
    plt.plot(p_values, posterior_norm, color='red', linewidth=2, label='Posterior analítica normalizada')
    plt.title(f'Metropolis-Hastings vs Posterior Analítica\n{label} | Tasa aceptación = {acc_rate:.3f}')
    plt.xlabel('p')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return samples, acc_rate


n = np.array([5, 50])
r = np.array([2, 18])
labels = ['Caso 1: Muestra pequeña', 'Caso 2: Muestra grande']

for n_i, r_i, label in zip(n, r, labels):
    samples, acc_rate = plot_mh_with_posterior(r_i, n_i, label)



