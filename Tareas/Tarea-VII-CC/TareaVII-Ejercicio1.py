import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from statsmodels.graphics.tsaplots import plot_acf

def posterior_log(alfa, beta, n, logr_1, r_2, c=1):
    """
    Calcula el logaritmo de la distribución posterior para parámetros Gamma.
    Distribución posterior en escala logarítmica para evitar underflow/overflow
    log f(α,β|x) ∝ nα·log(β) - n·logΓ(α) + (α-1)·log(r_1) - β(r_2+1)
    
    Parameters
    ----------
    alfa : float
        Parámetro de forma de la distribución Gamma 
    beta : float
        Parámetro de tasa de la distribución Gamma
    n : int
        Tamaño de la muestra
    logr_1 : float
        logaritmo del producto de las observaciones 
    r_2 : float
        Suma de las observaciones 
    c : float, optional
        Parametro de la exponencial a priori (default=1)
    
    Returns
    -------
    float
        Logaritmo de la densidad posterior, o -inf si parámetros fuera del soporte
    """
    if alfa < 1 or alfa > 4 or beta <= 0:  
        return -np.inf
    
    # Calcula el log-posterior usando forma logarítmica para estabilidad numérica
    log_post = (n * alfa * np.log(beta) - 
                n * gammaln(alfa) + 
                (alfa - 1) * logr_1 - 
                beta * (r_2 + c))
    
    return log_post

def simular_gamma(alpha, beta, n=1, seed=123):
    """
    Simula muestras de una distribución Gamma(alpha, beta) usando el método
    de suma de variables exponenciales (cuando alpha es entero).
    
    Parameters
    ----------
    alpha : int
        Parámetro de forma (debe ser entero positivo)
    beta : float
        Parámetro de tasa
    n : int
        Número de muestras a generar
    seed : int
        Semilla para reproducibilidad
    
    Returns
    -------
    numpy.ndarray
        Array con n muestras de la distribución Gamma(alpha, beta)
    """
    np.random.seed(seed)
    muestras = []
    for i in range(n):
        # Generar alpha variables uniformes
        U = np.random.uniform(0, 1, alpha)
        # Transformar a exponenciales usando método de transformada inversa
        exponenciales = (-1/beta) * np.log(U)
        # Sumar para obtener variable Gamma
        gamma_sample = np.sum(exponenciales)
        muestras.append(gamma_sample)
    
    return np.array(muestras)

def metropolis_hastings(n, logr_1, r_2, num_samples=15000, sigma_1=1.2, sigma_2=5, seed=123, c=1):
    """
    Metropolis-Hastings para muestrear de la distribución posterior.
    
    Parameters
    ----------
    n : int
        Tamaño de la muestra original
    logr_1 : float
        logaritmo del producto de las observaciones
    r_2 : float
        Suma de las observaciones
    num_samples : int
        Número total de iteraciones MCMC
    sigma_1 : float
        Desviación estándar para propuestas de alpha
    sigma_2 : float
        Desviación estándar para propuestas de beta
    seed : int
        Semilla para reproducibilidad
    c : float
        Parametro de la exponencial (Apriori para beta)
    
    Returns
    -------
    tuple
        - samples: array con las muestras MCMC
        - acceptance_rate: tasa de aceptación del algoritmo
    """
    np.random.seed(seed)

    # Inicialización desde las distribuciones previas
    alpha_init = np.random.uniform(2, 4)  # Uniforme(2,4)
    beta_init = np.random.exponential(1/c)  # Exponencial(1/c)
    
    p_current = np.array([alpha_init, beta_init])
    # Propuesta normal
    cov = np.diag([sigma_1**2, sigma_2**2])

    samples = np.zeros((num_samples, 2))
    acceptance_count = 0

    for i in range(num_samples):
        increment = np.random.multivariate_normal(mean=np.array([0, 0]), cov=cov)
        p_proposed = p_current + increment

        alpha_p, beta_p = p_proposed
        alpha_c, beta_c = p_current

        # Calcular ratio de aceptación 
        log_post_current = posterior_log(alpha_c, beta_c, n, logr_1, r_2, c=c)
        log_post_proposed = posterior_log(alpha_p, beta_p, n, logr_1, r_2, c=c)

        if np.isfinite(log_post_proposed) and np.isfinite(log_post_current):
            log_acceptance_ratio = log_post_proposed - log_post_current
            # Usar min(0, ratio) para estabilidad numérica
            acceptance_ratio = np.exp(min(0, log_acceptance_ratio))
        else:
            acceptance_ratio = 0

        # Criterio de aceptación/rechazo
        if np.random.uniform() < acceptance_ratio:
            p_current = p_proposed
            acceptance_count += 1

        samples[i, :] = p_current

    acceptance_rate = acceptance_count / num_samples
    return samples, acceptance_rate

def graficar_diagnosticos_mcmc(samples, samples_burnt, a, b, burn_in, n):
    """
    Graficas para las cadenas MCMC: series de tiempo y marginales.
    
    Parameters
    ----------
    samples : numpy.ndarray
        Todas las muestras MCMC (incluyendo burn-in)
    samples_burnt : numpy.ndarray
        Muestras después de eliminar el burn-in
    a : float
        Valor verdadero de alpha
    b : float
        Valor verdadero de beta
    burn_in : int
        Número de iteraciones descartadas como burn-in
    n : int
        Tamaño de muestra
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SERIES DE TIEMPO
    axes[0, 0].plot(samples[:, 0], 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(y=a, color='red', linestyle='--', label='Verdadero')
    axes[0, 0].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 0].set_xlabel('Iteración')
    axes[0, 0].set_ylabel(r'$\alpha$')
    axes[0, 0].set_title(f'Cadena de Markov - $\\alpha$ para n = {n}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(samples[:, 1], 'g-', alpha=0.7, linewidth=0.5)
    axes[0, 1].axhline(y=b, color='red', linestyle='--', label='Verdadero')
    axes[0, 1].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 1].set_xlabel('Iteración')
    axes[0, 1].set_ylabel(r'$\beta$')
    axes[0, 1].set_title(f'Cadena de Markov - $\\beta$ para n = {n}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # DISTRIBUCIONES MARGINALES
    axes[1, 0].hist(samples_burnt[:, 0], bins=30, density=True, alpha=0.7, 
                   color='blue', edgecolor='black')
    axes[1, 0].axvline(x=a, color='red', linestyle='--', label='Verdadero')
    axes[1, 0].axvline(x=np.mean(samples_burnt[:, 0]), color='blue', 
                      linestyle='-', label='Media MCMC')
    axes[1, 0].set_xlabel(r'$\alpha$')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Distribución marginal - ' + r'$\alpha$')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(samples_burnt[:, 1], bins=30, density=True, alpha=0.7, 
                   color='green', edgecolor='black')
    axes[1, 1].axvline(x=b, color='red', linestyle='--', label='Verdadero')
    axes[1, 1].axvline(x=np.mean(samples_burnt[:, 1]), color='green', 
                      linestyle='-', label='Media MCMC')
    axes[1, 1].set_xlabel(r'$\beta$')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].set_title('Distribución marginal - ' + r'$\beta$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_espacio_parametros(samples, samples_burnt, a, b, n, logr_1, r_2, c=1):
    """
    Genera gráficos del espacio de parámetros: curvas de nivel y trayectoria de la cadena.
    
    Parameters
    ----------
    samples : numpy.ndarray
        Muestras MCMC 
    samples_burnt : numpy.ndarray
        Muestras después de eliminar el burn-in
    a : float
        Valor verdadero de alpha
    b : float
        Valor verdadero de beta
    n : int
        Tamaño de la muestra original
    logr_1 : float
        logaritmo del producto de las observaciones
    r_2 : float
        Suma de las observaciones
    c : float
        Parametro de beta
    """
    # Crear malla de puntos para las curvas de nivel
    alpha_grid = np.linspace(1, 4, 50)
    beta_grid = np.linspace(0.1, 200, 50)
    
    # Calcular log-posterior en cada punto de la malla
    log_post_grid = np.zeros((len(alpha_grid), len(beta_grid)))
    for i, alpha_val in enumerate(alpha_grid):
        for j, beta_val in enumerate(beta_grid):
            log_post_grid[i, j] = posterior_log(alpha_val, beta_val, n, logr_1, r_2, c=c)
    
    # Convertir a densidad posterior
    post_grid = np.exp(log_post_grid)
    
    # CURVAS DE NIVEL CON MUESTRAS
    plt.figure(figsize=(10, 8))
    
    contour_main = plt.contour(alpha_grid, beta_grid, post_grid.T, levels=15, 
                              colors='black', linewidths=1.5, alpha=0.7)

    plt.scatter(samples_burnt[:, 0], samples_burnt[:, 1], alpha=0.4, 
               s=2, color='blue', label='Muestras MCMC (post burn-in)')
    
    plt.plot(a, b, 'ro', markersize=10, label='Verdadero', markeredgecolor='black')
    
    plt.xlabel(r'$\alpha$', fontsize=12)
    plt.ylabel(r'$\beta$', fontsize=12)
    plt.title(f'Curvas de Nivel Posterior con Muestras MCMC para n = {n}', fontsize=12)
    plt.legend()
    #Por motivos de esteticas hay 2 disitntos niveles
    plt.ylim(0, 200)
    plt.xlim(0.99, 4.1)
    # plt.ylim(0, 50)
    # plt.xlim(0.99, 3.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parámetros del análisis
    n_muestras = 5
    a_verdadero = 3
    b_verdadero = 100
    c_param = 0.1  
    # c_param=1
    seed = 123
    
    print(f"Parámetros verdaderos: α={a_verdadero}, β={b_verdadero}")
    print(f"Parámetro de priori c: {c_param}")
    
    # 1. Simular datos
    muestras_gamma = simular_gamma(a_verdadero, b_verdadero, n=n_muestras, seed=seed)
    
    # 2. Calcular estadísticos suficientes
    logr_1 = np.log(np.prod(muestras_gamma))
    r_2 = np.sum(muestras_gamma)
    
    print(f"Datos simulados (n={n_muestras}):")
    print(f"  Media muestral: {np.mean(muestras_gamma):.4f}")
    print(f"  log(r_1) = {logr_1:.6f}")
    print(f"  r_2 = {r_2:.3f}")
    
    # 3. Ejecutar MCMC
    print("\nEjecutando Metropolis-Hastings...")
    samples, acc_rate = metropolis_hastings(
        n=n_muestras, 
        logr_1=logr_1, 
        r_2=r_2, 
        num_samples=10_000,
        sigma_1=1,  
        sigma_2=5,
        seed=seed,
        c=c_param
    )
    # samples, acc_rate = metropolis_hastings(
    #     n=n_muestras, 
    #     logr_1=logr_1, 
    #     r_2=r_2, 
    #     num_samples=10_000,
    #     sigma_1=1,  
    #     sigma_2=2,
    #     seed=seed,
    #     c=c_param
    # )
    
    #4. Eliminar burn-in
    burn_in = 2000
    samples_burnt = samples[burn_in:]
    
    # 5. Mostrar resultados
    print(f"Tasa de aceptación: {acc_rate:.3f}")
    print(f"Media de α: {np.mean(samples_burnt[:, 0]):.3f} (verdadero: {a_verdadero})")
    print(f"Media de β: {np.mean(samples_burnt[:, 1]):.3f} (verdadero: {b_verdadero})")
    

    # 6. Generar gráficos
    graficar_diagnosticos_mcmc(samples, samples_burnt, a_verdadero, b_verdadero, burn_in, n_muestras)
    graficar_espacio_parametros(samples, samples_burnt, a_verdadero, b_verdadero, n_muestras, logr_1, r_2, c=c_param)
    
    # 7. Análisis de autocorrelación
    alpha_samples = samples_burnt[:, 0]
    beta_samples = samples_burnt[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Autocorrelación sin thinning
    plot_acf(alpha_samples, lags=50, ax=axes[0, 0], title="Autocorrelación α (sin thinning)")
    plot_acf(beta_samples, lags=50, ax=axes[1, 0], title="Autocorrelación β (sin thinning)")
    
    # Aplicar thinning
    thinning = 50
    alpha_thinned = alpha_samples[::thinning]
    beta_thinned = beta_samples[::thinning]
    
    # Autocorrelación después de thinning
    plot_acf(alpha_thinned, lags=50, ax=axes[0, 1], title=f"Autocorrelación α (thinning={thinning})")
    plot_acf(beta_thinned, lags=50, ax=axes[1, 1], title=f"Autocorrelación β (thinning={thinning})")
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar tamaños de muestra
    print(f"Muestras totales después de burn-in: {len(alpha_samples)}")
    print(f"Muestras después de thinning (cada {thinning}): {len(alpha_thinned)}")