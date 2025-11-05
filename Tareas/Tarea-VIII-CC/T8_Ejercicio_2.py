import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, gamma, expon, norm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf



def generar_datos_weibull(n=30, alpha_verdadero=1.0, lambda_verdadero=1.0):
    """Genera datos de la distribución Weibull con parámetros alpha y lambda"""
    # np.random.weibull(a) genera Weibull con forma=a y escala=1
    # Para nuestra parametrización: f(t) = alpha * lambda * t^(alpha-1) * exp(-lambda * t^alpha)
    # La escala debería ser: scale = (1/lambda)^(1/alpha)
    
    scale = (1/lambda_verdadero) ** (1/alpha_verdadero)
    t = np.random.weibull(alpha_verdadero, n) * scale
    return t
def log_verosimilitud_weibull(t, alpha, lambda_):
    """Log-verosimilitud de la distribución Weibull"""
    if alpha <= 0 or lambda_ <= 0:
        return -np.inf
    
    n = len(t)
    log_lik = n * np.log(alpha) + n * np.log(lambda_)
    log_lik += (alpha - 1) * np.sum(np.log(t))
    log_lik -= lambda_ * np.sum(t ** alpha)
    return log_lik

def log_prior(alpha, lambda_, c=1, b=1):
    """Log-distribución a priori"""
    if alpha <= 0 or lambda_ <= 0:
        return -np.inf
    
    # alpha ~ exp(c)
    log_prior_alpha = np.log(c) - c * alpha
    
    # lambda|alpha ~ Gamma(alpha, b)
    log_prior_lambda = gamma.logpdf(lambda_, alpha, scale=1/b)
    
    return log_prior_alpha + log_prior_lambda

def log_posterior(t, alpha, lambda_, c=1, b=1):
    """Log-distribución posterior """
    log_prior_val = log_prior(alpha, lambda_, c, b)
    if np.isinf(log_prior_val):
        return -np.inf
    
    log_lik_val = log_verosimilitud_weibull(t, alpha, lambda_)
    if np.isinf(log_lik_val):
        return -np.inf
    
    return log_lik_val + log_prior_val

def propuesta_1(alpha_actual, t, b=1):
    """Propuesta 1: lambda_p|alpha,t ~ Gamma(alpha + n, b + sum(t_i^alpha))"""
    n = len(t)
    shape = alpha_actual + n
    rate = b + np.sum(t ** alpha_actual)
    lambda_prop = np.random.gamma(shape, 1/rate)
    return lambda_prop

def log_densidad_propuesta_1(lambda_prop, alpha_actual, t, b=1):
    """Log-densidad de la propuesta 1"""
    n = len(t)
    shape = alpha_actual + n
    rate = b + np.sum(t ** alpha_actual)
    return gamma.logpdf(lambda_prop, shape, scale=1/rate)

def propuesta_2(lambda_actual, t, c=1, b=1):
    """Propuesta 2: alpha_p|lambda,t ~ Gamma(n+1, -log(b) - log(r1) + c)"""
    n = len(t)
    r1 = np.prod(t)
    shape = n + 1
    rate = -np.log(b) - np.log(r1) + c
    # CORRECCIÓN: Asegurar que rate sea positivo
    if rate <= 0:
        rate = 0.1
    alpha_prop = np.random.gamma(shape, 1/rate)
    return alpha_prop

def log_densidad_propuesta_2(alpha_prop, lambda_actual, t, c=1, b=1):
    """Log-densidad de la propuesta 2"""
    n = len(t)
    r1 = np.prod(t)
    shape = n + 1
    rate = -np.log(b) - np.log(r1) + c
    if rate <= 0:
        rate = 0.1
    return gamma.logpdf(alpha_prop, shape, scale=1/rate)

def propuesta_3(c=1, b=1):
    """Propuesta 3: alpha_p ~ exp(c), lambda_p|alpha_p ~ Gamma(alpha_p, b)"""
    alpha_prop = np.random.exponential(1/c)
    lambda_prop = np.random.gamma(alpha_prop, 1/b)
    return alpha_prop, lambda_prop

def log_densidad_propuesta_3(alpha_prop, lambda_prop, c=1, b=1):
    """Log-densidad de la propuesta 3"""
    dens_alpha = expon.logpdf(alpha_prop, scale=1/c)
    dens_lambda = gamma.logpdf(lambda_prop, alpha_prop, scale=1/b)
    return dens_alpha + dens_lambda

def propuesta_4(alpha_actual, sigma=0.1):
    """Propuesta 4 (RWMH): alpha_p = alpha + epsilon, epsilon ~ N(0,sigma)"""
    alpha_prop = alpha_actual + np.random.normal(0, sigma)
    return alpha_prop

def log_densidad_propuesta_4(alpha_prop, alpha_actual, sigma=0.1):
    """Log-densidad de la propuesta 4 (simétrica)"""
    return norm.logpdf(alpha_prop, alpha_actual, sigma)

def paso_metropolis_hastings(alpha_actual, lambda_actual, t, tipo_propuesta, params):
    """Paso del algoritmo Metropolis-Hastings"""
    c, b, sigma = params
    
    if tipo_propuesta == 1:
        # Propuesta 1: actualizar lambda
        lambda_prop = propuesta_1(alpha_actual, t, b)
        alpha_prop = alpha_actual
        
        # Ratio de aceptación
        log_posterior_actual = log_posterior(t, alpha_actual, lambda_actual, c, b)
        log_posterior_prop = log_posterior(t, alpha_prop, lambda_prop, c, b)
        
        # log-densidades
        q_actual_dado_prop = log_densidad_propuesta_1(lambda_actual, alpha_prop, t, b)
        q_prop_dado_actual = log_densidad_propuesta_1(lambda_prop, alpha_actual, t, b)
        
        log_ratio = (log_posterior_prop - log_posterior_actual + 
                    q_actual_dado_prop - q_prop_dado_actual)
        
    elif tipo_propuesta == 2:
        # Propuesta 2: actualizar alpha
        alpha_prop = propuesta_2(lambda_actual, t, c, b)
        lambda_prop = lambda_actual
        
        # Ratio de aceptación
        log_posterior_actual = log_posterior(t, alpha_actual, lambda_actual, c, b)
        log_posterior_prop = log_posterior(t, alpha_prop, lambda_prop, c, b)
        
        q_actual_dado_prop = log_densidad_propuesta_2(alpha_actual, lambda_prop, t, c, b)
        q_prop_dado_actual = log_densidad_propuesta_2(alpha_prop, lambda_actual, t, c, b)
        
        log_ratio = (log_posterior_prop - log_posterior_actual + 
                    q_actual_dado_prop - q_prop_dado_actual)
        
    elif tipo_propuesta == 3:
        # Propuesta 3: actualizar ambos
        alpha_prop, lambda_prop = propuesta_3(c, b)
        
        # Ratio de aceptación
        log_posterior_actual = log_posterior(t, alpha_actual, lambda_actual, c, b)
        log_posterior_prop = log_posterior(t, alpha_prop, lambda_prop, c, b)
        
        q_actual_dado_prop = log_densidad_propuesta_3(alpha_actual, lambda_actual, c, b)
        q_prop_dado_actual = log_densidad_propuesta_3(alpha_prop, lambda_prop, c, b)
        
        log_ratio = (log_posterior_prop - log_posterior_actual + 
                    q_actual_dado_prop - q_prop_dado_actual)
        
    elif tipo_propuesta == 4:
        # Propuesta 4: RWMH para alpha
        alpha_prop = propuesta_4(alpha_actual, sigma)
        lambda_prop = lambda_actual
        
        # Ratio de aceptación (q es simétrica)
        log_posterior_actual = log_posterior(t, alpha_actual, lambda_actual, c, b)
        log_posterior_prop = log_posterior(t, alpha_prop, lambda_prop, c, b)
        
        log_ratio = log_posterior_prop - log_posterior_actual
        
    # Aceptar o rechazar
    if np.log(np.random.uniform()) < min(0, log_ratio):
        return alpha_prop, lambda_prop, True
    else:
        return alpha_actual, lambda_actual, False

def muestreador_kernel_hibrido_weibull(n_muestras, t, alpha_inicial, lambda_inicial, 
                                      pesos=[0.25, 0.25, 0.25, 0.25], c=1, b=1, sigma=0.1):
    """Algoritmo de Metropolis-Hastings con kernel híbrido para Weibull"""
    
    muestras_alpha = np.zeros(n_muestras)
    muestras_lambda = np.zeros(n_muestras)
    
    alpha_actual = alpha_inicial
    lambda_actual = lambda_inicial
    
    tasas_aceptacion = [0, 0, 0, 0]
    conteo_kernels = [0, 0, 0, 0]
    
    params = (c, b, sigma)
    
    for i in range(n_muestras):
        # Elegir kernel según pesos
        tipo_propuesta = np.random.choice([1, 2, 3, 4], p=pesos)
        conteo_kernels[tipo_propuesta - 1] += 1
        
        # Ejecutar paso de MH
        alpha_actual, lambda_actual, aceptado = paso_metropolis_hastings(
            alpha_actual, lambda_actual, t, tipo_propuesta, params)
        
        if aceptado:
            tasas_aceptacion[tipo_propuesta - 1] += 1
        
        muestras_alpha[i] = alpha_actual
        muestras_lambda[i] = lambda_actual
    
    # Calcular tasas de aceptación 
    tasas_norm = []
    for j in range(4):
        if conteo_kernels[j] > 0:
            tasas_norm.append(tasas_aceptacion[j] / conteo_kernels[j])
        else:
            tasas_norm.append(0.0)
    
    return muestras_alpha, muestras_lambda, tasas_norm, conteo_kernels

# Parámetros de la simulación
if __name__ == "__main__":
    np.random.seed(123)
    
    # Generar datos
    n = 30
    alpha_verdadero = 1.0
    lambda_verdadero = 1.0
    t = generar_datos_weibull(n, alpha_verdadero, lambda_verdadero)
    
    print(f"Datos generados: n = {n}")
    print(f"Media empírica: {np.mean(t):.3f}")
    print(f"Desviación empírica: {np.std(t):.3f}")
    
    # Parámetros MCMC
    n_muestras = 10_000
    burn_in = 5000
    alpha_inicial = 2.0  
    lambda_inicial =0.01
    
    # Ejecutar MCMC
    pesos = [0.25, 0.25, 0.25, 0.25]  # Pesos iguales para los 4 kernels
    muestras_alpha, muestras_lambda, tasas_acept, conteo = muestreador_kernel_hibrido_weibull(
        n_muestras, t, alpha_inicial, lambda_inicial, pesos=pesos)
    
    # Eliminar burn-in
    alpha_burnt = muestras_alpha[burn_in:]
    lambda_burnt = muestras_lambda[burn_in:]
    
    print(f"\nTasas de aceptación por kernel:")
    print(f"  Kernel 1 (lambda|alpha): {tasas_acept[0]:.3f}")
    print(f"  Kernel 2 (alpha|lambda): {tasas_acept[1]:.3f}")
    print(f"  Kernel 3 (ambos): {tasas_acept[2]:.3f}")
    print(f"  Kernel 4 (RWMH alpha): {tasas_acept[3]:.3f}")
    
    print(f"\nKernels usados: {conteo}")
    
    # Estadísticas
    media_alpha = np.mean(alpha_burnt)
    media_lambda = np.mean(lambda_burnt)
    std_alpha = np.std(alpha_burnt)
    std_lambda = np.std(lambda_burnt)
    
    print(f"\nEstadísticas después de burn-in:")
    print(f"  Alpha: {media_alpha:.3f} ± {std_alpha:.3f} (verdadero: {alpha_verdadero})")
    print(f"  Lambda: {media_lambda:.3f} ± {std_lambda:.3f} (verdadero: {lambda_verdadero})")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Series de tiempo
    axes[0, 0].plot(muestras_alpha, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(y=alpha_verdadero, color='red', linestyle='--', label='Verdadero')
    axes[0, 0].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 0].set_xlabel('Iteración')
    axes[0, 0].set_ylabel('Alpha')
    axes[0, 0].set_title('Cadena de Markov - Alpha')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(muestras_lambda, 'g-', alpha=0.7, linewidth=0.5)
    axes[0, 1].axhline(y=lambda_verdadero, color='red', linestyle='--', label='Verdadero')
    axes[0, 1].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 1].set_xlabel('Iteración')
    axes[0, 1].set_ylabel('Lambda')
    axes[0, 1].set_title('Cadena de Markov - Lambda')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribuciones marginales
    axes[1, 0].hist(alpha_burnt, bins=30, density=True, alpha=0.7, color='pink', edgecolor='black')
    axes[1, 0].axvline(x=alpha_verdadero, color='red', linestyle='--', label='Verdadero')
    axes[1, 0].axvline(x=media_alpha, color='blue', linestyle='-', label='Media MCMC')
    axes[1, 0].set_xlabel('Alpha')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Distribución posterior - Alpha')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(lambda_burnt, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(x=lambda_verdadero, color='red', linestyle='--', label='Verdadero')
    axes[1, 1].axvline(x=media_lambda, color='black', linestyle='-', label='Media MCMC')
    axes[1, 1].set_xlabel('Lambda')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].set_title('Distribución posterior - Lambda')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
# Gráfico de dispersión conjunta con curvas de nivel
    plt.figure(figsize=(12, 10))

# Crear grid para las curvas de nivel
    alpha_grid = np.linspace(np.percentile(alpha_burnt, 1), np.percentile(alpha_burnt, 99), 100)
    lambda_grid = np.linspace(np.percentile(lambda_burnt, 1), np.percentile(lambda_burnt, 99), 100)
    X, Y = np.meshgrid(alpha_grid, lambda_grid)

# Calcular la densidad posterior en el grid 
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = np.exp(log_posterior(t, X[i,j], Y[i,j], c=1, b=1))
    contour_lines = plt.contour(X, Y, Z, levels=8, colors='black', linewidths=1, alpha=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=8)

# Muestras MCMC
    plt.scatter(alpha_burnt, lambda_burnt, alpha=0.4, s=2, color='blue', label='Muestras MCMC')


# Medias posteriores
    media_alpha = np.mean(alpha_burnt)
    media_lambda = np.mean(lambda_burnt)

    plt.xlabel('Alpha', fontsize=12)
    plt.ylabel('Lambda', fontsize=12)
    plt.title('Distribución Posterior Conjunta con Curvas de Nivel', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Análisis de autocorrelación
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Autocorrelación sin thinning
    plot_acf(alpha_burnt, lags=50, ax=axes[0, 0], title="Autocorrelación Alpha (sin thinning)")
    plot_acf(lambda_burnt, lags=50, ax=axes[1, 0], title="Autocorrelación Lambda (sin thinning)")
    
    # Aplicar thinning
    thinning = 20
    alpha_thinned = alpha_burnt[::thinning]
    lambda_thinned = lambda_burnt[::thinning]
    
    # Autocorrelación después de thinning
    plot_acf(alpha_thinned, lags=50, ax=axes[0, 1], title=f"Autocorrelación Alpha (thinning={thinning})")
    plot_acf(lambda_thinned, lags=50, ax=axes[1, 1], title=f"Autocorrelación Lambda (thinning={thinning})")
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar tamaños de muestra
    print(f"\nMuestras totales después de burn-in: {len(alpha_burnt)}")
    print(f"Muestras después de thinning (cada {thinning}): {len(alpha_thinned)}")
    
   