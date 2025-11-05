import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


def Distribucion_Objetivo(mu, sigma1, sigma2, rho):
    """
    Crea la distribución conjunta de  (X1,X2) 
    normal bivariada objetivo
    -mu=(mu_X1,mu_X2) media de la distribución
    -sigma1 varianza de X1
    -sigma2 varianza de X2
    -rho correlacion
    """
    Sigma = np.array([[sigma1**2, rho*sigma1*sigma2],
                      [rho*sigma1*sigma2, sigma2**2]])
    return multivariate_normal(mean=mu, cov=Sigma)

def Distribucion_Condicional(x_actual, mu, sigma1, sigma2, rho, variable_a_actualizar):
    """
    Calcula parámetros de las distribuciones condicionales normales
    - variable_a_actualizar = 1: X1|X2=x2
    - variable_a_actualizar = 2: X2|X1=x1  
    """
    if variable_a_actualizar == 1:  
        # Distribución condicional X1|X2
        x2 = x_actual[1]
        cond_media = mu[0] + rho * (sigma1/sigma2) * (x2 - mu[1])
        cond_var = sigma1**2 * (1 - rho**2)
    else:
        # Distribución condicional X2|X1
        x1 = x_actual[0]
        cond_media = mu[1] + rho * (sigma2/sigma1) * (x1 - mu[0])
        cond_var = sigma2**2 * (1 - rho**2)
    
    return cond_media, cond_var

def propuesta(x_actual, mu, sigma1, sigma2, rho, variable_a_actualizar):
    """
    Genera propuestas para los 3 kernels:
    - 0: Kernel de rechazo 
    - 1: Actualiza X1 manteniendo X2 fijo  
    - 2: Actualiza X2 manteniendo X1 fijo  
    """
    if variable_a_actualizar == 0:  
        return x_actual  # Kernel de recchazo
    
    # Kernels que usan distribuciones condicionales exactas
    cond_media, cond_var = Distribucion_Condicional(x_actual, mu, sigma1, sigma2, rho, variable_a_actualizar)
    
    if variable_a_actualizar == 1:
        x1_nuevo = np.random.normal(cond_media, np.sqrt(cond_var))
        return np.array([x1_nuevo, x_actual[1]])
    else:  # variable_a_actualizar == 2
        x2_nuevo = np.random.normal(cond_media, np.sqrt(cond_var))
        return np.array([x_actual[0], x2_nuevo])

def densidad_propuesta(x_actual, x_propuesta, mu, sigma1, sigma2, rho, variable_a_actualizar):
    """
    Calcula la densidad de la distribución de propuesta q(x'|x)
    """
    if variable_a_actualizar == 0:
        # Kernel de rechazo 
        return 0.0
    
    cond_media, cond_var = Distribucion_Condicional(x_actual, mu, sigma1, sigma2, rho, variable_a_actualizar)
    
    if variable_a_actualizar == 1:
        # Densidad para X1|X2
        return norm.pdf(x_propuesta[0], cond_media, np.sqrt(cond_var))
    else:  # variable_a_actualizar == 2
        # Densidad para X2|X1
        return norm.pdf(x_propuesta[1], cond_media, np.sqrt(cond_var))

def paso_metropolis_hastings(x_actual, dist_objetivo, params, variable_a_actualizar):
    """
    Paso  del algoritmo Metropolis-Hastings
    Calcula a donde se mueve la cadena según el kernell
    """
    mu, sigma1, sigma2, rho = params
    
    # Kernel de rechazo
    if variable_a_actualizar == 0:
        return x_actual, False
    
    # Para kernels normales, generar propuesta y calcular ratio de aceptación
    x_propuesta = propuesta(x_actual, mu, sigma1, sigma2, rho, variable_a_actualizar)
    
    # Ratio de aceptación de Metropolis-Hastings
    # α(x, x') = min(1, [π(x') * q(x|x')] / [π(x) * q(x'|x)])
    
    densidad_actual = dist_objetivo.pdf(x_actual)
    densidad_propuesta_obj = dist_objetivo.pdf(x_propuesta)
    
    q_propuesta_dado_actual = densidad_propuesta(x_actual, x_propuesta, mu, sigma1, sigma2, rho, variable_a_actualizar)
    q_actual_dado_propuesta = densidad_propuesta(x_propuesta, x_actual, mu, sigma1, sigma2, rho, variable_a_actualizar)
    
    # Evitar división por cero
    if q_propuesta_dado_actual == 0:
        ratio_aceptacion = 0
    else:
        ratio_aceptacion = min(1, (densidad_propuesta_obj * q_actual_dado_propuesta) / 
                              (densidad_actual * q_propuesta_dado_actual))
    #El ratio es 1, pero se calcula a modo de confirmacion
    # Decisión de aceptar/rechazar
    
    if np.random.uniform(0, 1) < ratio_aceptacion:
        return x_propuesta, True
    else:
        return x_actual, False

def muestreador_kernel_hibrido(n_muestras, x_inicial, dist_objetivo, params, pesos=[0.495, 0.495, 0.01]):
    """
    Algoritmo  de Metropolis-Hastings con kernel híbrido
    Usa 3 kernels con los pesos especificados
    """
    mu, sigma1, sigma2, rho = params
    
    muestras = np.zeros((n_muestras, 2))
    x_actual = np.array(x_inicial, dtype=float)
    tasas_aceptacion = [0, 0, 0]  # Para los 3 kernels
    conteo_kernels = [0, 0, 0]    # Cuántas veces se usó cada kernel
    
    for i in range(n_muestras):
        # Elegir kernel según los pesos: 0=rechazo, 1=actualizar X1, 2=actualizar X2
        eleccion = np.random.choice([0, 1, 2], p=pesos)
        conteo_kernels[eleccion] += 1
        
        # Ejecutar paso de Metropolis-Hastings
        x_actual, aceptado = paso_metropolis_hastings(x_actual, dist_objetivo, params, eleccion)
        
        if aceptado:
            tasas_aceptacion[eleccion] += 1
        
        muestras[i] = x_actual
    
    # Calcular tasas de aceptación normalizadas
    tasas_aceptacion_normalizadas = []
    for j in range(3):
        if conteo_kernels[j] > 0:
            tasas_aceptacion_normalizadas.append(tasas_aceptacion[j] / conteo_kernels[j])
        else:
            tasas_aceptacion_normalizadas.append(0.0)
    
    return muestras, tasas_aceptacion_normalizadas, conteo_kernels

def graficar_diagnosticos_mcmc(samples, samples_burnt, mu_verdadero, rho_verdadero, burn_in, titulo=""):
    """
    Gráficas para las cadenas MCMC: series de tiempo y marginales.
    
    Parameters
    ----------
    samples : numpy.ndarray
        Todas las muestras MCMC (incluyendo burn-in)
    samples_burnt : numpy.ndarray
        Muestras después de eliminar el burn-in
    mu_verdadero : array
        Valores verdaderos de [mu1, mu2]
    rho_verdadero : float
        Valor verdadero de rho
    burn_in : int
        Número de iteraciones descartadas como burn-in
    titulo : str
        Título para los gráficos
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SERIES DE TIEMPO
    axes[0, 0].plot(samples[:, 0], 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(y=mu_verdadero[0], color='red', linestyle='--', label='Verdadero')
    axes[0, 0].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 0].set_xlabel('Iteración')
    axes[0, 0].set_ylabel('X1')
    axes[0, 0].set_title(f'Cadena de Markov - X1 {titulo}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(samples[:, 1], 'g-', alpha=0.7, linewidth=0.5)
    axes[0, 1].axhline(y=mu_verdadero[1], color='red', linestyle='--', label='Verdadero')
    axes[0, 1].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 1].set_xlabel('Iteración')
    axes[0, 1].set_ylabel('X2')
    axes[0, 1].set_title(f'Cadena de Markov - X2 {titulo}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # DISTRIBUCIONES MARGINALES
    axes[1, 0].hist(samples_burnt[:, 0], bins=30, density=True, alpha=0.7, 
                   color='blue', edgecolor='black')
    axes[1, 0].axvline(x=mu_verdadero[0], color='red', linestyle='--', label='Verdadero')
    axes[1, 0].axvline(x=np.mean(samples_burnt[:, 0]), color='blue', 
                      linestyle='-', label='Media MCMC')
    axes[1, 0].set_xlabel('X1')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Distribución marginal - X1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(samples_burnt[:, 1], bins=30, density=True, alpha=0.7, 
                   color='green', edgecolor='black')
    axes[1, 1].axvline(x=mu_verdadero[1], color='red', linestyle='--', label='Verdadero')
    axes[1, 1].axvline(x=np.mean(samples_burnt[:, 1]), color='green', 
                      linestyle='-', label='Media MCMC')
    axes[1, 1].set_xlabel('X2')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].set_title('Distribución marginal - X2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_espacio_parametros(samples, samples_burnt, mu_verdadero, rho_verdadero, titulo=""):
    """
    Genera gráficos del espacio de parámetros: curvas de nivel y trayectoria de la cadena.
    
    Parameters
    ----------
    samples : numpy.ndarray
        Muestras MCMC 
    samples_burnt : numpy.ndarray
        Muestras después de eliminar el burn-in
    mu_verdadero : array
        Valores verdaderos de [mu1, mu2]
    rho_verdadero : float
        Valor verdadero de rho
    titulo : str
        Título para los gráficos
    """
    # Crear malla de puntos para las curvas de nivel
    x1_grid = np.linspace(-3, 3, 50)
    x2_grid = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Calcular densidad de la normal bivariada en cada punto
    pos = np.dstack((X1, X2))
    Sigma = np.array([[1, rho_verdadero], [rho_verdadero, 1]])  # σ1=σ2=1
    dist_teorica = multivariate_normal(mean=mu_verdadero, cov=Sigma)
    Z = dist_teorica.pdf(pos)
    
    # FIGURA 1: Curvas de nivel con muestras
    plt.figure(figsize=(10, 8))
    
    # Curvas de nivel teóricas
    contour_main = plt.contour(X1, X2, Z, levels=10, colors='black', 
                              linewidths=1.5, alpha=0.7)
    
    # Muestras MCMC (post burn-in)
    plt.scatter(samples_burnt[:, 0], samples_burnt[:, 1], alpha=0.3, 
               s=1, color='blue', label='Muestras MCMC')
    
    # Valor verdadero
    plt.plot(mu_verdadero[0], mu_verdadero[1], 'ro', markersize=8, 
            label='Verdadero', markeredgecolor='black')
    
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.title(f'Curvas de Nivel con Muestras MCMC {titulo}', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
  

    
      

if __name__ == "__main__":
    np.random.seed(42)
    
    # Parámetros de la simulación
    mu_verdadero = np.array([1, 1])
    sigma1, sigma2 = 1, 1
    rho =0.85
    n_muestras = 50_000
    x_inicial = np.array([2, 2])
    burn_in = 1000
    
    params = (mu_verdadero, sigma1, sigma2, rho)
    dist_objetivo = Distribucion_Objetivo(mu_verdadero, sigma1, sigma2, rho)
    
    # Usar kernel híbrido con los 3 kernels
    pesos = [0.01, 0.495,0.495] 
    # Kernel 0: rechazo,
    #Kernel 1: X1,
    #Kernel 2: X2
    
    muestras, tasas_acept, conteo_kernels = muestreador_kernel_hibrido(
        n_muestras, x_inicial, dist_objetivo, params, pesos=pesos)
    
    # Eliminar burn-in
    muestras_burnt = muestras[burn_in:]
    
    print(f"Tasas de aceptación por kernel para rho :{rho} ")
    print(f"pesos={pesos} ")
    print(f"  Kernel 0 (rechazo): {tasas_acept[0]:.3e}")
    print(f"  Kernel 1 (X1|X2): {tasas_acept[1]:.3e}")
    print(f"  Kernel 2 (X2|X1): {tasas_acept[2]:.3e}")
    print(f"Kernels usados: {conteo_kernels}")
    
    # Estadísticas
    media_empirica = np.mean(muestras_burnt, axis=0)
    cov_empirica = np.cov(muestras_burnt.T)
    print(cov_empirica)
    rho_empirico = cov_empirica[0, 1] / (np.sqrt(cov_empirica[0, 0]) * np.sqrt(cov_empirica[1, 1]))
    error_relativo_me=np.linalg.norm(media_empirica-mu_verdadero)/np.linalg.norm(mu_verdadero)*100 
    error_relativo_rho=np.linalg.norm(rho_empirico-rho)/np.linalg.norm(rho)*100 

    print(f"\nEstadísticas después de burn-in:")
    print(f"  Media empírica: {media_empirica} (verdadero: {mu_verdadero} ) ")
    print(f"Error relativo: {error_relativo_me:.4f}% " )
    print(f"  Rho empírico: {rho_empirico:.4f} (verdadero: {rho})")
    print(f"Error relativo: {error_relativo_rho:.4f}% " )

    
    # Gráficos de diagnóstico
    titulo = f"(ρ = {rho})"
    graficar_diagnosticos_mcmc(muestras, muestras_burnt, mu_verdadero, rho, burn_in, titulo)
    graficar_espacio_parametros(muestras, muestras, mu_verdadero, rho, titulo)
    
    
    # Extraer muestras de x1, x2 después del burn-in
    x1_samples = muestras_burnt[:, 0]   # Columna 0: x1
    x2_samples = muestras_burnt[:, 1]   # Columna 1: x2
    
    # Graficar autocorrelación para x1 y x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Autocorrelación de las muestras completas (sin thinning)
    plot_acf(x1_samples, lags=50, ax=axes[0, 0], title=f"Autocorrelación x1 con rho de {rho}   (sin thinning)")
    plot_acf(x2_samples, lags=50, ax=axes[1, 0], title=f"Autocorrelación x2 con rho de {rho}   (sin thinning)")
    
    thinning=45
    # Aplicar thinning (ubmuestreo 
    x1_thinned = x1_samples[::thinning]
    x2_thinned = x2_samples[::thinning]
    
    # Autocorrelación después de thinning
    plot_acf(x1_thinned, lags=50, ax=axes[0, 1], title=f"Autocorrelación x1 thinning de {thinning} con rho de {rho}")
    plot_acf(x2_thinned, lags=50, ax=axes[1, 1], title=f"Autocorrelación x2 thinning de {thinning} con rho de {rho}")
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar tamaños de muestra
    print(f"Muestras totales después de burn-in: {len(x1_samples)}")
    print(f"Muestras después de thinning (cada {thinning} ): {len(x1_thinned)}")