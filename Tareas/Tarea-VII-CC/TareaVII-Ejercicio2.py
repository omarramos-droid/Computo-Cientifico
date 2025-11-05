import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, uniform
import arviz as az
from statsmodels.graphics.tsaplots import plot_acf

def metropolis_hastings_gamma(alpha_true, proposal_type="gamma", proposal_param=None, 
                             num_samples=15000, x0=950, seed=123):
    """
    Metropolis-Hastings para distribución Gamma(α,1) con diferentes propuestas.
    
    Parameters
    ----------
    alpha_true : Parámetro de forma verdadero de la distribución Gamma(α,1)      
    proposal_type : Tipo de propuesta: "gamma" o "uniform"        
    proposal_param : 
        Parámetro de la propuesta:
        - Para propuesta gamma: parámetro de forma (entero de alpha_true)
        - Para propuesta uniforme: delta (tamaño del intervalo)
    num_samples :        Número total de iteraciones MCMC
    x0 :        Punto inicial de la cadena
    seed :         Semilla para reproducibilidad

    Returns
    -------
    samples :         Array con  muestras MCMC
    acceptance_rate :         Tasa de aceptación del algoritmo

    """
    np.random.seed(seed)
    
    # Inicialización desde punto extremo
    current = x0
    
    samples = np.zeros(num_samples)
    acceptance_count = 0
    
    # Parámetro para la propuesta gamma
    if proposal_type == "gamma":
        proposal_param = int(alpha_true)  # Parte entera de alpha
    
    for i in range(num_samples):
        # Generar propuesta según el tipo
        if proposal_type == "gamma":
            # Propuesta Gamma([alpha], 1)
            proposed = gamma.rvs(proposal_param, scale=1)
            
            # Ratio
            log_target_current = gamma.logpdf(current, alpha_true, scale=1)
            log_target_proposed = gamma.logpdf(proposed, alpha_true, scale=1)
            
            log_prop_current = gamma.logpdf(current, proposal_param, scale=1)
            log_prop_proposed = gamma.logpdf(proposed, proposal_param, scale=1)
            
            log_acceptance_ratio = (log_target_proposed - log_target_current) - (log_prop_proposed - log_prop_current)
            
        elif proposal_type == "uniform":
            # Propuesta uniforme
            delta = proposal_param
            proposed = current + np.random.uniform(-delta, delta)
            
            # Asegurar que la propuesta esté en el soporte positivo
            if proposed <= 0:
                proposed = current  # Rechazar , pues se sale del soporte.
            # Ratio de aceptación (propuesta simétrica)
            log_target_current = gamma.logpdf(current, alpha_true, scale=1)
            log_target_proposed = gamma.logpdf(proposed, alpha_true, scale=1)
            
            log_acceptance_ratio = log_target_proposed - log_target_current
        
        # Aceptación
        if  np.log(np.random.uniform()) < log_acceptance_ratio:
            current = proposed
            acceptance_count += 1
            
        samples[i] = current
    
    acceptance_rate = acceptance_count / num_samples
    return samples, acceptance_rate

def graficar_evolucion_gamma(samples, alpha_true, burn_in, proposal_type, proposal_param):
    """
    Gráficas para la cadena MCMC de la distribución Gamma.
    
    Parameters
    ----------
    samples : numpy.ndarray
        Muestras MCMC 
    alpha_true : float
        Parámetro de la Gamma de forma verdadero  
    burn_in : int
        Iteraciones de burn-in
    proposal_type : str
        Tipo de propuesta
    proposal_param : float
        Parámetro de la propuesta
    """
    samples_burnt = samples[burn_in:]
    
    # Series de tiempo y evolución de f(X_t)
    fig1, axes1 = plt.subplots(1, 1, figsize=(12, 5))
    
 

    # Evolución de f(X_t) = densidad Gamma evaluada en X_t
    f_values = gamma.pdf(samples, alpha_true, scale=1)
    axes1.plot(f_values, 'g-', alpha=0.7, linewidth=0.5)
    axes1.axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes1.set_xlabel('Iteración')
    axes1.set_ylabel(r'$f(X_t)$')
    axes1.set_title(f'Evolución de $f(X_t)$ - {proposal_type} prop = {proposal_param}')
    axes1.legend()
    axes1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    #  Distribución marginal
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    # Histograma de la distribución marginal
    x_range = np.linspace(0, max(alpha_true * 3, 10), 100)
    true_density = gamma.pdf(x_range, alpha_true, scale=1)
    
    ax2.hist(samples_burnt, bins=50, density=True, alpha=0.7, 
             color='blue', edgecolor='black', label='MCMC')
    ax2.plot(x_range, true_density, 'r-', linewidth=2, label='Verdadero')
    ax2.axvline(x=alpha_true, color='red', linestyle='--', alpha=0.7, label='Media verdadera')
    ax2.axvline(x=np.mean(samples_burnt), color='blue', 
                linestyle='-', alpha=0.7, label='Media MCMC')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel('Densidad')
    ax2.set_title(f'Distribución marginal - {proposal_type} prop = {proposal_param}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parámetros 
    alpha_true = 3.7  # Parámetro de forma 
    x0 = 950  # Punto inicial 
 
    # casos = [
    #     {"tipo": "gamma", "param": int(alpha_true), "nombre": f"Gamma([{alpha_true}], 1)"},
    #     {"tipo": "uniform", "param": 2.0, "nombre": "Uniforme δ=2.0"}
    #     ]
    casos = [
        {"tipo": "gamma", "param": int(alpha_true), "nombre": f"Gamma([{alpha_true}], 1)"}
        ]
    # casos = [
    #     {"tipo": "uniform", "param": 2.0, "nombre": "Uniforme δ=2.0"}
    #     ]
    for caso in casos:
        print(f"PROPUESTA: {caso['nombre']}")
       
        # Parámetros del caso actual
        proposal_type = caso["tipo"]
        proposal_param = caso["param"]
        num_samples = 10_000
        burn_in = 500
        seed = 123
        
        # Ejecutar Metropolis-Hastings
        samples, acc_rate = metropolis_hastings_gamma(
            alpha_true, proposal_type, proposal_param, num_samples, x0, seed
        )
        
        samples_burnt = samples[burn_in:]
        
        # Estadísticas
        print(f"Tasa de aceptación: {acc_rate:.3f}")
        print(f"Media MCMC: {np.mean(samples_burnt):.3f} (verdadero: {alpha_true})")
        print(f"Desviación MCMC: {np.std(samples_burnt):.3f} (verdadero: {np.sqrt(alpha_true):.3f})")
        print(f"Muestras totales después de burn-in: {len(samples_burnt)}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        # Aplicar thinning
        thinning = 2
        plot_acf(samples_burnt, lags=50, ax=axes[0], title="Autocorrelación x1 (sin thinning)")
        
        # Aplicar thinning (ubmuestreo 
        samples_thinned = samples_burnt[::thinning]

        
        # Autocorrelación después de thinning
        plot_acf(samples_thinned, lags=50, ax=axes[1], title="Autocorrelación")
        
        plt.tight_layout()
        plt.show()
        print(f"Muestras después de thinning (cada {thinning}): {len(samples_thinned)}")

        # Gráficos
        graficar_evolucion_gamma(samples, alpha_true, burn_in, proposal_type, proposal_param)

        # Gráfico de convergencia de la media
        plt.figure(figsize=(10, 6))
        plt.plot(samples, 'b-', linewidth=1)
        plt.axhline(y=alpha_true, color='red', linestyle='--', 
                   label=f'Media verdadera = {alpha_true}')
        plt.xlabel('Iteración después de burn-in')
        plt.ylabel('Media acumulada')
        plt.title(f'Convergencia cadena - {caso["nombre"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Media de simulación: {np.mean(samples_burnt):.3f}")
        print(f"Probabilidad de aceptación: {acc_rate:.3f}")

