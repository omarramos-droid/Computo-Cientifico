import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import arviz as az
from statsmodels.graphics.tsaplots import plot_acf

def rwmh(mu, Sigma, proposal_type="normal", proposal_param=1.0, num_samples=15000, seed=123):
    """
    Random Walk Metropolis-Hastings para distribución normal bivariada.
    
    Parameters
    ----------
    mu : Vector de medias de la distribución objetivo [μ1, μ2]
        
    Sigma : Matriz de covarianza de la distribución objetivo
        
    proposal_type : Tipo de propuesta: "normal" o "uniform"
        
    proposal_param : float
        Parámetro de la propuesta:
        - Para propuesta normal: desviación estándar
        - Para propuesta uniforme: delta (tamaño del intervalo)
    num_samples :  Número total de iteraciones MCMC
       
    seed : int
        Semilla para reproducibilidad
        
    Returns
    -------
    samples :  Array con las muestras MCMC
    acceptance_rate :         Tasa de aceptación del algoritmo

    """
    np.random.seed(seed)
    
    # Inicialización desde punto extremo
    current = np.array([1000.0, 1.0])
    
    samples = np.zeros((num_samples, 2))
    acceptance_count = 0
    
    for i in range(num_samples):
        # Generar propuesta según el tipo
        if proposal_type == "normal":
            # Propuesta normal
            cov_prop = proposal_param**2 * np.eye(2)
            increment = np.random.multivariate_normal(mean=np.array([0, 0]), cov=cov_prop)
            proposed = current + increment
            
        elif proposal_type == "uniform":
            # Propuesta uniforme
            if np.isscalar(proposal_param):
                deltas = np.array([proposal_param, proposal_param])
            else:
                deltas = np.array(proposal_param)
            
            increment = np.random.uniform(-deltas[0], deltas[0], 1)[0], np.random.uniform(-deltas[1], deltas[1], 1)[0]
            proposed = current + increment
            

        # Calcular log-densidades objetivo
        log_target_current = multivariate_normal.logpdf(current, mean=mu, cov=Sigma)
        log_target_proposed = multivariate_normal.logpdf(proposed, mean=mu, cov=Sigma)
        
        # Ratio de aceptación (ambas propuestas son simétricas)
        log_acceptance_ratio = log_target_proposed - log_target_current
        
        # Criterio de aceptación/rechazo
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            current = proposed
            acceptance_count += 1
            
        samples[i, :] = current
    
    acceptance_rate = acceptance_count / num_samples
    return samples, acceptance_rate


def Series_Tiempo_Marginales(samples, mu, Sigma, burn_in, proposal_param, proposal_type):
    """
    Gráficas para las cadenas MCMC: series de tiempo y marginales.
    
    Parameters
    ----------
    samples :  muestras MCMC
    mu :         Vector de medias verdaderas [mu_1, mu_2]
    Sigma :         Matriz de covarianza verdadera
    burn_in :         Número de iteraciones descartadas como burn-in
    proposal_param :         Parámetro de la propuesta usado
    proposal_type :   Tipo de propuesta: "normal" o "uniform"
     
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    samples_burnt = samples[burn_in:]
    
    # SERIES DE TIEMPO
    
    # Traza de la cadena para x1
    axes[0, 0].plot(samples[:, 0], 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(y=mu[0], color='red', linestyle='--', label=f'Verdadero μ₁ = {mu[0]}')
    axes[0, 0].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 0].set_xlabel('Iteración')
    axes[0, 0].set_ylabel(r'$x_1$')
    axes[0, 0].set_title(f'Cadena de Markov - $x_1$ ({proposal_type} prop = {proposal_param})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Traza de la cadena para x₂
    axes[0, 1].plot(samples[:, 1], 'g-', alpha=0.7, linewidth=0.5)
    axes[0, 1].axhline(y=mu[1], color='red', linestyle='--', label=f'Verdadero μ₂ = {mu[1]}')
    axes[0, 1].axvline(x=burn_in, color='gray', linestyle=':', label='Burn-in')
    axes[0, 1].set_xlabel('Iteración')
    axes[0, 1].set_ylabel(r'$x_2$')
    axes[0, 1].set_title(f'Cadena de Markov - $x_2$ ({proposal_type} prop = {proposal_param})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # DISTRIBUCIONES MARGINALES
    
    # Histograma de la distribución marginal de x1
    x1_range = np.linspace(mu[0] - 3, mu[0] + 3, 100)
    true_marginal_1 = norm.pdf(x1_range, loc=mu[0], scale=np.sqrt(Sigma[0, 0]))
    
    axes[1, 0].hist(samples_burnt[:, 0], bins=30, density=True, alpha=0.7, 
                   color='blue', edgecolor='black', label='MCMC')
    axes[1, 0].plot(x1_range, true_marginal_1, 'r-', linewidth=2, label='Verdadero')
    axes[1, 0].axvline(x=mu[0], color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=np.mean(samples_burnt[:, 0]), color='blue', 
                      linestyle='-', alpha=0.7, label='Media MCMC')
    axes[1, 0].set_xlabel(r'$x_1$')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Distribución marginal - ' + r'$x_1$')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Histograma de la distribución marginal de x2
    x2_range = np.linspace(mu[1] - 3, mu[1] + 3, 100)
    true_marginal_2 = norm.pdf(x2_range, loc=mu[1], scale=np.sqrt(Sigma[1, 1]))
    
    axes[1, 1].hist(samples_burnt[:, 1], bins=30, density=True, alpha=0.7, 
                   color='green', edgecolor='black', label='MCMC')
    axes[1, 1].plot(x2_range, true_marginal_2, 'r-', linewidth=2, label='Verdadero')
    axes[1, 1].axvline(x=mu[1], color='red', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(samples_burnt[:, 1]), color='green', 
                      linestyle='-', alpha=0.7, label='Media MCMC')
    axes[1, 1].set_xlabel(r'$x_2$')
    axes[1, 1].set_ylabel('Densidad')
    axes[1, 1].set_title('Distribución marginal - ' + r'$x_2$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_espacio_parametros(samples, mu, Sigma, proposal_param, proposal_type, burn_in=2000):
    """
    Genera gráficos de curvas de nivel.
    
    Parameters
    ----------
    samples :  Muestras MCMC 
    mu :         Vector de medias verdaderas [mu_1, mu_2]
    Sigma :         Matriz de covarianza verdadera
    proposal_param :         Parámetro de la propuesta usado
    proposal_type :         Tipo de propuesta: "normal" o "uniform"
    burn_in :         Número de iteraciones descartadas como burn-in

    """
    samples_burnt = samples[burn_in:]
    
    # Crear malla de puntos para las curvas de nivel
    x1_grid = np.linspace(mu[0] - 3, mu[0] + 3, 50)
    x2_grid = np.linspace(mu[1] - 3, mu[1] + 3, 50)
    
    # Calcular densidad verdadera en cada punto de la malla
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    pos = np.dstack((X1, X2))
    
    true_density = multivariate_normal.pdf(pos, mean=mu, cov=Sigma)
    
    # CURVAS DE NIVEL CON MUESTRAS
    plt.figure(figsize=(12, 5))
    
    # Curvas de nivel de la distribución verdadera, algunas
    contour_main = plt.contour(X1, X2, true_density, levels=8, 
                              colors='red', linewidths=1.5, alpha=0.7)
    plt.clabel(contour_main, inline=True, fontsize=8)
    
    # Muestras MCMC superpuestas 
    plt.scatter(samples_burnt[::10, 0], samples_burnt[::10, 1], alpha=0.3, 
               s=1, color='blue', label='Muestras MCMC')
    
    # Valor verdadero
    plt.plot(mu[0], mu[1], 'ro', markersize=10, label='Media verdadera', 
             markeredgecolor='black')
    
    plt.xlabel(r'$x_1$', fontsize=12)
    plt.ylabel(r'$x_2$', fontsize=12)
    plt.title(f'Curvas de Nivel con Muestras MCMC\n({proposal_type} prop = {proposal_param})', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    mu = np.array([3.0, 5.0])
    Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    
    
    
    
    # Si quieres ver todos los casos
    # casos = [
    #     {"tipo": "normal", "param": 1.0, "nombre": "Normal σ=1.0"},
    #     {"tipo": "normal", "param": 0.1, "nombre": "Normal σ=0.1"},
    #     {"tipo": "uniform", "param": 1.0, "nombre": "Uniforme δ=1.0"},
    #     {"tipo": "uniform", "param": 10.0, "nombre": "Uniforme δ=10.0"}
    # ]
    casos = [
        {"tipo": "normal", "param": 1.0, "nombre": "Normal σ=1.0"}
      ]
    
    
    
    # casos = [
    #     {"tipo": "uniform", "param": 1.0, "nombre": "Uniforme δ=1.0"},
    #     {"tipo": "uniform", "param": 10, "nombre": "Uniforme δ=10"}
    # ]
    # casos = [
    #     {"tipo": "uniform", "param": 10, "nombre": "Uniforme δ=10"},
    # ]

    
    for caso in casos:
   
        
        # Parámetros del caso actual
        proposal_type = caso["tipo"]
        proposal_param = caso["param"]
        num_samples = 10000
        burn_in = 4_000
        seed = 123
     
        # Ejecutar RWMH
        samples, acc_rate = rwmh(mu, Sigma, proposal_type, proposal_param, num_samples, seed)
        
        samples_burnt = samples[burn_in:]
        print(f"Tasa de aceptación: {acc_rate:.3f}")
        print(f"Media MCMC x₁: {np.mean(samples_burnt[:, 0]):.3f} (verdadero: {mu[0]})")
        print(f"Media MCMC x₂: {np.mean(samples_burnt[:, 1]):.3f} (verdadero: {mu[1]})")
        print(f"Desviación MCMC x₁: {np.std(samples_burnt[:, 0]):.3f} (verdadero: {np.sqrt(Sigma[0, 0]):.3f})")
        print(f"Desviación MCMC x₂: {np.std(samples_burnt[:, 1]):.3f} (verdadero: {np.sqrt(Sigma[1, 1]):.3f})")
        print(f"Correlación MCMC: {np.corrcoef(samples_burnt.T)[0,1]:.3f} (verdadero: {Sigma[0,1]:.3f})")
        
        # Extraer muestras de x1, x2 después del burn-in
        x1_samples = samples_burnt[:, 0]   # Columna 0: x1
        x2_samples = samples_burnt[:, 1]   # Columna 1: x2
        
        # Graficar autocorrelación para x1 y x2
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Autocorrelación de las muestras completas (sin thinning)
        plot_acf(x1_samples, lags=50, ax=axes[0, 0], title="Autocorrelación x1 (sin thinning)")
        plot_acf(x2_samples, lags=50, ax=axes[1, 0], title="Autocorrelación x2 (sin thinning)")
        
        # Aplicar thinning (ubmuestreo 
        x1_thinned = x1_samples[::10]
        x2_thinned = x2_samples[::10]
        
        # Autocorrelación después de thinning
        plot_acf(x1_thinned, lags=50, ax=axes[0, 1], title="Autocorrelación x1 (thinning=10)")
        plot_acf(x2_thinned, lags=50, ax=axes[1, 1], title="Autocorrelación x2 (thinning=10)")
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar tamaños de muestra
        print(f"Muestras totales después de burn-in: {len(x1_samples)}")
        print(f"Muestras después de thinning (cada 10): {len(x1_thinned)}")
        
        Series_Tiempo_Marginales(samples, mu, Sigma, burn_in, proposal_param, proposal_type)
        graficar_espacio_parametros(samples, mu, Sigma, proposal_param, proposal_type, burn_in)
    
 