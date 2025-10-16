import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

# 1. Densidades y utilidades

def gamma_log_density(x, alpha, beta):
    """Log-densidad de Gamma(alpha, beta) en punto x, debe ser mayor a 0"""
    if x <= 0:
        return -np.inf
    return (alpha - 1) * np.log(x) - beta * x - gammaln(alpha) + alpha * np.log(beta)

def gamma_density(x, alpha, beta):
    """Densidad Gamma(alpha, beta) en punto x""" 
    log_val = gamma_log_density(x, alpha, beta)
    return np.exp(log_val) if log_val > -np.inf else 0.0

    # Función de log-densidad específica para Gamma
def log_f_gamma(x): 
    return gamma_log_density(x, alpha, beta)


# 2. CÁLCULO DE SECANTES E INTERSECCIONES

def calculate_secant_slope(x1, x2, log_f):
    """Pendiente de secante entre x1 y x2: (log_f(x2) - log_f(x1)) / (x2 - x1)"""
    h1, h2 = log_f(x1), log_f(x2)
    return (h2 - h1) / (x2 - x1)

def find_intersection_two_lines(x1, h1, pen1, x2, h2, pen2):
    """Intersección entre línea en (x1,h1) con pendiente pen1
    y línea en (x2,h2) con pendiente pen2"""
    if abs(pen1 - pen2) < 1e-16:
        return (x1 + x2) / 2.0
    numerator = h2 - h1 - pen2 * x2 + pen1 * x1
    return numerator / (pen1 - pen2)

# 3. CONSTRUCCIÓN DE ENVOLVENTES
# =============================================================================
def build_upper_envelope_segments(points, log_f):
    """
    Construye envolvente superior usando secantes extendidas
    Retorna: (segmentos, intersecciones)
    """

    sorted_points = sorted(points)
    n = len(sorted_points)
    
    # Calcular pendientes de secantes entre puntos consecutivos
    slopes = []
    for i in range(n - 1):
        slope = calculate_secant_slope(sorted_points[i], sorted_points[i + 1], log_f)
        slopes.append(slope)
    slopes.append(slopes[-1])  # Última pendiente para segmento final
    
    # Calcular intersecciones entre secantes consecutivas
    intersections = []
    for i in range(n - 1):
        h_i, h_ip1 = log_f(sorted_points[i]), log_f(sorted_points[i + 1])
        z = find_intersection_two_lines(
            sorted_points[i], h_i, slopes[i],
            sorted_points[i + 1], h_ip1, slopes[i + 1]
        )
        intersections.append(z)
    
    # Construir segmentos de la envolvente
    segments = []
    
    # Primer segmento: (-∞, z₀] usando primera secante extendida hacia izquierda
    segments.append({
        'start': -np.inf,
        'end': intersections[0],
        'a_i': log_f(sorted_points[0]) - slopes[0] * sorted_points[0],
        'b_i': slopes[0]
    })
    
    # Segmentos intermedios: [z_{i-1}, z_i] usando secante en punto i
    for i in range(1, n - 1):
        segments.append({
            'start': intersections[i - 1],
            'end': intersections[i], 
            'a_i': log_f(sorted_points[i]) - slopes[i] * sorted_points[i],
            'b_i': slopes[i]
        })
    
    # Último segmento: [z_{n-2}, ∞) usando última secante extendida hacia derecha
    segments.append({
        'start': intersections[-1],
        'end': np.inf,
        'a_i': log_f(sorted_points[-1]) - slopes[-1] * sorted_points[-1],
        'b_i': slopes[-1]
    })
    
    return segments, intersections

def build_lower_envelope_function(points, log_f):
    """Construye envolvente inferior usando 
    puntos conocidos por las secantes"""
    sorted_points = sorted(points)
    n = len(sorted_points)
    h_vals = [log_f(x) for x in sorted_points]
    
    # Calcular pendientes de secantes
    slopes = []
    for i in range(n - 1):
        slope = calculate_secant_slope(sorted_points[i], sorted_points[i + 1], log_f)
        slopes.append(slope)
    
    def lower_envelope(x):
        """Envolvente inferior, conectando lineas entre puntos de soporte"""
        for i in range(n - 1):
            if sorted_points[i] <= x <= sorted_points[i + 1]:
                return h_vals[i] + slopes[i] * (x - sorted_points[i])
        return -np.inf
    
    return lower_envelope

# 4. CÁLCULO DE CONSTANTES DE NORMALIZACIÓN

def calculate_segment_constant(a_i, b_i, start, end, practical_limits=(0.01, 12.0)):
    """
    Calcula c_i = ∫_{start}^{end} exp(a_i + b_i x) dx 
    """
    # Para Gamma, el soporte es (0, ∞) pero usamos límites prácticos
    actual_start = practical_limits[0] if start == -np.inf else max(start, practical_limits[0])
    actual_end = practical_limits[1] if end == np.inf else min(end, practical_limits[1])
    


  # [exp(a_i + b_i·end) - exp(a_i + b_i·start)] / b_i
    numerator = np.exp(a_i + b_i * actual_end) - np.exp(a_i + b_i * actual_start)
    result = numerator / b_i
  
    return result
 
    

def calculate_total_constant(segments, practical_limits=(0.001, 12.0)):
    """
    Calcula todas las constantes c_i y la constante total c = sum(c_i)
    """
    c_i_list = []
    for seg in segments:
        c_i = calculate_segment_constant(seg['a_i'], seg['b_i'], seg['start'], seg['end'], practical_limits)
        c_i_list.append(float(c_i))
    
    c_total = float(sum(c_i_list))
    
   
    
    return c_i_list, c_total

# 5. MUESTREO DE LA ENVOLVENTE

def build_envelope_density_function(segments, c_total, practical_limits=(0.001, 12.0)):
    """
    Construye la función de densidad de la envolvente g(x)
    """
    def g(x):
        for seg in segments:
            # Para Gamma: límites practicos
            start = practical_limits[0] if seg['start'] == -np.inf else seg['start']
            end = practical_limits[1] if seg['end'] == np.inf else seg['end']
            if start <= x <= end:
                return np.exp(seg['a_i'] + seg['b_i'] * x) / c_total
        return 0.0
    return g

def sample_from_envelope(segments, c_i_list, c_total, practical_limits=(0.001, 12.0)):
    """
    Muestrea de la mezcla de exponenciales truncadas
    """
    # 1. Seleccionar segmento según pesos proporcionales a c_i
    weights = np.array(c_i_list) / c_total
    

    seg_idx = np.random.choice(len(segments), p=weights)
    seg = segments[seg_idx]
    
    # 2. Muestrear del segmento exponencial truncado en límites Gamma
    a_i, b_i = seg['a_i'], seg['b_i']
    start = practical_limits[0] if seg['start'] == -np.inf else seg['start']
    end = practical_limits[1] if seg['end'] == np.inf else seg['end']
    
    # Asegurar límites válidos
    start = max(start, practical_limits[0])
    end = min(end, practical_limits[1])
    
    if abs(b_i) < 1e-12:
        # Caso uniforme cuando pendiente ≈ 0
        x = np.random.uniform(start, end)
    else:
        # Transformada inversa para distribución exponencial truncada
        u = np.random.uniform(0, 1)
     
        exp_b_start = np.exp(b_i * start)
        exp_b_end = np.exp(b_i * end)
        val = exp_b_start + u * (exp_b_end - exp_b_start)
        x = np.log(val) / b_i
        x = np.clip(x, start, end)
    
    
    return float(x), seg_idx


# 6. ALGORITMO ARS 
#Los valores en las funciones son solo de referencia, en la implementacion final
# se adaptan
def adaptive_rejection_sampling_secantes(log_f, initial_points, 
                                       n_samples=10000,
                                       practical_limits=(0.01, 12.0),
                                       max_adaptation_samples=20,
                                       adaptation_stop_threshold=0.95):
    """
    Algoritmo A.7 - ARS Algorithm 
    """
    S_n = sorted(initial_points.copy())
    samples = []
    evaluations_count = 0
    adaptation_active = True
    recent_acceptance_rate = 1.0
    acceptance_history = []
    
    while len(samples) < n_samples:
        # Construir envolventes
        segments, _ = build_upper_envelope_segments(S_n, log_f)
        c_i_list, ϖ_n = calculate_total_constant(segments, practical_limits)
        lower_env = build_lower_envelope_function(S_n, log_f)
        
        # 1. Generate X ~ g_n(x), U ~ U[0,1]
        X, seg_idx = sample_from_envelope(segments, c_i_list, ϖ_n, practical_limits)
        U = np.random.uniform(0, 1)
        
        # Calcular ϖ_n * g_n(X) (upper envelope en X)
        ϖ_n_g_n_X = np.exp(segments[seg_idx]['a_i'] + segments[seg_idx]['b_i'] * X)
        
        # Calcular lower_f_n(X)
        lower_f_n_X = np.exp(lower_env(X)) if lower_env(X) > -np.inf else 0.0
        
        # 2. If U ≤ lower_f_n(X) / (ϖ_n g_n(X)), accept X
        if U <= lower_f_n_X / ϖ_n_g_n_X:
            samples.append(X)
            acceptance_history.append(1)
        
        # 3. Otherwise, if U ≤ f(X) / (ϖ_n g_n(X)), accept X and update S_n
        else:
            # Evaluar f(X)
            log_f_X = log_f(X)
            f_X = np.exp(log_f_X) if log_f_X > -np.inf else 0.0
            evaluations_count += 1
            
            if U <= f_X / ϖ_n_g_n_X:
                samples.append(X)
                acceptance_history.append(1)
                
                # UPDATE S_n = S_n ∪ {X} solo si la adaptación está activa
                if adaptation_active:
                    S_n.append(X)
                    S_n.sort()
            else:
                acceptance_history.append(0)
        
        # CRITERIO PARA DEJAR DE ADAPTAR
        if adaptation_active and len(samples) >= 50:
            # Calcular tasa de aceptacion
            recent_window = min(50, len(acceptance_history))
            recent_acceptance_rate = np.mean(acceptance_history[-recent_window:])
            
            # Criterio 1: Número máximo de muestras de adaptación
            if len(samples) >= max_adaptation_samples:
                adaptation_active = False
                print(f"Adaptación detenida: alcanzado máximo de {max_adaptation_samples} muestras")
                print(f"Tasa de aceptación final: {recent_acceptance_rate:.3f}")
                print(f"Puntos de soporte finales: {len(S_n)}")
            
            # Criterio 2: Tasa de aceptación suficientemente alta y estable
            elif (recent_acceptance_rate >= adaptation_stop_threshold and 
                  len(samples) >= 100):
                # Verificar estabilidad en ventana más larga
                long_window = min(100, len(acceptance_history))
                long_term_rate = np.mean(acceptance_history[-long_window:])
                if long_term_rate >= adaptation_stop_threshold:
                    adaptation_active = False
                    print(f"Adaptación detenida: tasa de aceptación estable > {adaptation_stop_threshold}")
                    print(f"Tasa de aceptación final: {recent_acceptance_rate:.3f}")
                    print(f"Puntos de soporte finales: {len(S_n)}")
    
    return np.array(samples), S_n, acceptance_history

# 7. FUNCIONES DE VISUALIZACIÓN Y TEST

def plot_histogram_and_test(samples, alpha, beta):
    """Grafica histograma y realiza test de Kolmogorov-Smirnov"""
    
    # Crear figura
    plt.figure(figsize=(10, 6))
    
    # Histograma vs densidad teórica
    n, bins, patches = plt.hist(samples, bins=50, density=True, alpha=0.7, 
                               color='lightblue', edgecolor='black', label='Muestras ARS')
    
    # Densidad teórica
    xs = np.linspace(0.01, max(samples)*1.1, 200)
    true_pdf = [gamma_density(x, alpha, beta) for x in xs]
    plt.plot(xs, true_pdf, 'r-', lw=2, label=f'Gamma({alpha},{beta}) teórica')
    
    plt.xlabel('x')
    plt.ylabel('Densidad')
    plt.title(f'Histograma vs Densidad Teórica (n={len(samples)} muestras)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Test de Kolmogorov-Smirnov 
    from scipy import stats
    

    ks_stat, ks_pvalue = stats.kstest(samples, 'gamma', args=(alpha, 0, 1/beta))

    print(f"Test Kolmogorov-Smirnov: estadístico = {ks_stat:.4f}, p-value = {ks_pvalue:.4f}")
    print(f"Media de simulación: {np.mean(samples):.4f} ")
    print(f"Varianza de simulación: {np.var(samples):.4f} ")
    
    
# 8. FUNCIÓN PRINCIPAL 

def run_secantes_ars_with_adaptation(alpha=2.0, beta=1.0, n_samples=10000):
    """Ejemplo completo de ARS con criterio de parada para adaptación"""
    
    # Puntos iniciales 
    initial_points = [0.5, 1.0, 1.5]
    practical_limits = (0.001, 15.0)
    

    
    # Ejecutar ARS con adaptación controlada
    samples, final_points, acceptance_history = adaptive_rejection_sampling_secantes(
        log_f=log_f_gamma,
        initial_points=initial_points,
        n_samples=n_samples,
        practical_limits=practical_limits,
        max_adaptation_samples=20,  # Máximo de muestras para adaptar
        adaptation_stop_threshold=0.92  # Parar cuando tasa > 92%
    )
    
    # Calcular estadísticas de aceptación
    overall_acceptance_rate = np.mean(acceptance_history)
    print(f"\n--- ESTADÍSTICAS FINALES ---")
    print(f"Muestras totales: {len(samples)}")
    print(f"Puntos de soporte finales: {len(final_points)}")
    print(f"Tasa de aceptación global: {overall_acceptance_rate:.3f}")
    
    # Mostrar resultados
    plot_histogram_and_test(samples, alpha, beta)
    
    return samples, final_points, acceptance_history


if __name__ == "__main__":
    from scipy.stats import gamma

    alpha = 2   # shape parameter
    beta = 1    # scale parameter  
    rv = gamma(alpha, scale=beta)

 # Probabilidad en la cola superior (x > 12)
    prob_tail = 1 - rv.cdf(12)
    print(f"Probabilidad para x > 12: {prob_tail:.6f}")
    print(f"Porcentaje de masa perdida: {prob_tail*100:.4f}%")
    np.random.seed(123)
    samples, final_points, acceptance_history = run_secantes_ars_with_adaptation(
        alpha=2.0, beta=1.0, n_samples=10000
    )

