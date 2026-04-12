import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Define a model with two harmonics (Fundamental + Overtones)
# This captures the "non-perfect" sine shape in your data
def periodic_model(t, p):
    # p = [Amplitude1, Amplitude2, Period, Phase1, Phase2, Offset]
    A1, A2, T, phi1, phi2, C = p
    omega = 2 * np.pi / T
    # Adding a second harmonic (2*omega) helps capture the depth/peak better
    model = A1 * np.cos(omega * t + phi1) + \
            A2 * np.cos(2 * omega * t + phi2) + C
    return model

def objective(p, t, m):
    # Adding a small penalty to A2 to keep the model from getting too "jittery"
    residuals = np.sum((periodic_model(t, p) - m)**2)
    return residuals

# 2. Setup Data (Use the binned data for a cleaner fit)
t_data = t_binned - t_binned.min()
m_data = m_binned

# 3. Initial Guesses (Crucial for periodicity)
# Looking at your plot, period is ~0.8 to 1.0 units
p0 = [
    (np.max(m_data) - np.min(m_data)) / 2, # A1
    (np.max(m_data) - np.min(m_data)) / 8, # A2 (secondary amplitude)
    0.8,                                   # Period (T)
    0.0,                                   # Phase 1
    0.0,                                   # Phase 2
    np.median(m_data)                      # Offset (C)
]

# 4. Optimize
res = minimize(objective, p0, args=(t-min(t), m), method='Nelder-Mead')

if res.success:
    best_p = res.x
    print("Fitted Period:", best_p[2])
    
    # 5. Extrapolate to 1000 days
    t_future = np.linspace(0, 1000, 50000)
    m_future = periodic_model(t_future, best_p)
    
    # 6. Plot
    plt.figure(figsize=(12, 5))
    plt.scatter(t-min(t), m, label="Data", alpha=0.5)
    plt.plot(t_future, m_future, color='red', label="Extrapolated Fit (1000 days)")
    
    # plt.xlim(0, 1000) # Show the full 1000 days
    plt.xlim(0, 30) # To see how well it fits the original data
    
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
else:
    print("Optimization failed:", res.message)