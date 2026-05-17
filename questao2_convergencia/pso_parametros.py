# ============================================================
# Análise dos parâmetros c1 e c2 na convergência do PSO
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Função de Rastrigin
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Função para malha (contorno)
def rastrigin_mesh(X, Y):
    A = 10
    return A*2 + (X**2 - A*np.cos(2*np.pi*X)) + (Y**2 - A*np.cos(2*np.pi*Y))

# Parâmetros gerais
Np = 30
n = 2
max_iter = 100
bounds = [-5.12, 5.12]
w = 0.7

# Combinações de c1 e c2 para análise
param_combinations = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, 0.1),
    (2, 2),
    (4, 4)
]

# Função PSO para cada par de parâmetros
def run_pso(c1, c2):
    x = np.random.uniform(bounds[0], bounds[1], (Np, n))
    v = np.zeros((Np, n))
    pbest = x.copy()
    pbest_val = np.array([rastrigin(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)
    positions_history = []

    for t in range(max_iter):
        for i in range(Np):
            if rastrigin(x[i]) < pbest_val[i]:
                pbest[i] = x[i]
                pbest_val[i] = rastrigin(x[i])
                if rastrigin(x[i]) < gbest_val:
                    gbest = x[i]
                    gbest_val = rastrigin(x[i])

            r1, r2 = np.random.rand(), np.random.rand()
            v[i] = w * v[i] + c1 * r1 * (pbest[i] - x[i]) + c2 * r2 * (gbest - x[i])
            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], bounds[0], bounds[1])

        positions_history.append(x.copy())

    print(f"c1={c1}, c2={c2} → Melhor posição: {gbest}, Valor mínimo: {gbest_val:.6f}")
    return positions_history

# Geração dos vídeos/GIFs
X = np.linspace(bounds[0], bounds[1], 200)
Y = np.linspace(bounds[0], bounds[1], 200)
XX, YY = np.meshgrid(X, Y)
ZZ = rastrigin_mesh(XX, YY)

for c1, c2 in param_combinations:
    positions_history = run_pso(c1, c2)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(XX, YY, ZZ, levels=50, cmap='viridis')
    scat = ax.scatter([], [], color='red', s=30)
    ax.set_title(f"Distribuição das partículas (c1={c1}, c2={c2})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    def update(frame):
        scat.set_offsets(positions_history[frame])
        ax.set_title(f"Iteração {frame+1}/{max_iter} | c1={c1}, c2={c2}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=max_iter, interval=100, blit=True)
    ani.save(f"pso_c1_{c1}_c2_{c2}.gif", writer='pillow', fps=10)
    plt.close(fig)
