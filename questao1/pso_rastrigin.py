# ============================================================
# Implementação do Algoritmo PSO para a Função de Rastrigin
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Função de Rastrigin para vetores
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Função de Rastrigin para malha (meshgrid)
def rastrigin_mesh(X, Y):
    A = 10
    return A*2 + (X**2 - A*np.cos(2*np.pi*X)) + (Y**2 - A*np.cos(2*np.pi*Y))

# Parâmetros
Np = 30           # número de partículas
n = 2             # dimensões
max_iter = 100    # critério de parada
bounds = [-5.12, 5.12]
w, c1, c2 = 0.7, 1.5, 1.5

# Inicialização da nuvem de partículas
x = np.random.uniform(bounds[0], bounds[1], (Np, n))
v = np.zeros((Np, n))
pbest = x.copy()
pbest_val = np.array([rastrigin(p) for p in pbest])
gbest = pbest[np.argmin(pbest_val)]
gbest_val = np.min(pbest_val)

# Histórico para animação
positions_history = []

# Loop principal
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

print("Melhor posição encontrada:", gbest)
print("Valor mínimo:", gbest_val)

# Gráfico de contorno e animação
X = np.linspace(bounds[0], bounds[1], 200)
Y = np.linspace(bounds[0], bounds[1], 200)
XX, YY = np.meshgrid(X, Y)
ZZ = rastrigin_mesh(XX, YY)  # matriz 2D correta

fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(XX, YY, ZZ, levels=50, cmap='viridis')
scat = ax.scatter([], [], color='red', s=30)
ax.set_title("Distribuição das partículas (PSO)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

def update(frame):
    scat.set_offsets(positions_history[frame])
    ax.set_title(f"Iteração {frame+1}/{max_iter}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=max_iter, interval=100, blit=True)
ani.save("pso_rastrigin.png", writer='pillow')
plt.show()
