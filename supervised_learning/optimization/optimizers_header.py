import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def loss_fn(X, Y):
    return (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

minima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]

START = [4.0, 2.0]

def run(make_optimizer):
    """Run one optimizer until loss is <= .005 and return the visited (x, y) points."""
    p = torch.tensor(START, requires_grad=True)
    opt = make_optimizer([p])
    path = [p.detach().numpy().copy()]
    while abs(loss_fn(p[0], p[1])) > 0.005:
        opt.zero_grad()
        loss_fn(p[0], p[1]).backward()          # autograd computes the gradient
        opt.step()
        path.append(p.detach().numpy().copy())
    return np.array(path)

runs = {
    "SGD, η=0.02":              (lambda ps: torch.optim.SGD(ps, lr=0.02), "#7f8c8d"),
    "Momentum, η=0.02":         (lambda ps: torch.optim.SGD(ps, lr=0.02, momentum=0.9), "#e67e22"),
    "Momentum, η=0.015":        (lambda ps: torch.optim.SGD(ps, lr=0.015, momentum=0.9), "#e6a122"),
    "Momentum, η=0.01":         (lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9), "#e6cf22"),
    "RMSProp, η=0.12":          (lambda ps: torch.optim.RMSprop(ps, lr=0.12, alpha=0.9, eps=1e-8), "#27ae60"),
    "Adam, η=0.25":             (lambda ps: torch.optim.Adam(ps, lr=0.25, betas=(0.9, 0.999), eps=1e-8), "#2980b9"),
}

fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
Z = loss_fn(X, Y)
ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), colors="#b8c4d6", linewidths=0.8)

for name, (make_opt, color) in runs.items():
    path = run(make_opt)
    ax.plot(path[:, 0], path[:, 1], "-o", ms=3, lw=1.6, color=color, label=name)

ax.plot(*START, "k*", ms=14)   # start
for min_x, min_y in minima:
    ax.plot(min_x, min_y, "g*", ms=16)

ax.legend(loc="lower right", fontsize=12)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

plt.savefig("optimizers_header.png")