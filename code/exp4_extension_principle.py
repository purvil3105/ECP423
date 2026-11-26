# Experiment 4: Fuzzy extension principle
import numpy as np
import matplotlib.pyplot as plt

def extend1(xs, mu, f):
    """B(y) = max { A(x) : y = f(x) }  (one variable)"""
    out = {}
    for x, m in zip(xs, mu):
        y = f(x)
        out[y] = max(out.get(y, 0), m)
    return dict(sorted(out.items()))

def extend2(A, B, f):
    """C(z) = max { min(A(x), B(y)) : z = f(x, y) }  (two variables)"""
    out = {}
    for x, ma in A.items():
        for y, mb in B.items():
            z = f(x, y)
            out[z] = max(out.get(z, 0), min(ma, mb))
    return dict(sorted(out.items()))

# ---------------- Part A : single variable, y = x^2 ----------------
X  = [-2, -1, 0, 1, 2]
mA = [0.1, 0.4, 0.8, 0.9, 0.3]
print("PART A : y = f(x) = x^2")
print("A =", dict(zip(X, mA)))
print("B = f(A) =", extend1(X, mA, lambda x: x ** 2))

# ---------------- Part B : fuzzy arithmetic (two variables) ----------------
about2 = {1: 0.5, 2: 1.0, 3: 0.5}
about3 = {2: 0.5, 3: 1.0, 4: 0.5}
print("\nPART B : Fuzzy arithmetic using extension principle")
print("A ('about 2') =", about2)
print("B ('about 3') =", about3)
print("A + B =", extend2(about2, about3, lambda x, y: x + y))
print("A x B =", extend2(about2, about3, lambda x, y: x * y))
print("B - A =", extend2(about2, about3, lambda x, y: y - x))

# ---------------- Part C : continuous fuzzy set, y = x^2 ----------------
A_mf = lambda x: np.maximum(np.minimum((x + 1) / 2, (3 - x) / 2), 0)  # tri(-1,1,3)
x = np.linspace(-1, 3, 401)
muA = A_mf(x)
y_grid = np.linspace(0, 9, 181)
# pre-images of y under f(x) = x^2 are x = +sqrt(y) and x = -sqrt(y)
muB = np.maximum(A_mf(np.sqrt(y_grid)), A_mf(-np.sqrt(y_grid)))

print("\nPART C : Continuous A = triangle(-1, 1, 3), y = x^2")
for yv in [0, 1, 2, 4, 6, 9]:
    print(f"  mu_B({yv}) = {muB[np.argmin(abs(y_grid - yv))]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(x, muA); ax[0].set_title("Input fuzzy set A ('about 1')")
ax[0].set_xlabel("x"); ax[0].grid()
ax[1].plot(y_grid, muB, "r"); ax[1].set_title("Output fuzzy set B = f(A), f(x)=x^2")
ax[1].set_xlabel("y"); ax[1].grid()
plt.tight_layout();
plt.show()
