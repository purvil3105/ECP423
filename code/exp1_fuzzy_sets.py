# Experiment 1: Crisp sets, fuzzy sets, membership functions and fuzzification
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Membership functions ----------------
def trimf(x, a, b, c):                      # triangular
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def trapmf(x, a, b, c, d):                  # trapezoidal
    return np.clip(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 0, 1)

def gaussmf(x, c, s):                       # gaussian
    return np.exp(-0.5 * ((x - c) / s) ** 2)

def gbellmf(x, a, b, c):                    # generalised bell
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

def sigmf(x, a, c):                         # sigmoidal
    return 1 / (1 + np.exp(-a * (x - c)))

# ---------------- Part A : crisp set vs fuzzy set ----------------
heights = np.array([150, 160, 165, 170, 175, 180, 190])
crisp_tall = (heights >= 170).astype(int)            # sharp boundary
fuzzy_tall = trapmf(heights, 155, 185, 250, 251)     # gradual boundary

print("PART A : Crisp set vs Fuzzy set  (set 'TALL')")
print("Height(cm)  Crisp mu  Fuzzy mu")
for h, c, f in zip(heights, crisp_tall, fuzzy_tall):
    print(f"{h:8d}  {c:8d}  {f:9.3f}")

# ---------------- Part B : properties of a fuzzy set ----------------
X = np.array([1, 2, 3, 4, 5, 6, 7])
A = np.array([0.0, 0.2, 0.5, 1.0, 0.7, 0.3, 0.0])
print("\nPART B : Properties of discrete fuzzy set A")
print("A =", {int(x): float(m) for x, m in zip(X, A)})
print("Support      :", X[A > 0])
print("Core         :", X[A == 1])
print("Height       :", A.max())
print("Crossover pts:", X[A == 0.5])
print("Alpha-cut 0.3:", X[A >= 0.3])
print("Normal set?  :", A.max() == 1)

# ---------------- Part C : fuzzification of temperature ----------------
t = np.linspace(0, 40, 401)
cold = lambda x: trapmf(x, -1, 0, 10, 18)
warm = lambda x: trimf(x, 12, 20, 28)
hot  = lambda x: trapmf(x, 22, 30, 40, 41)

print("\nPART C : Fuzzification of crisp temperature")
print("Temp(C)   mu_Cold  mu_Warm  mu_Hot")
for T in [5, 15, 20, 25, 35]:
    print(f"{T:6d}   {cold(T):7.3f}  {warm(T):7.3f}  {hot(T):7.3f}")

# ---------------- Plots ----------------
x = np.linspace(0, 10, 501)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(x, trimf(x, 1, 3, 5), label="Triangular (1,3,5)")
ax[0].plot(x, trapmf(x, 3, 4, 6, 7), label="Trapezoidal (3,4,6,7)")
ax[0].plot(x, gaussmf(x, 5, 1), label="Gaussian (c=5, s=1)")
ax[0].plot(x, gbellmf(x, 1.5, 3, 7), label="Bell (1.5,3,7)")
ax[0].plot(x, sigmf(x, 3, 8), label="Sigmoid (a=3, c=8)")
ax[0].set_title("Membership functions"); ax[0].set_xlabel("x")
ax[0].set_ylabel("mu(x)"); ax[0].legend(fontsize=8); ax[0].grid()

ax[1].plot(t, cold(t), label="Cold"); ax[1].plot(t, warm(t), label="Warm")
ax[1].plot(t, hot(t), label="Hot")
ax[1].axvline(25, color="k", ls="--")
ax[1].plot([25, 25], [warm(25), hot(25)], "ro")
ax[1].set_title("Fuzzification of T = 25 C"); ax[1].set_xlabel("Temperature (C)")
ax[1].set_ylabel("mu"); ax[1].legend(); ax[1].grid()
plt.tight_layout()
plt.show()
