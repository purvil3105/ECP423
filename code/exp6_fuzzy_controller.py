# Experiment 6: Complete fuzzy logic controller - water tank level control
# Plant : A dh/dt = q_in - a*sqrt(h)        (non-linear single tank)
# FLC   : inputs e, de  ->  output du (incremental / PI type),  q_in += du
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

# ---------------- 1. Fuzzy sets on normalised universe [-1, 1] ----------------
labels  = ["NB", "NS", "ZE", "PS", "PB"]
centres = [-1.0, -0.5, 0.0, 0.5, 1.0]
def fuzzify(x):                                   # degree in each of 5 sets
    x = np.clip(x, -1, 1)
    return np.array([trimf(x, c - 0.5, c, c + 0.5) for c in centres])

u = np.linspace(-1, 1, 201)                       # output universe
out_mf = np.array([trimf(u, c - 0.5, c, c + 0.5) for c in centres])

# ---------------- 2. Rule base  (row = e, column = de) ----------------
#            de: NB  NS  ZE  PS  PB
rule = np.array([[0,  0,  0,  1,  2],     # e = NB
                 [0,  0,  1,  2,  3],     # e = NS
                 [0,  1,  2,  3,  4],     # e = ZE
                 [1,  2,  3,  4,  4],     # e = PS
                 [2,  3,  4,  4,  4]])    # e = PB

# ---------------- 3. Inference (min-max) + centroid defuzzification ----------
def flc(e, de):
    w = np.minimum.outer(fuzzify(e), fuzzify(de))        # firing strength of 25 rules
    agg = np.zeros_like(u)
    for k in range(5):
        if np.any(rule == k):
            agg = np.maximum(agg, np.minimum(w[rule == k].max(), out_mf[k]))
    return np.sum(u * agg) / np.sum(agg) if agg.sum() > 0 else 0.0

# ---------------- 4. Closed-loop simulation ----------------
A, a = 2.0, 0.5                    # tank area (m^2), outflow coefficient
Ke, Kd, Ku = 2.0, 4.0, 0.6         # scaling gains (e, de, output)
dt, T = 0.1, 150
t = np.arange(0, T, dt)
sp = np.where(t < 75, 1.0, 1.5)    # set-point: 1.0 m, then 1.5 m at t = 75 s
dist = np.where(t >= 115, 0.1, 0)  # extra outflow (leak) at t = 115 s
h, q, e_old = 0.0, 0.0, sp[0]
H, Q = [], []
for k in range(len(t)):
    e = sp[k] - h
    de = (e - e_old) / dt
    e_old = e
    q = np.clip(q + Ku * flc(Ke * e, Kd * de) * dt, 0, 2.0)   # valve limits 0..2 m^3/s
    h = max(h + dt * (q - a * np.sqrt(h) - dist[k]) / A, 0)
    H.append(h); Q.append(q)
H, Q = np.array(H), np.array(Q)

# ---------------- 5. Performance of the first step (0 -> 1 m) ----------------
def metrics(t, y, r):
    rise = t[np.argmax(y >= 0.9 * r)] - t[np.argmax(y >= 0.1 * r)]
    os = max(0, (y.max() - r) / r * 100)
    outside = np.where(abs(y - r) > 0.02 * r)[0]
    ts = t[outside[-1]] if len(outside) else 0
    return rise, os, ts, abs(r - y[-1])

m = t < 75
rise, os, ts, ess = metrics(t[m], H[m], 1.0)
print("Rule base (rows = e, cols = de):")
print("      " + "  ".join(labels))
for i in range(5):
    print(f"  {labels[i]}  " + "  ".join(labels[j] for j in rule[i]))
print("\nFLC sample outputs (normalised):")
for e_, de_ in [(1, 0), (0.5, -0.5), (0, 0), (-0.3, 0.2), (-1, -1)]:
    print(f"  e = {e_:5.2f}, de = {de_:5.2f}  ->  du = {flc(e_, de_):6.3f}")
print("\nStep response 0 -> 1.0 m:")
print(f"  Rise time (10-90%)   = {rise:.2f} s")
print(f"  Peak overshoot       = {os:.2f} %")
print(f"  Settling time (2%)   = {ts:.2f} s")
print(f"  Steady-state error   = {ess:.4f} m")
print(f"Level at t = 110 s (set-point 1.5 m) = {H[t < 110][-1]:.3f} m")
print(f"Level at t = 149 s (after leak)      = {H[-1]:.3f} m")

# ---------------- 6. Plots ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t, sp, "k--", label="Set-point"); ax[0].plot(t, H, label="Level h")
ax[0].axvline(115, color="r", ls=":", label="Leak starts")
ax[0].set_title("Tank level response"); ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Level (m)"); ax[0].legend(); ax[0].grid()
ax[1].plot(t, Q, "g"); ax[1].set_title("Control input (inflow)")
ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("q_in (m^3/s)"); ax[1].grid()
E, D = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
Z = np.vectorize(flc)(E, D)
cs = ax[2].contourf(E, D, Z, 15, cmap="coolwarm"); plt.colorbar(cs, ax=ax[2])
ax[2].set_title("FLC control surface du(e, de)")
ax[2].set_xlabel("e (normalised)"); ax[2].set_ylabel("de (normalised)")
plt.tight_layout(); plt.savefig("outputs/exp6_flc.png", dpi=120)
print("\nPlot saved: outputs/exp6_flc.png")
