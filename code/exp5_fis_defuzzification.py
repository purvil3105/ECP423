# Experiment 5: Mamdani fuzzy inference system and defuzzification methods
# Problem: fan speed (%) from room temperature (C) and humidity (%)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def trapmf(x, a, b, c, d):
    return np.clip(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 0, 1)

# ---------------- Fuzzy sets ----------------
temp = {"Cold": lambda x: trapmf(x, -1, 0, 10, 20),
        "Warm": lambda x: trimf(x, 10, 20, 30),
        "Hot":  lambda x: trapmf(x, 20, 30, 40, 41)}
hum  = {"Low":  lambda x: trapmf(x, -1, 0, 20, 50),
        "Med":  lambda x: trimf(x, 20, 50, 80),
        "High": lambda x: trapmf(x, 50, 80, 100, 101)}
u = np.linspace(0, 100, 1001)                       # output universe
speed = {"Slow": trapmf(u, -1, 0, 20, 45),
         "Medium": trimf(u, 25, 50, 75),
         "Fast": trapmf(u, 55, 80, 100, 101)}
centre = {"Slow": 15, "Medium": 50, "Fast": 85}     # used by weighted average

# ---------------- Rule base (3 x 3) ----------------
rules = [("Cold", "Low", "Slow"), ("Cold", "Med", "Slow"), ("Cold", "High", "Medium"),
         ("Warm", "Low", "Slow"), ("Warm", "Med", "Medium"), ("Warm", "High", "Fast"),
         ("Hot", "Low", "Medium"), ("Hot", "Med", "Fast"), ("Hot", "High", "Fast")]

def infer(T, H, verbose=False):
    agg = np.zeros_like(u)
    strength = {k: 0.0 for k in speed}
    for t_lbl, h_lbl, out in rules:
        w = min(temp[t_lbl](T), hum[h_lbl](H))      # AND = min
        strength[out] = max(strength[out], w)
        agg = np.maximum(agg, np.minimum(w, speed[out]))   # min implication, max aggregation
        if verbose and w > 0:
            print(f"  IF T is {t_lbl:4s} AND H is {h_lbl:4s} THEN {out:6s}  w = {w:.3f}")
    return agg, strength

# ---------------- Defuzzification methods ----------------
def defuzz(mu, method, strength=None):
    if method == "Centroid":
        return np.sum(u * mu) / np.sum(mu)
    if method == "Bisector":
        cs = np.cumsum(mu)
        return u[np.searchsorted(cs, cs[-1] / 2)]
    if method == "Weighted avg":
        return sum(strength[k] * centre[k] for k in strength) / sum(strength.values())
    top = u[np.isclose(mu, mu.max())]               # points of maximum membership
    return {"MOM": top.mean(), "SOM": top.min(), "LOM": top.max()}[method]

methods = ["Centroid", "Bisector", "MOM", "SOM", "LOM", "Weighted avg"]

# ---------------- Detailed example ----------------
T0, H0 = 27, 65
print(f"Input: Temperature = {T0} C, Humidity = {H0} %")
print("Fuzzification:")
print("  T :", {k: round(float(f(T0)), 3) for k, f in temp.items()})
print("  H :", {k: round(float(f(H0)), 3) for k, f in hum.items()})
print("Fired rules:")
agg, st = infer(T0, H0, verbose=True)
print("Defuzzified fan speed (%):")
res = {m: defuzz(agg, m, st) for m in methods}
for m, v in res.items():
    print(f"  {m:12s} = {v:6.2f}")

# ---------------- Comparison over several inputs ----------------
print("\nComparison of defuzzification methods (fan speed %)")
print(f"{'T':>4s}{'H':>5s}" + "".join(f"{m:>14s}" for m in methods))
for T, H in [(5, 10), (15, 40), (20, 50), (27, 65), (32, 30), (38, 90)]:
    a, s = infer(T, H)
    print(f"{T:4d}{H:5d}" + "".join(f"{defuzz(a, m, s):14.2f}" for m in methods))

# ---------------- Plot ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, mf in speed.items():
    ax[0].plot(u, mf, "--", label=k)
ax[0].fill_between(u, agg, color="orange", alpha=0.6, label="Aggregated")
for m, c in zip(methods, ["r", "b", "g", "m", "c", "k"]):
    ax[0].axvline(res[m], color=c, lw=1.2, label=f"{m} = {res[m]:.1f}")
ax[0].set_title(f"Aggregated output (T={T0} C, H={H0} %)")
ax[0].set_xlabel("Fan speed (%)"); ax[0].legend(fontsize=7); ax[0].grid()

TT, HH = np.meshgrid(np.linspace(0, 40, 21), np.linspace(0, 100, 21))
Z = np.vectorize(lambda a, b: defuzz(infer(a, b)[0], "Centroid"))(TT, HH)
cs = ax[1].contourf(TT, HH, Z, 15, cmap="viridis"); plt.colorbar(cs, ax=ax[1])
ax[1].set_title("Control surface (centroid)"); ax[1].set_xlabel("Temperature (C)")
ax[1].set_ylabel("Humidity (%)")
plt.tight_layout(); plt.savefig("outputs/exp5_fis.png", dpi=120)
print("\nPlot saved: outputs/exp5_fis.png")
