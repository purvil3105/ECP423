# Experiment 7: Comparison of fuzzy controller structures with a conventional PID
# Plant : G(s) = 1 / ((s + 1)(0.5 s + 1))   ->   y'' = 2u - 3y' - 2y
import numpy as np
import matplotlib.pyplot as plt

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

# ---------------- Common fuzzy core (5 x 5 rules, min-max, centroid) --------
centres = [-1.0, -0.5, 0.0, 0.5, 1.0]                 # NB NS ZE PS PB
fuzzify = lambda x: np.array([trimf(np.clip(x, -1, 1), c - 0.5, c, c + 0.5) for c in centres])
U = np.linspace(-1, 1, 201)
out_mf = np.array([trimf(U, c - 0.5, c, c + 0.5) for c in centres])
rule = np.array([[0, 0, 0, 1, 2],
                 [0, 0, 1, 2, 3],
                 [0, 1, 2, 3, 4],
                 [1, 2, 3, 4, 4],
                 [2, 3, 4, 4, 4]])

def F(e, de):
    w = np.minimum.outer(fuzzify(e), fuzzify(de))
    agg = np.zeros_like(U)
    for k in range(5):
        agg = np.maximum(agg, np.minimum(w[rule == k].max(), out_mf[k]))
    return np.sum(U * agg) / np.sum(agg) if agg.sum() > 0 else 0.0

# ---------------- Controller structures ----------------
# each returns u(e, de, dt) and keeps its own memory (integral) in s
def make(kind):
    s = {"I": 0.0}
    def ctrl(e, de, dt):
        if kind == "PID":                                # conventional PID
            s["I"] += e * dt
            return 4.0 * e + 3.0 * s["I"] + 0.8 * de
        if kind == "Fuzzy PD":                           # position form: u = Ku.F
            return 5.0 * F(e, 0.4 * de)
        if kind == "Fuzzy PI":                           # incremental form: du = Ku.F
            s["I"] += 4.0 * F(e, 1.0 * de) * dt
            return s["I"]
        if kind == "Fuzzy PD+I":                         # u = a.F + b.integral(F)
            f = F(e, 0.4 * de)
            s["I"] += f * dt
            return 5.0 * f + 4.0 * s["I"]
    return ctrl

# ---------------- Closed-loop simulation ----------------
dt, T = 0.01, 20
t = np.arange(0, T, dt)
r = np.ones_like(t)                                      # unit step set-point
d = np.where(t >= 10, 0.3, 0.0)                          # load disturbance at t = 10 s

def simulate(ctrl):
    y = yd = 0.0
    e_old = r[0] - y
    Y, Uc = [], []
    for k in range(len(t)):
        e = r[k] - y
        u = ctrl(e, (e - e_old) / dt, dt)
        e_old = e
        ydd = 2 * (u - d[k]) - 3 * yd - 2 * y
        yd += ydd * dt
        y += yd * dt
        Y.append(y); Uc.append(u)
    return np.array(Y), np.array(Uc)

def metrics(y):
    m = t < 10                                           # set-point part only
    tt, yy = t[m], y[m]
    yf = yy[-1]                                          # final (steady) value
    rise = tt[np.argmax(yy >= 0.9 * yf)] - tt[np.argmax(yy >= 0.1 * yf)]
    os = max(0, (yy.max() - yf) / yf * 100)
    out = np.where(abs(yy - yf) > 0.02 * yf)[0]
    ts = tt[out[-1]] if len(out) else 0
    ess = abs(1 - yf)
    iae = np.sum(abs(r - y)) * dt
    dip = (1 - y[t >= 10].min()) * 100                   # max deviation due to load
    return rise, os, ts, ess, iae, dip

kinds = ["PID", "Fuzzy PD", "Fuzzy PI", "Fuzzy PD+I"]
results = {k: simulate(make(k)) for k in kinds}

print("Plant: G(s) = 1/((s+1)(0.5s+1)), unit step at t=0, load 0.3 at t=10 s\n")
print(f"{'Controller':12s}{'Rise(s)':>9s}{'OS(%)':>8s}{'Ts(s)':>8s}{'Ess':>8s}"
      f"{'IAE':>8s}{'Dip(%)':>8s}")
for k in kinds:
    rise, os, ts, ess, iae, dip = metrics(results[k][0])
    print(f"{k:12s}{rise:9.2f}{os:8.2f}{ts:8.2f}{ess:8.3f}{iae:8.3f}{dip:8.2f}")
print("\nRise, overshoot and settling time are measured w.r.t. the final value.")

# ---------------- Plots ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(t, r, "k--", label="Set-point")
for k in kinds:
    ax[0].plot(t, results[k][0], label=k)
    ax[1].plot(t, results[k][1], label=k)
ax[0].axvline(10, color="gray", ls=":")
ax[0].set_title("Output response (load disturbance at t = 10 s)")
ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("y(t)"); ax[0].legend(); ax[0].grid()
ax[1].set_title("Control signal u(t)"); ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("u(t)"); ax[1].legend(); ax[1].grid()
plt.tight_layout();
plt.show()
