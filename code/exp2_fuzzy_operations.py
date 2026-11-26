# Experiment 2: Operations on fuzzy sets and verification of their properties
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = ["x1", "x2", "x3", "x4", "x5", "x6"]
A = np.array([0.2, 0.5, 0.8, 1.0, 0.6, 0.1])
B = np.array([0.6, 0.3, 0.9, 0.4, 0.7, 0.0])
C = np.array([0.4, 0.8, 0.1, 0.5, 0.2, 0.9])
EMPTY, UNIV = np.zeros(6), np.ones(6)

U   = lambda p, q: np.maximum(p, q)     # union        (max)
I   = lambda p, q: np.minimum(p, q)     # intersection (min)
NOT = lambda p: 1 - p                   # complement

# ---------------- Part A : basic operations ----------------
ops = {
    "A": A, "B": B,
    "A U B  (max)": U(A, B),
    "A n B  (min)": I(A, B),
    "A'  (1-A)": NOT(A),
    "B'  (1-B)": NOT(B),
    "A - B = A n B'": I(A, NOT(B)),
    "Algebraic sum": A + B - A * B,
    "Algebraic product": A * B,
    "Bounded sum": np.minimum(1, A + B),
    "Bounded difference": np.maximum(0, A - B),
}
print("PART A : Operations on fuzzy sets")
print(f"{'Operation':22s}" + "".join(f"{x:>6s}" for x in X))
for name, v in ops.items():
    print(f"{name:22s}" + "".join(f"{m:6.2f}" for m in v))

# ---------------- Part B : verification of properties ----------------
eq = np.allclose
props = [
    ("Commutative  A U B = B U A", eq(U(A, B), U(B, A))),
    ("Commutative  A n B = B n A", eq(I(A, B), I(B, A))),
    ("Associative  (AUB)UC = AU(BUC)", eq(U(U(A, B), C), U(A, U(B, C)))),
    ("Associative  (AnB)nC = An(BnC)", eq(I(I(A, B), C), I(A, I(B, C)))),
    ("Distributive An(BUC)=(AnB)U(AnC)", eq(I(A, U(B, C)), U(I(A, B), I(A, C)))),
    ("Distributive AU(BnC)=(AUB)n(AUC)", eq(U(A, I(B, C)), I(U(A, B), U(A, C)))),
    ("Idempotent   A U A = A", eq(U(A, A), A)),
    ("Idempotent   A n A = A", eq(I(A, A), A)),
    ("Identity     A U O = A", eq(U(A, EMPTY), A)),
    ("Identity     A n X = A", eq(I(A, UNIV), A)),
    ("Absorption   A U (A n B) = A", eq(U(A, I(A, B)), A)),
    ("Involution   (A')' = A", eq(NOT(NOT(A)), A)),
    ("De Morgan    (A U B)' = A' n B'", eq(NOT(U(A, B)), I(NOT(A), NOT(B)))),
    ("De Morgan    (A n B)' = A' U B'", eq(NOT(I(A, B)), U(NOT(A), NOT(B)))),
    ("Excluded middle  A U A' = X", eq(U(A, NOT(A)), UNIV)),
    ("Contradiction    A n A' = O", eq(I(A, NOT(A)), EMPTY)),
]
print("\nPART B : Properties of fuzzy sets")
for name, ok in props:
    print(f"{name:36s} : {'HOLDS' if ok else 'DOES NOT HOLD'}")
print("\nA U A' =", U(A, NOT(A)), " (not equal to X)")
print("A n A' =", I(A, NOT(A)), " (not equal to empty set)")

# ---------------- Plot : operations on continuous fuzzy sets ----------------
x = np.linspace(0, 10, 501)
a = np.maximum(np.minimum((x - 1) / 3, (7 - x) / 3), 0)   # tri(1,4,7)
b = np.maximum(np.minimum((x - 3) / 3, (9 - x) / 3), 0)   # tri(3,6,9)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for k, (res, title) in enumerate([(U(a, b), "Union  max(A,B)"),
                                  (I(a, b), "Intersection  min(A,B)"),
                                  (NOT(a), "Complement  1 - A")]):
    ax[k].plot(x, a, "b--", label="A"); ax[k].plot(x, b, "g--", label="B")
    ax[k].fill_between(x, res, color="orange", alpha=0.5, label="Result")
    ax[k].set_title(title); ax[k].set_xlabel("x"); ax[k].legend(); ax[k].grid()
plt.tight_layout(); plt.savefig("outputs/exp2_operations.png", dpi=120)
print("\nPlot saved: outputs/exp2_operations.png")
