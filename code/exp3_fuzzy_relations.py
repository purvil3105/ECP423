# Experiment 3: Fuzzy relations, composition and approximate reasoning
import numpy as np
np.set_printoptions(precision=2, suppress=True)

def cartesian(A, B):                     # R(x,y) = min(A(x), B(y))
    return np.minimum.outer(A, B)

def max_min(R, S):                       # T(x,z) = max_y min(R(x,y), S(y,z))
    return np.max(np.minimum(R[:, :, None], S[None, :, :]), axis=1)

def max_prod(R, S):                      # T(x,z) = max_y R(x,y) * S(y,z)
    return np.max(R[:, :, None] * S[None, :, :], axis=1)

# ---------------- Part A : relation from Cartesian product ----------------
A = np.array([0.2, 1.0, 0.6])            # fuzzy set on X = {x1,x2,x3}
B = np.array([0.5, 0.9, 0.3, 1.0])       # fuzzy set on Y = {y1..y4}
print("PART A : Fuzzy relation R = A x B")
print("A =", A, "\nB =", B)
print("R = A x B =\n", cartesian(A, B))

# ---------------- Part B : composition of relations ----------------
R = np.array([[0.7, 0.5, 0.2],           # R on X x Y
              [0.8, 0.4, 0.9],
              [0.1, 0.6, 1.0]])
S = np.array([[0.9, 0.3],                # S on Y x Z
              [0.2, 0.8],
              [0.5, 0.6]])
print("\nPART B : Composition of R (X x Y) and S (Y x Z)")
print("R =\n", R, "\nS =\n", S)
print("Max-min composition  T = R o S =\n", max_min(R, S))
print("Max-product composition T = R . S =\n", max_prod(R, S))

# properties of max-min composition
Q = np.array([[0.4, 1.0], [0.7, 0.2]])   # Q on Z x W
print("Associative (RoS)oQ = Ro(SoQ) :",
      np.allclose(max_min(max_min(R, S), Q), max_min(R, max_min(S, Q))))
print("Inverse (RoS)^T = S^T o R^T   :",
      np.allclose(max_min(R, S).T, max_min(S.T, R.T)))

# ---------------- Part C : approximate reasoning (Generalised Modus Ponens) ----
# Rule : IF temperature is HIGH THEN fan speed is FAST
HIGH = np.array([0.0, 0.3, 0.7, 1.0])    # on temperature {20,25,30,35}
FAST = np.array([0.1, 0.4, 0.8, 1.0])    # on speed {500,1000,1500,2000}
R_mam = np.minimum.outer(HIGH, FAST)     # Mamdani (min) implication
R_lar = np.multiply.outer(HIGH, FAST)    # Larsen (product) implication
print("\nPART C : Approximate reasoning")
print("Rule: IF temp is HIGH THEN speed is FAST")
print("HIGH =", HIGH, "  FAST =", FAST)
print("Rule relation (Mamdani min) R =\n", R_mam)

A_new = np.array([0.0, 0.5, 1.0, 0.6])   # observed: temp is 'FAIRLY HIGH'
B_mam = max_min(A_new[None, :], R_mam)[0]
B_lar = max_min(A_new[None, :], R_lar)[0]
print("Given A' (fairly high) =", A_new)
print("Inferred B' = A' o R (Mamdani) =", B_mam)
print("Inferred B' = A' o R (Larsen)  =", B_lar)
print("Check: if A' = HIGH then B' =", max_min(HIGH[None, :], R_mam)[0], "(= FAST)")
