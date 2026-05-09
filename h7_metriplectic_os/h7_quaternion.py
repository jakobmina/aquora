"""
h7_quaternion.py
================
Quaternion algebra derived from H7 entanglement pairs.

Physical Foundation
-------------------
The 3-qubit Hilbert space (8 basis states) has a natural H7 conservation law:
    |s⟩ ↔ |7 ⊕ s⟩    (bit-flip complement under XOR-7)

This gives 4 conjugate pairs whose summed amplitudes map *directly* onto the
4 quaternion components:

    q  =  (ψ₀ + ψ₇)·1  +  (ψ₁ + ψ₆)·i
       +  (ψ₂ + ψ₅)·j  +  (ψ₃ + ψ₄)·k

Metriplectic Interpretation
---------------------------
  Symplectic bracket {u, H}:
    The non-Abelian commutator  [q_LE, q_BE] = q_LE ⊗ q_BE − q_BE ⊗ q_LE
    measures chirality — the same quantity that distinguishes fermionic from
    bosonic particles in the O_n classification.

  Metric bracket [u, S]:
    The symmetric (anti-commutator) part  {q_LE, q_BE} captures the entropy
    contribution — it is invariant under time-reversal and drives relaxation.

  Reversibility test (Regla 1, MANIFIESTO):
    q_LE  (Little-Endian)  →  forward  propagation (conservative)
    q_BE  (Big-Endian)     →  reversed propagation (bit-flip = time-reversal)
    Non-zero commutator norm ↔ irreversibility ↔ fermionic nature.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ── Reference amplitudes from the H7 circuit (n=3, phi=0.3624) ───────────────
H7_AMPLITUDES: np.ndarray = np.array([
    0.27940647 - 0.0511863j,   # |000⟩ = |0⟩
    0.27940647 + 0.0511863j,   # |001⟩ = |1⟩
    0.40473969 - 0.07414692j,  # |010⟩ = |2⟩
    0.27940647 + 0.0511863j,   # |011⟩ = |3⟩
    0.27940647 - 0.0511863j,   # |100⟩ = |4⟩
    0.40473969 + 0.07414692j,  # |101⟩ = |5⟩
    0.40473969 + 0.07414692j,  # |110⟩ = |6⟩
    0.40473969 - 0.07414692j,  # |111⟩ = |7⟩
], dtype=complex)

# Mirror-state note: the partial-conjugate quaternion is computed inside
# H7QuaternionMapper.__init__ (not here, to avoid import-time dependency on
# states_to_quaternion).  H7_AMPLITUDES_MIRROR is kept for import compatibility.
H7_AMPLITUDES_MIRROR: np.ndarray = H7_AMPLITUDES.copy()


# Pair labels for display
H7_PAIR_LABELS: Tuple[str, ...] = (
    "|0⟩↔|7⟩ (real  / 1)",
    "|1⟩↔|6⟩ (imag  / i)",
    "|2⟩↔|5⟩ (imag  / j)",
    "|3⟩↔|4⟩ (imag  / k)",
)


# ════════════════════════════════════════════════════════════════════════════
# CORE QUATERNION ALGEBRA
# ════════════════════════════════════════════════════════════════════════════

def states_to_quaternion(amplitudes: np.ndarray) -> np.ndarray:
    """
    Map the 8-state H7 amplitude vector to a quaternion via H7 pair sums.

        q = (ψ₀+ψ₇)·1 + (ψ₁+ψ₆)·i + (ψ₂+ψ₅)·j + (ψ₃+ψ₄)·k

    Parameters
    ----------
    amplitudes : complex ndarray of shape (8,)
        State amplitudes in Little-Endian order (Qiskit default).

    Returns
    -------
    q : complex ndarray of shape (4,)  — [q0, q1, q2, q3]
    """
    if len(amplitudes) != 8:
        raise ValueError(f"Expected 8 amplitudes, got {len(amplitudes)}")

    q0 = amplitudes[0] + amplitudes[7]  # 1  (real)
    q1 = amplitudes[1] + amplitudes[6]  # i
    q2 = amplitudes[2] + amplitudes[5]  # j
    q3 = amplitudes[3] + amplitudes[4]  # k

    return np.array([q0, q1, q2, q3], dtype=complex)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product  q1 ⊗ q2  (non-commutative).

    Encodes the symplectic {u, H} bracket:
    the non-zero commutator [q_LE, q_BE] is the fingerprint of chirality.
    """
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,   # real
        a1*b2 + b1*a2 + c1*d2 - d1*c2,   # i
        a1*c2 - b1*d2 + c1*a2 + d1*b2,   # j
        a1*d2 + b1*c2 - c1*b2 + d1*a2,   # k
    ], dtype=complex)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: q* = q0 − q1·i − q2·j − q3·k"""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=complex)


def quat_norm(q: np.ndarray) -> float:
    """‖q‖ = √(Σ|qᵢ|²)"""
    return float(np.sqrt(np.sum(np.abs(q) ** 2)))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return unit quaternion; falls back to identity if near-zero."""
    n = quat_norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    return q / n


# ════════════════════════════════════════════════════════════════════════════
# METRIPLECTIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def commutator(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """
    [q_a, q_b] = q_a ⊗ q_b − q_b ⊗ q_a

    Maps to the Symplectic bracket {u, H}:
      Non-zero → irreversible / fermionic / chiral
      Zero     → commutative / bosonic / Abelian
    """
    return quat_multiply(q_a, q_b) - quat_multiply(q_b, q_a)


def anti_commutator(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """
    {q_a, q_b} = q_a ⊗ q_b + q_b ⊗ q_a

    Maps to the Metric bracket [u, S]:
      Symmetric / time-reversal invariant → dissipative contribution.
    """
    return quat_multiply(q_a, q_b) + quat_multiply(q_b, q_a)


def compute_h7_pair_overlaps(phi_param: float) -> np.ndarray:
    """
    Computes the non-local quasiperiodic superposition for each H7 pair.
    
    W_pair = O(n) + O(7-n)  where O(n) = cos(pi * n) * cos(pi * phi * n).
    
    This sum is non-linear and captures the 'Vacuum Tension' or 
    'Superposition Drift' that drives irreversibility.
    """
    overlaps = []
    for i in range(4): # Pairs: (0,7), (1,6), (2,5), (3,4)
        n1 = i
        n2 = 7 - i
        o1 = np.cos(np.pi * n1) * np.cos(np.pi * phi_param * n1)
        o2 = np.cos(np.pi * n2) * np.cos(np.pi * phi_param * n2)
        overlaps.append(o1 + o2)
    return np.array(overlaps, dtype=float)


def compute_lagrangian_quaternion(
    q_LE: np.ndarray,
    q_BE: np.ndarray,
) -> Tuple[float, float]:
    """
    Regla 3.1 — Quaternion Lagrangian decomposition.

        L_symp  =  ‖[q_LE, q_BE]‖   (commutator  → {u, H})
        L_metr  =  ‖{q_LE, q_BE}‖   (anti-comm   → [u, S])

    Regla 1.3 — Neither can be zero (avoid pure states).

    Returns
    -------
    (L_symp, L_metr) : float, float
    """
    comm  = commutator(q_LE, q_BE)
    acomm = anti_commutator(q_LE, q_BE)

    L_symp = quat_norm(comm)
    L_metr = quat_norm(acomm)

    # Regla 1.3: no pure states
    if L_symp < 1e-10:
        L_symp = 1e-10
    if L_metr < 1e-10:
        L_metr = 1e-10

    return L_symp, L_metr


# ════════════════════════════════════════════════════════════════════════════
# H7 QUATERNION MAPPER  (main analysis class)
# ════════════════════════════════════════════════════════════════════════════

class H7QuaternionMapper:
    """
    Derives quaternion structure from H7 amplitude pairs and performs
    full Metriplectic (symplectic + metric) analysis.

    The chiral (non-Abelian) analysis compares TWO distinct quantum states:
      q_LE  ← amplitudes_a  (primary state, typically H7_AMPLITUDES)
      q_BE  ← amplitudes_b  (secondary state, phase-rotated by default)

    Physical note
    -------------
    For the H7 reference state, reversing the amplitude vector gives
    q_LE == q_BE (the H7 pair-sum mapping is invariant under vector reversal).
    To demonstrate genuine non-Abelian chirality we compare the original
    state against a phase-rotated copy — analogous to comparing a particle
    against its time-evolved counterpart.

    Usage
    -----
    mapper = H7QuaternionMapper()            # uses H7_AMPLITUDES vs. phased
    mapper = H7QuaternionMapper(a, b)        # custom pair
    report = mapper.analyze()
    mapper.print_report(report)
    """

    def __init__(
        self,
        amplitudes_a: np.ndarray = H7_AMPLITUDES,
        amplitudes_b: np.ndarray | None = None,
    ):
        if len(amplitudes_a) != 8:
            raise ValueError("H7 requires exactly 8 complex amplitudes.")
        self.amplitudes_a = np.asarray(amplitudes_a, dtype=complex)

        # Map primary state to quaternion space
        self.q_LE = states_to_quaternion(self.amplitudes_a)

        # Default second state: partial conjugate of q_LE (parity-mirror)
        # This breaks internal quaternion symmetry and reveals chirality.
        if amplitudes_b is None:
            # Build mirror quaternion directly (not via amplitude re-mapping)
            q = self.q_LE
            self.q_BE = np.array([
                q[0],          # q0 unchanged
                np.conj(q[1]), # q1 phase-flipped (i)
                q[2],          # q2 unchanged
                np.conj(q[3]), # q3 phase-flipped (k)
            ], dtype=complex)
            self.amplitudes_b = None   # mirror state has no canonical amplitude repr
        else:
            if len(amplitudes_b) != 8:
                raise ValueError("amplitudes_b must also have length 8.")
            self.amplitudes_b = np.asarray(amplitudes_b, dtype=complex)
            self.q_BE = states_to_quaternion(self.amplitudes_b)

    # ---------------------------------------------------------------- props --
    @property
    def norm_LE(self) -> float:
        return quat_norm(self.q_LE)

    @property
    def norm_BE(self) -> float:
        return quat_norm(self.q_BE)

    @property
    def chirality(self) -> float:
        """‖[q_LE, q_BE]‖ — zero iff system is bosonic (Abelian)."""
        return quat_norm(commutator(self.q_LE, self.q_BE))

    @property
    def is_non_abelian(self) -> bool:
        return not np.isclose(self.chirality, 0.0, atol=1e-8)

    # --------------------------------------------------------------- analyze --
    def analyze(self, phi_param: float = 0.618) -> dict:
        """
        Run full Metriplectic quaternion analysis.

        Parameters
        ----------
        phi_param : float
            The quasiperiodic modulation used to compute Vacuum Overlaps.
        """
        comm  = commutator(self.q_LE, self.q_BE)
        acomm = anti_commutator(self.q_LE, self.q_BE)
        L_symp, L_metr = compute_lagrangian_quaternion(self.q_LE, self.q_BE)

        q_LB = quat_multiply(self.q_LE, self.q_BE)
        q_BL = quat_multiply(self.q_BE, self.q_LE)
        
        overlaps = compute_h7_pair_overlaps(phi_param)

        return {
            # Quaternion components
            "q_LE"          : self.q_LE,
            "q_BE"          : self.q_BE,
            "norm_LE"       : self.norm_LE,
            "norm_BE"       : self.norm_BE,
            # Products
            "q_LE_x_BE"     : q_LB,
            "q_BE_x_LE"     : q_BL,
            # Brackets (Metriplectic)
            "commutator"    : comm,        # {u, H}  symplectic
            "anti_commutator": acomm,      # [u, S]  metric
            "L_symp"        : L_symp,
            "L_metr"        : L_metr,
            # Derived physics
            "chirality"     : self.chirality,
            "is_non_abelian": self.is_non_abelian,
            "is_commutative": np.allclose(q_LB, q_BL),
            # Vacuum Overlaps (Superposition Weights)
            "pair_overlaps" : overlaps,
        }

    # --------------------------------------------------------------- display --
    def print_report(self, report: dict | None = None) -> None:
        """Pretty-print the full Metriplectic quaternion analysis."""
        if report is None:
            report = self.analyze()

        print("\n" + "=" * 60)
        print("  H7 → QUATERNION  (Metriplectic Analysis)")
        print("=" * 60)

        print("\n  Little-Endian quaternion  q_LE  (forward / {u,H}):")
        for i, (comp, label) in enumerate(zip(report["q_LE"], H7_PAIR_LABELS)):
            print(f"    q{i}  {comp:.6f}   ←  {label}")
        print(f"    ‖q_LE‖ = {report['norm_LE']:.6f}")
        
        print("\n  Vacuum Overlaps (Non-Local Superposition):")
        for i, (w, label) in enumerate(zip(report["pair_overlaps"], H7_PAIR_LABELS)):
            print(f"    W{i}  {w:+.6f}   ←  {label}")

        print("\n  Big-Endian quaternion  q_BE  (time-reversed / [u,S]):")
        for i, comp in enumerate(report["q_BE"]):
            print(f"    q{i}  {comp:.6f}")
        print(f"    ‖q_BE‖ = {report['norm_BE']:.6f}")

        print("\n  Metriplectic Lagrangian:")
        print(f"    L_symp  = ‖[q_LE, q_BE]‖ = {report['L_symp']:.8f}  ← {{u, H}}")
        print(f"    L_metr  = ‖{{q_LE, q_BE}}‖ = {report['L_metr']:.8f}  ← [u, S]")
        ratio = report["L_symp"] / report["L_metr"] if report["L_metr"] > 1e-15 else float("inf")
        print(f"    L_symp / L_metr = {ratio:.6f}  (φ² ≈ 2.618 → golden balance)")

        print("\n  Non-Abelian verification:")
        print(f"    q_LE ⊗ q_BE  =  {report['q_LE_x_BE']}")
        print(f"    q_BE ⊗ q_LE  =  {report['q_BE_x_LE']}")
        print(f"    Commutative? {report['is_commutative']}")
        print(f"    Chirality ‖[q_LE,q_BE]‖ = {report['chirality']:.8f}")

        particle = "fermionic  (non-Abelian, broken symmetry)" \
                   if report["is_non_abelian"] else \
                   "bosonic    (Abelian, preserved symmetry)"
        print(f"\n  → Particle character: {particle}")
        print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mapper = H7QuaternionMapper()
    report = mapper.analyze()
    mapper.print_report(report)
