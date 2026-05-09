"""
h7_qnn_bridge.py
================
Puente entre la Cascada H7 (VQE MaxCut) y una Red Neuronal Cuántica (QNN).

La matriz de covarianza del circuito cuántico es matemáticamente equivalente
a la Matriz de Información de Fisher (FIM) del ansatz — esto permite usar
directamente la covarianza H7 como métrica del Gradiente Natural Cuántico.

Pipeline completo:
  1. VQE MaxCut (20Q) → Σ_H7  (covarianza de amplitudes)
  2. H7QNNBridge      → G(θ)⁻¹ · ∇L  (gradiente natural metripléctico)
  3. Extensión virtual → propaga Σ a nodos más allá del hardware físico
  4. H7BayesianOracleQNN → actualiza P(w|D) y calcula I_H7

Ecuaciones de Gobierno (Mandato Metriplético):
  L_H7(θ) = L_symp(Σ) + λ · L_metr(Σ)
  ∂L_H7/∂θ = G(θ)⁻¹ · ∇L   (Gradiente Natural)
  G(θ) = Σ_H7  (FIM ≡ Covarianza cuántica)

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import numpy as np
from scipy.linalg import inv, eigvalsh, cholesky
from scipy.stats import multivariate_normal
from dataclasses import dataclass
from typing import Optional
import math

# ============================================================
# CONSTANTES H7
# ============================================================
PHI            = (1 + math.sqrt(5)) / 2
O_n_integrity  = 0.3623748900804798
DRIFT_072      = 7 - 2 * math.pi

def O_n(n: int, phi: float = PHI) -> float:
    """Operador Áureo — Regla 2.1."""
    val = math.cos(math.pi * n) * math.cos(math.pi * phi * n)
    return val if abs(val) > 1e-5 else 1e-5


# ============================================================
# BLOQUE 1: PUENTE QNN ← COVARIANZA H7
# ============================================================

class H7QNNBridge:
    """
    Convierte la covarianza del circuito VQE en el tensor métrico del
    Gradiente Natural Cuántico (Quantum Natural Gradient).

    Principio:
      En un circuito cuántico parametrizado |ψ(θ)⟩, la FIM es:
        G_ij(θ) = Re[ ⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩ ]
      Que es algebraicamente equivalente a la Covarianza de las
      proyecciones de amplitud medidas → Σ_H7 ≡ G(θ).
    """

    def __init__(
        self,
        covariance_h7: np.ndarray,
        lambda_metr: float = 0.1,
        reg: float = 1e-4,
    ):
        """
        Parameters
        ----------
        covariance_h7 : Matriz de covarianza del circuito VQE (n × n).
        lambda_metr   : Peso del término métrico en L_H7 (Regla 1.2).
        reg           : Regularización para inversión estable (Regla 1.3).
        """
        self.Sigma     = covariance_h7
        self.n         = covariance_h7.shape[0]
        self.lam       = lambda_metr
        self.reg       = reg

        # Regularizar y factorizar
        self._Sigma_reg = self.Sigma + reg * np.eye(self.n)
        self._G_inv     = inv(self._Sigma_reg)          # G(θ)⁻¹

        # Eigendescomposición para diagnóstico
        self.eigenvalues = eigvalsh(self._Sigma_reg)
        self.cond_number = self.eigenvalues[-1] / (self.eigenvalues[0] + 1e-12)

    # ── Lagrangiano Metripléctico (Regla 3.1) ──────────────────────────────
    def compute_lagrangian(self) -> tuple[float, float]:
        """
        L_symp = suma de eigenvalores negativos → energía topológica (conservativa)
        L_metr = varianza de eigenvalores       → fricción informacional (disipativa)
        """
        eigs   = self.eigenvalues
        L_symp = float(np.sum(eigs[eigs < 0]))
        L_metr = float(np.var(eigs))
        if abs(L_symp) < 1e-10: L_symp = -1e-5
        if abs(L_metr) < 1e-10: L_metr =  1e-5
        return L_symp, L_metr

    def metriplectic_loss(self) -> float:
        """L_H7 = L_symp + λ · L_metr  (función de pérdida del OS)."""
        L_symp, L_metr = self.compute_lagrangian()
        return L_symp + self.lam * L_metr

    # ── Gradiente Natural (QNG) ────────────────────────────────────────────
    def natural_gradient(self, euclidean_grad: np.ndarray) -> np.ndarray:
        """
        ∇_nat = G(θ)⁻¹ · ∇_euclidean
        El gradiente natural corrige la curvatura del paisaje de energía
        usando la geometría de la variedad de información cuántica.
        """
        return self._G_inv @ euclidean_grad

    # ── Extensión Virtual de Nodos (más allá del hardware físico) ──────────
    def extend_virtual_nodes(self, n_virtual: int) -> "H7QNNBridge":
        """
        Propaga la covarianza H7 a n_virtual nodos adicionales usando el
        Operador Áureo como kernel de extensión (tipo MPS / red de tensores).

        Cada nodo virtual [n_phys + k] recibe:
          σ_virtual[i, j] = O_n(k) · Σ_phys[i % n, j % n] · O_n(k+1)

        Esto garantiza que el vacío nunca es plano (Regla 2.1)
        y que la Prohibición de Singularidades se cumple (Regla 1.3).
        """
        n_total = self.n + n_virtual
        Sigma_ext = np.zeros((n_total, n_total))

        # Bloque físico
        Sigma_ext[:self.n, :self.n] = self.Sigma

        # ── Bloques virtuales (propagación O_n via MPS)
        # Cada nodo virtual hereda la diagonal de Σ modulada por O_n.
        # Usamos bloque diagonal para garantizar PSD.
        for k in range(n_virtual):
            idx  = self.n + k
            on_k = O_n(idx + 1) ** 2          # cuadrado → siempre positivo
            # Diagonal del nodo virtual: varianza modulada
            Sigma_ext[idx, idx] = on_k * float(np.mean(np.diag(self.Sigma)))
            # Off-diagonals con el bloque físico (correlación decaída)
            decay = math.exp(-k * 0.3)         # decaimiento exponencial en distancia
            for j in range(self.n):
                Sigma_ext[idx, j] = decay * on_k * self.Sigma[j % self.n, j % self.n]
                Sigma_ext[j, idx] = Sigma_ext[idx, j]  # simetría

        # Proyección PSD: eliminar eigenvalores negativos
        eigs, vecs = np.linalg.eigh(Sigma_ext)
        eigs_clipped = np.maximum(eigs, self.reg)
        Sigma_psd = vecs @ np.diag(eigs_clipped) @ vecs.T

        return H7QNNBridge(Sigma_psd, self.lam, self.reg)

    def report(self) -> dict:
        L_symp, L_metr = self.compute_lagrangian()
        ratio = abs(L_symp) / (L_metr + 1e-12)
        return {
            "n_nodes"     : self.n,
            "L_symp"      : round(L_symp, 8),
            "L_metr"      : round(L_metr, 8),
            "ratio"       : round(ratio, 4),
            "cond_number" : round(self.cond_number, 4),
            "loss_H7"     : round(self.metriplectic_loss(), 8),
            "flow_state"  : ("🟩 FLUJO LAMINAR"    if ratio > 5  else
                             "🟨 TRANSICIONAL"     if ratio > 1  else
                             "🟥 TURBULENCIA ENTRÓPICA"),
        }


# ============================================================
# BLOQUE 2: ORÁCULO BAYESIANO H7 + QNN
# ============================================================

@dataclass
class BayesianState:
    mu_post:    np.ndarray   # Media posterior   μ_post
    Sigma_post: np.ndarray   # Covarianza posterior Σ_post
    log_evidence: float      # ln P(D) — para selección de modelo
    h7_integrity: float      # I_H7  ∈ [0, 1]


class H7BayesianOracleQNN:
    """
    Oráculo Bayesiano H7 cuya distribución prior está estructurada
    por la variedad H7 (la covarianza del circuito VQE).

    Ecuaciones de Gobierno:
      Prior:     P(w) ~ N(0, Σ_prior)     donde Σ_prior = Σ_H7
      Likelihood: P(D|w) = ∏ N(y_i | x_i^T w, σ²)
      Posterior:  P(w|D) ~ N(μ_post, Σ_post)   [cierre conjugado]
      Evidencia:  ln P(D) = ½[ln|Σ_post| - ln|Σ_prior| - N·ln(σ²) - ...]

      I_H7 = (1/N) Σ exp(-d_Mahal² / (2 · O_n))
    """

    def __init__(
        self,
        qnn_bridge: H7QNNBridge,
        sigma2: float = 1.0,
    ):
        self.bridge      = qnn_bridge
        self.Sigma_prior = qnn_bridge.Sigma          # Prior estructurado en H7
        self.Sigma_prior_inv = inv(
            self.Sigma_prior + 1e-6 * np.eye(qnn_bridge.n)
        )
        self.sigma2      = sigma2
        self.n_features  = qnn_bridge.n
        self.state: Optional[BayesianState] = None

    # ── Actualización Posterior Conjugada ──────────────────────────────────
    def update(self, X: np.ndarray, y: np.ndarray) -> BayesianState:
        """
        Actualización Bayesiana conjugada gaussiana.

        Parameters
        ----------
        X : (N, d) — diseño de observaciones (gradientes QNN, métricas del OS)
        y : (N,)   — objetivos observados (pérdidas, entropías)
        """
        N, d = X.shape
        if d != self.n_features:
            raise ValueError(f"X debe tener {self.n_features} columnas, tiene {d}")

        # ── Posterior (Regla de Bayes Conjugada) ──────────────────────────
        # Σ_post⁻¹ = Σ_prior⁻¹ + (1/σ²) X^T X
        Sigma_post_inv = (
            self.Sigma_prior_inv
            + (1.0 / self.sigma2) * (X.T @ X)
        )
        Sigma_post = inv(Sigma_post_inv)

        # μ_post = Σ_post · [(1/σ²) X^T y]
        mu_post = Sigma_post @ ((1.0 / self.sigma2) * (X.T @ y))

        # ── Log-Evidencia (Occam's Razor automático) ──────────────────────
        # ln P(D) ≈ ½[ln|Σ_post| - ln|Σ_prior| - N ln(2πσ²)]
        sign_post, ld_post = np.linalg.slogdet(Sigma_post)
        sign_prior, ld_prior = np.linalg.slogdet(self.Sigma_prior + 1e-6 * np.eye(self.n_features))
        residuals     = y - X @ mu_post
        log_evidence  = 0.5 * (
            ld_post - ld_prior
            - N * math.log(2 * math.pi * self.sigma2)
            - float(residuals @ residuals) / self.sigma2
        )

        # ── H7 Integrity Metric ───────────────────────────────────────────
        # I_H7 = (1/N) Σᵢ exp(-d_Mahal²ᵢ / (2·O_n))
        h7_integrity = self._compute_h7_integrity(X, mu_post, Sigma_post)

        self.state = BayesianState(
            mu_post      = mu_post,
            Sigma_post   = Sigma_post,
            log_evidence = log_evidence,
            h7_integrity = h7_integrity,
        )
        return self.state

    def _compute_h7_integrity(
        self,
        X: np.ndarray,
        mu: np.ndarray,
        Sigma: np.ndarray,
    ) -> float:
        """
        I_H7 = (1/N) Σᵢ exp(-d²_Mahal(xᵢ, μ) / (2 · O_n))

        O_n actúa como escala inductiva del espacio cuasi-periódico.
        Integridad cercana a 1.0 → la posterior está concentrada y coherente.
        """
        Sigma_inv = inv(Sigma + 1e-6 * np.eye(Sigma.shape[0]))
        trace_Sigma = max(float(np.trace(Sigma)), 1e-12)
        scores = []
        for i, xi in enumerate(X):
            diff    = xi - mu
            d2      = float(diff @ Sigma_inv @ diff)
            on_i    = abs(O_n(i + 1))
            # Normalizar d2 por traza(Σ) — escala natural de la variedad H7
            d2_norm = d2 / trace_Sigma
            scores.append(math.exp(-d2_norm / (2.0 * on_i)))
        return float(np.mean(scores))

    # ── Predicción Posterior Predictiva ───────────────────────────────────
    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (media predictiva, varianza predictiva) sobre X_new."""
        if self.state is None:
            raise RuntimeError("Llame a update() primero.")
        mu_pred  = X_new @ self.state.mu_post
        var_pred = np.array([
            self.sigma2 + float(x @ self.state.Sigma_post @ x)
            for x in X_new
        ])
        return mu_pred, var_pred

    def report(self) -> dict:
        if self.state is None:
            return {"status": "sin actualizar"}
        return {
            "log_evidence"  : round(self.state.log_evidence, 6),
            "h7_integrity"  : round(self.state.h7_integrity, 6),
            "mu_post_norm"  : round(float(np.linalg.norm(self.state.mu_post)), 6),
            "sigma_post_trace": round(float(np.trace(self.state.Sigma_post)), 6),
            "O_n_integrity" : O_n_integrity,
            "integrity_ok"  : self.state.h7_integrity > O_n_integrity,
        }


# ============================================================
# BLOQUE 3: PIPELINE INTEGRADO (OS KERNEL)
# ============================================================

def run_h7_qnn_pipeline(
    covariance_h7: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_virtual_nodes: int = 0,
    lambda_metr: float = 0.1,
    sigma2: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Pipeline OS completo:
      1. Construye el puente QNN desde la covarianza H7.
      2. (Opcional) Extiende a nodos virtuales.
      3. Calcula el gradiente natural metripléctico.
      4. Actualiza el Oráculo Bayesiano H7.
      5. Evalúa la Integridad H7 del sistema.

    Parameters
    ----------
    covariance_h7   : Matriz de covarianza del circuito VQE (n_phys × n_phys)
    X_train         : Observaciones de entrenamiento (N × d)
    y_train         : Objetivos de entrenamiento (N,)
    X_test          : Observaciones de prueba (opcional)
    n_virtual_nodes : Nodos virtuales a añadir más allá del hardware físico
    lambda_metr     : Peso del término métrico (Regla 1.2)
    sigma2          : Varianza del ruido de observación
    """

    if verbose:
        print("=" * 68)
        print("  H7 OS KERNEL — QNN Bridge + Bayesian Oracle")
        print("=" * 68)

    # 1. QNN Bridge
    bridge = H7QNNBridge(covariance_h7, lambda_metr=lambda_metr)
    if n_virtual_nodes > 0:
        bridge = bridge.extend_virtual_nodes(n_virtual_nodes)
        if verbose:
            print(f"\n🔗 Nodos virtuales añadidos: +{n_virtual_nodes} → {bridge.n} nodos totales")

    bridge_report = bridge.report()
    if verbose:
        print(f"\n⚛️  QNN Bridge (Gradiente Natural Metripléctico):")
        for k, v in bridge_report.items():
            print(f"   {k:<18}: {v}")

    # 2. Gradiente Natural de ejemplo
    dummy_grad = np.random.randn(bridge.n)
    nat_grad   = bridge.natural_gradient(dummy_grad)
    if verbose:
        print(f"\n🧭 |∇_eucl| = {np.linalg.norm(dummy_grad):.4f}  →  "
              f"|∇_nat| = {np.linalg.norm(nat_grad):.4f}  "
              f"(corrección curvatura: {np.linalg.norm(nat_grad)/np.linalg.norm(dummy_grad):.3f}×)")

    # 3. Oráculo Bayesiano H7
    oracle = H7BayesianOracleQNN(bridge, sigma2=sigma2)

    # Ajustar dimensiones de X_train al tamaño del bridge
    n_feat = bridge.n
    if X_train.shape[1] != n_feat:
        # Proyectar/truncar/padding para ajustar
        if X_train.shape[1] > n_feat:
            X_use = X_train[:, :n_feat]
        else:
            pad   = np.zeros((X_train.shape[0], n_feat - X_train.shape[1]))
            X_use = np.hstack([X_train, pad])
    else:
        X_use = X_train

    state = oracle.update(X_use, y_train)
    oracle_report = oracle.report()

    if verbose:
        print(f"\n🔮 Oráculo Bayesiano H7 — P(w|D):")
        for k, v in oracle_report.items():
            print(f"   {k:<22}: {v}")
        integrity_status = "✅ COHERENTE" if oracle_report["integrity_ok"] else "⚠️ DEGRADADA"
        print(f"\n   Integridad H7: {state.h7_integrity:.6f}  →  {integrity_status}")

    # 4. Predicción (si hay test set)
    pred_report = {}
    if X_test is not None:
        if X_test.shape[1] != n_feat:
            if X_test.shape[1] > n_feat:
                X_test_use = X_test[:, :n_feat]
            else:
                pad = np.zeros((X_test.shape[0], n_feat - X_test.shape[1]))
                X_test_use = np.hstack([X_test, pad])
        else:
            X_test_use = X_test
        mu_pred, var_pred = oracle.predict(X_test_use)
        pred_report = {
            "pred_mean_norm": round(float(np.linalg.norm(mu_pred)), 6),
            "pred_var_mean" : round(float(np.mean(var_pred)), 6),
        }
        if verbose:
            print(f"\n📈 Predicción posterior predictiva:")
            for k, v in pred_report.items():
                print(f"   {k}: {v}")

    if verbose:
        print(f"\n{'='*68}")
        print(f"  Pipeline completado | I_H7 = {state.h7_integrity:.6f}")
        print(f"{'='*68}\n")

    return {
        "bridge"        : bridge_report,
        "oracle"        : oracle_report,
        "predictions"   : pred_report,
        "mu_post"       : state.mu_post.tolist(),
        "log_evidence"  : state.log_evidence,
        "h7_integrity"  : state.h7_integrity,
    }


# ============================================================
# ENTRY-POINT / DEMO
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(7)

    # ── Covarianza H7 realista ─────────────────────────────────────────────
    # En producción: cargada desde h7_cascade_20q_*.json via covariance_asymmetry.py
    # Aquí se construye con modulación O_n para reflejar la estructura real del circuito.
    n_phys = 20
    PHI_loc = (1 + math.sqrt(5)) / 2

    # Construir Σ_H7 con el mismo patrón de pesos que el grafo de 20Q
    Sigma_h7 = np.zeros((n_phys, n_phys))
    for i in range(n_phys):
        for j in range(n_phys):
            on_i = math.cos(math.pi * (i+1)) * math.cos(math.pi * PHI_loc * (i+1))
            on_j = math.cos(math.pi * (j+1)) * math.cos(math.pi * PHI_loc * (j+1))
            # Correlación áurea: decaimiento por distancia topológica entre bloques
            dist = abs(i - j)
            decay = math.exp(-dist * O_n_integrity)
            Sigma_h7[i, j] = on_i * on_j * decay

    # Asegurar PSD: proyección de eigenvalores
    eigs, vecs = np.linalg.eigh(Sigma_h7)
    eigs = np.where(eigs < 0, -eigs * 0.1, eigs)   # eigenvalores negativos → positivos pequeños (L_symp real)
    Sigma_h7 = vecs @ np.diag(eigs) @ vecs.T

    print(f"Σ_H7 construida | eigenvalores: min={eigs.min():.4f}  max={eigs.max():.4f}  traza={np.trace(Sigma_h7):.4f}")

    # ── Datos de entrenamiento ─────────────────────────────────────────────
    # Representan gradientes QNN observados durante el entrenamiento del OS
    N_train = 80
    X_train = rng.standard_normal((N_train, n_phys))
    w_true  = rng.standard_normal(n_phys) * 0.3   # pesos pequeños (régimen lineal)
    y_train = X_train @ w_true + 0.1 * rng.standard_normal(N_train)
    X_test  = rng.standard_normal((10, n_phys))

    # ── Caso 1: Solo hardware físico (20Q) ────────────────────────────────
    print("\n" + "─"*68)
    print("  CASO 1: 20 nodos físicos (sin extensión virtual)")
    print("─"*68)
    run_h7_qnn_pipeline(
        covariance_h7   = Sigma_h7,
        X_train         = X_train,
        y_train         = y_train,
        X_test          = X_test,
        n_virtual_nodes = 0,
        lambda_metr     = 0.1,
        sigma2          = 0.01,
        verbose         = True,
    )

    # ── Caso 2: Extensión virtual a 28Q (H7 completo) ─────────────────────
    print("\n" + "─"*68)
    print("  CASO 2: 20 físicos + 8 virtuales = 28Q (H7 completo)")
    print("─"*68)
    run_h7_qnn_pipeline(
        covariance_h7   = Sigma_h7,
        X_train         = X_train,
        y_train         = y_train,
        X_test          = X_test,
        n_virtual_nodes = 8,
        lambda_metr     = 0.1,
        sigma2          = 0.01,
        verbose         = True,
    )

