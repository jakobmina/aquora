import ctypes
import numpy as np
import os
from numpy.linalg import inv, LinAlgError

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), "h7_kernel.so")
_kernel = ctypes.CDLL(_lib_path)

# Configure argument and return types
_kernel.normalize_quaternion.argtypes = [ctypes.POINTER(ctypes.c_double)]
_kernel.quaternion_to_euler.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_kernel.euler_to_quaternion.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
_kernel.update_quaternion_weights.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
]
_kernel.update_quaternion_weights.restype = ctypes.c_double

def covariance_from_circuit_probs(
    counts: dict[str, int],
    n_qubits: int,
    reg_lambda: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduce la matriz de covarianza y su inversa desde las
    probabilidades del circuito cuántico (counts del sampler).
    """
    total = sum(counts.values())
    if total == 0:
        return np.zeros(n_qubits), np.eye(n_qubits) * reg_lambda, np.eye(n_qubits) / reg_lambda

    probs = {k: v / total for k, v in counts.items()}

    # Construir matriz de muestras ponderadas X: (n_samples, n_qubits)
    X = np.array(
        [[int(b) for b in bitstring] for bitstring in probs.keys()],
        dtype=np.float64
    )

    weights = np.array(list(probs.values()), dtype=np.float64)

    # Media ponderada
    mu = np.average(X, axis=0, weights=weights)

    # Covarianza ponderada: Sigma = sum_i w_i (x_i - mu)(x_i - mu)^T
    X_centered = X - mu
    cov = (X_centered * weights[:, None]).T @ X_centered

    # Regularización de Tikhonov
    cov_reg = cov + reg_lambda * np.eye(n_qubits)

    try:
        cov_inv = inv(cov_reg)
    except LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)

    return mu, cov_reg, cov_inv

def mahalanobis_distance(
    q_weights: np.ndarray,
    mu: np.ndarray,
    cov_inv: np.ndarray
) -> float:
    """
    Calcula la distancia de Mahalanobis proyectada.
    D_M = sqrt( (q - mu)^T * Sigma^{-1} * (q - mu) )
    """
    n = len(mu)
    # Proyección de los pesos cuaterniónicos al espacio de qubits
    q_proj = q_weights[:n] if len(q_weights) >= n else np.pad(q_weights, (0, n - len(q_weights)))
    delta = q_proj - mu
    d2 = delta @ cov_inv @ delta
    return float(np.sqrt(np.maximum(d2, 0.0)))

class CKernelWrapper:
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        q_copy = q.copy()
        q_ptr = q_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _kernel.normalize_quaternion(q_ptr)
        return q_copy

    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
        q_ptr = q.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        euler = np.zeros(3, dtype=np.float64)
        euler_ptr = euler.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _kernel.quaternion_to_euler(q_ptr, euler_ptr)
        return float(euler[0]), float(euler[1]), float(euler[2])

    @staticmethod
    def euler_to_quaternion(roll_x: float, pitch_y: float, yaw_z: float) -> np.ndarray:
        q = np.zeros(4, dtype=np.float64)
        q_ptr = q.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _kernel.euler_to_quaternion(roll_x, pitch_y, yaw_z, q_ptr)
        return q

    @staticmethod
    def update_weights(
        q_weights: np.ndarray,
        counts: dict[str, int],
        n_qubits: int,
        energy: float,
        learning_rate: float,
        dynamic_epsilon: float,
        reg_lambda: float = 1e-6
    ) -> tuple[float, float, np.ndarray]:
        """
        Actualiza los pesos cuaterniónicos usando el kernel C.
        Adapta la matriz de covarianza para que sea compatible con el kernel 4x4.
        """
        mu, cov, cov_inv = covariance_from_circuit_probs(counts, n_qubits, reg_lambda)
        mahal_dist = mahalanobis_distance(q_weights, mu, cov_inv)

        # Adaptación dimensional: Asegurar matriz 4x4 para el kernel C
        cov_4x4 = np.eye(4, dtype=np.float64) * reg_lambda
        n_min = min(n_qubits, 4)
        cov_4x4[:n_min, :n_min] = cov[:n_min, :n_min]

        # Pasamos 'cov' (no invertida) porque el kernel C la invierte internamente
        q_ptr = q_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cov_ptr = cov_4x4.flatten().astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        norm_grad = _kernel.update_quaternion_weights(
            q_ptr, cov_ptr, energy, learning_rate, dynamic_epsilon
        )

        return float(norm_grad), mahal_dist, cov_inv
