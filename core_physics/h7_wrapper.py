import ctypes
import numpy as np
import os

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
    def update_weights(q_weights: np.ndarray, covariance: np.ndarray, energy: float, learning_rate: float, dynamic_epsilon: float) -> float:
        # q_weights is updated in place
        q_ptr = q_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cov_ptr = covariance.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        norm_grad = _kernel.update_quaternion_weights(q_ptr, cov_ptr, energy, learning_rate, dynamic_epsilon)
        return norm_grad
