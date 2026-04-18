import pytest
import numpy as np
from core_physics.h7_wrapper import CKernelWrapper
from h7_framework import QuaternionMetrics

def test_kernel_normalization():
    q = np.array([3.0, 4.0, 0.0, 0.0])
    q_c = CKernelWrapper.normalize(q)
    q_py = QuaternionMetrics.normalize(q)
    assert np.allclose(q_c, q_py)

def test_kernel_euler_to_quaternion():
    roll_x, pitch_y, yaw_z = 0.5, -0.3, 1.2
    q_c = CKernelWrapper.euler_to_quaternion(roll_x, pitch_y, yaw_z)
    q_py = QuaternionMetrics.euler_to_quaternion(roll_x, pitch_y, yaw_z)
    assert np.allclose(q_c, q_py)

def test_kernel_quaternion_to_euler():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    euler_c = CKernelWrapper.quaternion_to_euler(q)
    euler_py = QuaternionMetrics.quaternion_to_euler(q)
    assert np.allclose(euler_c, euler_py)

def test_kernel_update_weights():
    # Setup test vectors
    q_weights = np.array([1.0, 0.0, 0.0, 0.0])
    covariance = np.eye(4, dtype=np.float64)
    energy = -0.5
    learning_rate = 0.05
    dynamic_epsilon = 1e-4

    # Run C version
    q_c = q_weights.copy()
    norm_grad_c = CKernelWrapper.update_weights(q_c, covariance, energy, learning_rate, dynamic_epsilon)

    # Run Python version to compare
    cov_reg = covariance + dynamic_epsilon * np.eye(4)
    cov_inv = np.linalg.inv(cov_reg)
    q_gradient = np.dot(cov_inv, q_weights) * energy * learning_rate
    norm_grad_py = np.linalg.norm(q_gradient)
    
    q_py = q_weights - q_gradient
    q_py = QuaternionMetrics.normalize(q_py)

    assert np.isclose(norm_grad_c, norm_grad_py)
    assert np.allclose(q_c, q_py)
