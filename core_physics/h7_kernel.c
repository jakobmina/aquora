#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Core C kernel for H7 computations (SU(2) & Metriplectic Dynamics)

void normalize_quaternion(double *q) {
    double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm < 1e-10) {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
    } else {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

// Convert quaternion to Euler angles (ZYX convention)
void quaternion_to_euler(const double *q, double *euler) {
    double w = q[0], x = q[1], y = q[2], z = q[3];

    // roll_x
    euler[0] = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    
    // pitch_y
    double sinp = 2.0 * (w * y - z * x);
    if (sinp > 1.0) sinp = 1.0;
    if (sinp < -1.0) sinp = -1.0;
    euler[1] = asin(sinp);
    
    // yaw_z
    euler[2] = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

// Convert Euler angles to quaternion
void euler_to_quaternion(double roll_x, double pitch_y, double yaw_z, double *q) {
    double cy = cos(yaw_z * 0.5);
    double sy = sin(yaw_z * 0.5);
    double cp = cos(pitch_y * 0.5);
    double sp = sin(pitch_y * 0.5);
    double cr = cos(roll_x * 0.5);
    double sr = sin(roll_x * 0.5);

    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}

// 4x4 matrix inversion using Gauss-Jordan elimination
// Returns 0 on success, 1 on singular matrix
int invert_matrix_4x4(const double m[16], double invOut[16]) {
    double inv[16], m_copy[16];
    int i, j, k;

    for (i = 0; i < 16; i++) {
        inv[i] = 0.0;
        m_copy[i] = m[i];
    }
    inv[0] = inv[5] = inv[10] = inv[15] = 1.0;

    for (i = 0; i < 4; i++) {
        // Pivot
        double max_val = fabs(m_copy[i * 4 + i]);
        int pivot = i;
        for (j = i + 1; j < 4; j++) {
            if (fabs(m_copy[j * 4 + i]) > max_val) {
                max_val = fabs(m_copy[j * 4 + i]);
                pivot = j;
            }
        }
        
        if (max_val < 1e-12) {
            return 1; // Singular
        }
        
        if (pivot != i) {
            for (k = 0; k < 4; k++) {
                double t1 = m_copy[i * 4 + k];
                m_copy[i * 4 + k] = m_copy[pivot * 4 + k];
                m_copy[pivot * 4 + k] = t1;
                
                double t2 = inv[i * 4 + k];
                inv[i * 4 + k] = inv[pivot * 4 + k];
                inv[pivot * 4 + k] = t2;
            }
        }
        
        double f = m_copy[i * 4 + i];
        for (k = 0; k < 4; k++) {
            m_copy[i * 4 + k] /= f;
            inv[i * 4 + k] /= f;
        }
        
        for (j = 0; j < 4; j++) {
            if (j != i) {
                double f2 = m_copy[j * 4 + i];
                for (k = 0; k < 4; k++) {
                    m_copy[j * 4 + k] -= m_copy[i * 4 + k] * f2;
                    inv[j * 4 + k] -= inv[i * 4 + k] * f2;
                }
            }
        }
    }
    
    for (i = 0; i < 16; i++) {
        invOut[i] = inv[i];
    }
    return 0;
}

// Compute Mahalanobis gradient step
double update_quaternion_weights(double *q_weights, const double *covariance, double energy, double learning_rate, double dynamic_epsilon) {
    double cov_reg[16];
    double inv_cov[16];
    
    for (int i = 0; i < 16; i++) {
        cov_reg[i] = covariance[i];
        if (i % 5 == 0) cov_reg[i] += dynamic_epsilon;
    }
    
    if (invert_matrix_4x4(cov_reg, inv_cov) != 0) {
        // Fallback if inversion fails (shouldn't happen with regularization)
        for (int i = 0; i < 16; i++) {
            inv_cov[i] = (i % 5 == 0) ? 1.0 : 0.0;
        }
    }
    
    double q_gradient[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            q_gradient[i] += inv_cov[i * 4 + j] * q_weights[j];
        }
        q_gradient[i] *= energy * learning_rate;
    }
    
    double norm_grad = sqrt(q_gradient[0]*q_gradient[0] + q_gradient[1]*q_gradient[1] + 
                            q_gradient[2]*q_gradient[2] + q_gradient[3]*q_gradient[3]);
                            
    for (int i = 0; i < 4; i++) {
        q_weights[i] -= q_gradient[i];
    }
    
    normalize_quaternion(q_weights);
    return norm_grad;
}
