#include "metriplex_core.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double compute_golden_operator(double n) {
    // O_n = cos(π n) * cos(π φ n)
    return cos(M_PI * n) * cos(M_PI * PHI * n);
}

// Helper to compute expectation value of an operator (flattened matrix) on a state
double compute_expectation_value(const QuantumState* state, const double* Op_real, const double* Op_imag) {
    double exp_val = 0.0;
    int N = state->dimension;
    
    for (int i = 0; i < N; i++) {
        double row_real = 0.0;
        double row_imag = 0.0;
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            // (Op_real + i Op_imag) * (state_real + i state_imag)
            row_real += Op_real[idx] * state->real_parts[j] - Op_imag[idx] * state->imag_parts[j];
            row_imag += Op_real[idx] * state->imag_parts[j] + Op_imag[idx] * state->real_parts[j];
        }
        // state_star * row
        exp_val += state->real_parts[i] * row_real + state->imag_parts[i] * row_imag; // Imaginary parts cancel out if Op is Hermitian
    }
    return exp_val;
}

Lagrangian compute_lagrangian(const QuantumState* state, const double* H_real, const double* H_imag, const double* S_real, const double* S_imag) {
    Lagrangian L;
    
    // Regla 1.1: Componente Simpléctica (Energía/Conservativo)
    // d_symp = {u, H} -> Asumimos L_symp es el valor esperado de H
    double symp_val = compute_expectation_value(state, H_real, H_imag);
    
    // Regla 1.2: Componente Métrica (Entropía/Disipación)
    // d_metr = [u, S] -> Asumimos L_metr es el valor esperado de S
    double metr_val = compute_expectation_value(state, S_real, S_imag);
    
    // Aplicamos operador dorado para modular el vacío
    double O_n = compute_golden_operator(state->dimension);
    symp_val *= (1.0 + fabs(O_n));
    metr_val *= (1.0 + fabs(O_n));
    
    // Regla 1.3: Prohibición de Singularidades
    if (symp_val < 1e-5 && symp_val > -1e-5) L.L_symp = (symp_val < 0) ? -1e-5 : 1e-5;
    else L.L_symp = symp_val;
    
    if (metr_val < 1e-5 && metr_val > -1e-5) L.L_metr = (metr_val < 0) ? -1e-5 : 1e-5;
    else L.L_metr = metr_val;
    
    return L;
}

void evolve_phase(QuantumState* state, double dt, Lagrangian lagr) {
    // Evolución basada en la competencia conservativo vs disipativo
    // d(psi)/dt = d_symp + d_metr
    // En una simulación real, integraríamos con H y S completos.
    // Aquí implementamos un avance temporal simplificado de fase modulado por L_symp y decaimiento por L_metr.
    
    double phase_shift = lagr.L_symp * dt;
    double decay = exp(-fabs(lagr.L_metr) * dt); // Disipación exponencial
    
    for (int i = 0; i < state->dimension; i++) {
        double r = state->real_parts[i];
        double c = state->imag_parts[i];
        
        // Rotación de fase (Conservativo)
        double new_r = r * cos(phase_shift) - c * sin(phase_shift);
        double new_c = r * sin(phase_shift) + c * cos(phase_shift);
        
        // Decaimiento (Disipativo)
        state->real_parts[i] = new_r * decay;
        state->imag_parts[i] = new_c * decay;
    }
}

void compute_covariance(const QuantumState* state, CovarianceData* out_cov) {
    int N = state->dimension;
    // Computa pseudo-covarianzas de amplitudes
    double mean_r = 0.0, mean_c = 0.0;
    for(int i = 0; i < N; i++) {
        mean_r += state->real_parts[i];
        mean_c += state->imag_parts[i];
    }
    mean_r /= N;
    mean_c /= N;
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double cov_r = (state->real_parts[i] - mean_r) * (state->real_parts[j] - mean_r);
            double cov_c = (state->imag_parts[i] - mean_c) * (state->imag_parts[j] - mean_c);
            out_cov->matrix[i * N + j] = cov_r + cov_c; // Simplified real covariance
        }
    }
    out_cov->delta = fabs(mean_r - mean_c); // Dummy gap for illustration
}
