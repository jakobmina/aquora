#ifndef METRIPLEX_CORE_H
#define METRIPLEX_CORE_H

#include <complex.h>

#define PHI 1.618033988749895 // Proporción Áurea

// Estructuras de Datos Físicos
typedef struct {
    double* real_parts;
    double* imag_parts;
    int dimension;
} QuantumState;

typedef struct {
    double L_symp;
    double L_metr;
} Lagrangian;

typedef struct {
    double* matrix; // flattened array
    double delta;
    int dimension;
} CovarianceData;

// Funciones Expuestas (C API)
double compute_golden_operator(double n);
Lagrangian compute_lagrangian(const QuantumState* state, const double* H_real, const double* H_imag, const double* S_real, const double* S_imag);
void evolve_phase(QuantumState* state, double dt, Lagrangian lagr);
void compute_covariance(const QuantumState* state, CovarianceData* out_cov);

#endif // METRIPLEX_CORE_H
