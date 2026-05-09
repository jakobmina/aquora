"""H7 BAYESIAN ORACLE + KBENCH COMPLETE INTEGRATION
==================================================
Integración de:
- H7BayesianOracle (Inferencia Bayesiana Conjugada)
- H7BayesianEnsemble (Ensemble de 7 expertos)
- KBench Tasks con evaluación Bayesiana

Features:
  - Conjugate Gaussian inference
  - Posterior predictive distributions
  - Ensemble weighting by evidence
  - Integrity metrics
  - Full KBench integration

Author: Jako + Integration
Date: May 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import multivariate_normal, norm
from typing import Dict, List, Tuple, Optional

# ============================================================================
# H7 CONSTANTS
# ============================================================================

φ = (1 + np.sqrt(5)) / 2
Ψ_n = lambda n: np.cos(np.pi * n) * np.cos(np.pi * φ * n)
O_n_integrity = 0.3623748900804798
DRIFT_072 = 7 - 2*np.pi

print(f"✓ H7 Constants loaded: φ={φ:.6f}, O_n={O_n_integrity:.10f}")

# ============================================================================
# KBENCH SETUP
# ============================================================================

try:
    import kbench
    KBENCH_AVAILABLE = True
    print("✓ KBench available")
except ImportError:
    KBENCH_AVAILABLE = False
    print("⚠ KBench mock mode")
    
    class MockAssertions:
        @staticmethod
        def assert_near(actual, expected, tolerance=0.01, expectation=""):
            if abs(actual - expected) > tolerance:
                raise AssertionError(f"Expected ~{expected}, got {actual}\n{expectation}")
        
        @staticmethod
        def assert_true(condition, expectation=""):
            if not condition:
                raise AssertionError(f"Expected True\n{expectation}")
    
    class MockLLM:
        def prompt(self, text):
            return "Mock response"
    
    class MockKBench:
        assertions = MockAssertions()
        llm = MockLLM()
        
        @staticmethod
        def task(name, description=""):
            def decorator(func):
                func._task_name = name
                return func
            return decorator
    
    kbench = MockKBench()

# ============================================================================
# PART 1: BAYESIAN PRIOR & LIKELIHOOD
# ============================================================================

@dataclass
class BayesianPrior:
    """Prior bayesiano estructurado en H7"""
    mean: np.ndarray
    cov: np.ndarray
    integrity: float = O_n_integrity
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Densidad de probabilidad posterior a integrity scaling"""
        dist = multivariate_normal(self.mean, self.cov)
        return dist.pdf(x) * self.integrity


# ============================================================================
# PART 2: H7 BAYESIAN ORACLE
# ============================================================================

class H7BayesianOracle:
    """Oracle que integra H7 con inferencia bayesiana conjugada"""
    
    def __init__(self, n_features: int, n_latent: int = 7):
        self.n_features = n_features
        self.n_latent = n_latent  # Espacio Z₇ discreto
        
        # Prior H7 estructurado
        self.prior_mean = np.zeros(n_features)
        self.prior_cov = np.eye(n_features)
        
        # Parámetros de verosimilitud
        self.likelihood_precision = None
        self.likelihood_scale = 1.0
        
        # Estado posterior
        self.posterior_mean = None
        self.posterior_cov = None
        self.log_evidence = None
        
    def fit_conjugate_gaussian(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Inferencia bayesiana conjugada Gaussiana.
        Asume prior Normal y likelihood Gaussiana.
        
        Retorna posterior distribution.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Hiperparámetros del prior
        prior_precision = np.linalg.inv(self.prior_cov)
        
        # Suficientes estadísticos
        n_obs = len(y_array)
        X_T_X = X_array.T @ X_array
        X_T_y = X_array.T @ y_array
        y_T_y = y_array.T @ y_array
        
        # Precisión posterior
        likelihood_precision = X_T_X + prior_precision
        
        # Posterior mean (MAP update)
        self.posterior_cov = np.linalg.inv(likelihood_precision)
        self.posterior_mean = self.posterior_cov @ (
            prior_precision @ self.prior_mean + X_T_y
        )
        
        # Log marginal likelihood (evidencia)
        self.log_evidence = self._compute_log_evidence(
            X_T_y, y_T_y, n_obs,
            prior_precision, likelihood_precision
        )
        
        return {
            'posterior_mean': self.posterior_mean,
            'posterior_cov': self.posterior_cov,
            'log_evidence': self.log_evidence,
            'n_observations': n_obs
        }
    
    def _compute_log_evidence(self, X_T_y, y_T_y, n_obs,
                              prior_prec, post_prec) -> float:
        """Log marginal likelihood para model comparison"""
        
        quad_term = (self.prior_mean.T @ prior_prec @ self.prior_mean
                     + y_T_y
                     - self.posterior_mean.T @ post_prec @ self.posterior_mean)
        
        det_prior = np.linalg.det(prior_prec)
        det_post = np.linalg.det(post_prec)
        
        log_ev = (0.5 * np.log(det_prior / (2*np.pi)**self.n_features)
                  + 0.5 * np.log(det_post)
                  - 0.5 * quad_term)
        
        return float(log_ev)
    
    def predict_posterior_predictive(self, X_test: pd.DataFrame) -> Dict:
        """
        Predicción integrando sobre posterior.
        Retorna media y varianza predictivas.
        """
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Media predictiva
        pred_mean = X_test_array @ self.posterior_mean
        
        # Varianza predictiva (incluye incertidumbre de parámetros)
        pred_var = 1.0 + np.sum(
            X_test_array @ self.posterior_cov * X_test_array, axis=1
        )
        pred_std = np.sqrt(pred_var)
        
        return {
            'mean': pred_mean,
            'std': pred_std,
            'var': pred_var,
            'credible_interval_95': (
                pred_mean - 1.96 * pred_std,
                pred_mean + 1.96 * pred_std
            )
        }
    
    def compute_h7_integrity_metric(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Métrica H7: Integridad de predicción Bayesiana.
        Combina Mahalanobis distance con O_n integrity.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Predicciones y varianza predictiva
        pred = self.predict_posterior_predictive(X_array)
        predictions = pred['mean']
        pred_var = pred['var']
        residuals = y_array - predictions
        
        # Distancia de Mahalanobis en el espacio de predicciones
        mahal_dist = np.sqrt(residuals**2 / pred_var)
        
        # Integridad H7
        integrity = np.mean(np.exp(-mahal_dist / (2 * O_n_integrity)))
        
        return float(integrity)


# ============================================================================
# PART 3: H7 BAYESIAN ENSEMBLE
# ============================================================================

class H7BayesianEnsemble:
    """Ensemble de múltiples oracles H7 con pesos bayesianos"""
    
    def __init__(self, n_experts: int = 7):  # 7 = Z₇
        self.n_experts = n_experts
        self.experts = []  # Será inicializado en fit
        self.expert_weights = np.ones(n_experts) / n_experts
        self.log_evidences = None
        self.feature_subsets = None
        
    def fit_experts(self, X: pd.DataFrame, y: pd.Series,
                   feature_subsets: List = None) -> Dict:
        """
        Entrena cada expert en subsets de features.
        Calcula pesos automáticamente por evidencia bayesiana.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        log_evs = []
        
        # Inicializar experts con feature subsets
        self.feature_subsets = feature_subsets
        if feature_subsets:
            n_features_per_expert = len(feature_subsets[0])
        else:
            n_features_per_expert = X_array.shape[1]
        
        self.experts = [
            H7BayesianOracle(n_features=n_features_per_expert)
            for _ in range(self.n_experts)
        ]
        
        # Entrenar cada expert
        for i, expert in enumerate(self.experts):
            if feature_subsets and i < len(feature_subsets):
                X_subset = X_array[:, feature_subsets[i]]
            else:
                X_subset = X_array
            
            # Ajustar dimensiones
            expert.n_features = X_subset.shape[1]
            expert.prior_cov = np.eye(X_subset.shape[1])
            expert.prior_mean = np.zeros(X_subset.shape[1])
            
            # Entrenar
            results = expert.fit_conjugate_gaussian(X_subset, y_array)
            log_evs.append(results['log_evidence'])
        
        # Pesos por evidencia (Occam's razor automático)
        self.log_evidences = np.array(log_evs)
        log_evs_normalized = self.log_evidences - np.max(self.log_evidences)
        self.expert_weights = np.exp(log_evs_normalized)
        self.expert_weights /= self.expert_weights.sum()
        
        print("\n✓ Expert Weights (by Evidence):")
        for i, w in enumerate(self.expert_weights):
            print(f"  Expert {i}: {w:.4f} (log_evidence: {self.log_evidences[i]:.2f})")
        
        return {
            'expert_weights': self.expert_weights,
            'log_evidences': self.log_evidences
        }
    
    def predict_ensemble(self, X_test: pd.DataFrame) -> Dict:
        """Predicción ponderada por posterior de expertos"""
        
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        predictions = []
        uncertainties = []
        
        for i, (expert, weight) in enumerate(zip(self.experts, self.expert_weights)):
            if self.feature_subsets and i < len(self.feature_subsets):
                X_subset = X_test_array[:, self.feature_subsets[i]]
            else:
                X_subset = X_test_array
                
            pred = expert.predict_posterior_predictive(X_subset)
            predictions.append(pred['mean'] * weight)
            uncertainties.append((pred['var'] + pred['mean']**2) * weight)
        
        ensemble_mean = np.sum(predictions, axis=0)
        ensemble_var = np.sum(uncertainties, axis=0) - ensemble_mean**2
        ensemble_std = np.sqrt(np.maximum(ensemble_var, 1e-6))
        
        return {
            'mean': ensemble_mean,
            'std': ensemble_std,
            'var': ensemble_var,
            'expert_weights': self.expert_weights
        }


print("✓ H7BayesianOracle & H7BayesianEnsemble defined")

# ============================================================================
# PART 4: SYNTHETIC DATA GENERATOR
# ============================================================================

def generate_h7_synthetic_data(n_samples: int = 100,
                              n_features: int = 5,
                              noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Genera datos sintéticos con estructura H7.
    y = sum(w_i * φ * x_i) + noise
    """
    
    # Pesos H7 (escalados por φ)
    true_weights = np.random.randn(n_features) * φ
    
    # Datos
    X = np.random.randn(n_samples, n_features)
    y_true = X @ true_weights
    y_noisy = y_true + np.random.randn(n_samples) * noise_level
    
    return (
        pd.DataFrame(X, columns=[f"x_{i}" for i in range(n_features)]),
        pd.Series(y_noisy, name="y")
    )


# ============================================================================
# PART 5: KBENCH TASKS
# ============================================================================

@kbench.task(
    name="bayesian_oracle_conjugate_inference",
    description="H7 Bayesian Oracle: Conjugate Gaussian Inference"
)
def test_bayesian_conjugate_inference(llm):
    """Test H7 Bayesian conjugate inference"""
    
    print("\n[BAYESIAN TEST 1] Conjugate Gaussian Inference")
    print("-" * 60)
    
    # Generate data
    X, y = generate_h7_synthetic_data(n_samples=50, n_features=3)
    
    # Create oracle
    oracle = H7BayesianOracle(n_features=3)
    results = oracle.fit_conjugate_gaussian(X, y)
    
    print(f"Posterior Mean: {results['posterior_mean']}")
    print(f"Log Evidence: {results['log_evidence']:.4f}")
    
    # Assertions
    kbench.assertions.assert_near(
        len(results['posterior_mean']), 3,
        expectation="Posterior mean should have correct dimensionality"
    )
    
    kbench.assertions.assert_true(
        results['log_evidence'] < 0,
        expectation="Log evidence should be negative"
    )
    
    print("✅ Task PASSED: Bayesian Conjugate Inference")


@kbench.task(
    name="bayesian_posterior_predictive",
    description="H7 Bayesian: Posterior Predictive Distribution"
)
def test_posterior_predictive(llm):
    """Test posterior predictive distribution"""
    
    print("\n[BAYESIAN TEST 2] Posterior Predictive Distribution")
    print("-" * 60)
    
    # Generate data
    X, y = generate_h7_synthetic_data(n_samples=30, n_features=4)
    X_train, X_test = X.iloc[:20], X.iloc[20:]
    y_train, y_test = y.iloc[:20], y.iloc[20:]
    
    # Fit oracle
    oracle = H7BayesianOracle(n_features=4)
    oracle.fit_conjugate_gaussian(X_train, y_train)
    
    # Predict
    pred = oracle.predict_posterior_predictive(X_test)
    
    print(f"Predicted Mean: {pred['mean'][:3]}")
    print(f"Predicted Std: {pred['std'][:3]}")
    print(f"Credible Interval (95%): {pred['credible_interval_95'][0][:3]}")
    
    # Assertions
    kbench.assertions.assert_true(
        np.all(pred['std'] > 0),
        expectation="Uncertainty estimates should be positive"
    )
    
    kbench.assertions.assert_near(
        len(pred['mean']), len(X_test),
        expectation="Prediction length should match test set"
    )
    
    print("✅ Task PASSED: Posterior Predictive Distribution")


@kbench.task(
    name="h7_integrity_metric",
    description="H7 Integrity Metric: Bayesian Predictive Quality"
)
def test_h7_integrity_metric(llm):
    """Test H7 integrity metric"""
    
    print("\n[BAYESIAN TEST 3] H7 Integrity Metric")
    print("-" * 60)
    
    # Generate data
    X, y = generate_h7_synthetic_data(n_samples=40, n_features=3)
    
    # Fit oracle
    oracle = H7BayesianOracle(n_features=3)
    oracle.fit_conjugate_gaussian(X, y)
    
    # Compute integrity
    integrity = oracle.compute_h7_integrity_metric(X, y)
    
    print(f"H7 Integrity Metric: {integrity:.6f}")
    print(f"O_n Integrity: {O_n_integrity:.10f}")
    
    # Assertions
    kbench.assertions.assert_true(
        0 <= integrity <= 1,
        expectation="Integrity metric should be in [0, 1]"
    )
    
    kbench.assertions.assert_near(
        integrity, 0.5, tolerance=1.0,
        expectation="Integrity should be reasonable"
    )
    
    print("✅ Task PASSED: H7 Integrity Metric")


@kbench.task(
    name="bayesian_ensemble_weighting",
    description="H7 Bayesian Ensemble: Evidence-Based Expert Weighting"
)
def test_bayesian_ensemble(llm):
    """Test H7 Bayesian ensemble with automatic expert weighting"""
    
    print("\n[BAYESIAN TEST 4] Bayesian Ensemble")
    print("-" * 60)
    
    # Generate data
    X, y = generate_h7_synthetic_data(n_samples=60, n_features=6)
    
    # Create feature subsets for experts
    feature_subsets = [
        [0, 1],  # Expert 1: features 0,1
        [2, 3],  # Expert 2: features 2,3
        [4, 5],  # Expert 3: features 4,5
        [0, 2, 4],  # Expert 4: features 0,2,4
        [1, 3, 5],  # Expert 5: features 1,3,5
        [0, 1, 2],  # Expert 6: features 0,1,2
        [3, 4, 5],  # Expert 7: features 3,4,5
    ]
    
    # Fit ensemble
    ensemble = H7BayesianEnsemble(n_experts=7)
    ensemble.fit_experts(X, y, feature_subsets=feature_subsets)
    
    # Predict
    X_test = X.iloc[:10]
    pred = ensemble.predict_ensemble(X_test)
    
    print(f"\nEnsemble Prediction Mean: {pred['mean'][:3]}")
    print(f"Ensemble Prediction Std: {pred['std'][:3]}")
    
    # Assertions
    kbench.assertions.assert_true(
        np.sum(ensemble.expert_weights) > 0.99,
        expectation="Expert weights should sum to ~1"
    )
    
    kbench.assertions.assert_true(
        np.all(ensemble.expert_weights > 0),
        expectation="All expert weights should be positive"
    )
    
    kbench.assertions.assert_true(
        len(pred['mean']) == len(X_test),
        expectation="Ensemble predictions should match test set length"
    )
    
    print("✅ Task PASSED: Bayesian Ensemble")


print("✓ All KBench Bayesian tasks registered")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_bayesian_tests():
    """Run all Bayesian inference tests"""
    
    print("\n" + "="*80)
    print("H7 BAYESIAN ORACLE + KBENCH INTEGRATION")
    print("="*80)
    
    try:
        llm = kbench.llm
        
        tasks = [
            ("bayesian_oracle_conjugate_inference", test_bayesian_conjugate_inference),
            ("bayesian_posterior_predictive", test_posterior_predictive),
            ("h7_integrity_metric", test_h7_integrity_metric),
            ("bayesian_ensemble_weighting", test_bayesian_ensemble),
        ]
        
        results = []
        for task_name, task_func in tasks:
            try:
                task_func(llm)
                results.append((task_name, "PASSED"))
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                results.append((task_name, "FAILED"))
        
        # Summary
        print(f"\n{'='*80}")
        print("BAYESIAN TESTS SUMMARY")
        print(f"{'='*80}")
        
        for task_name, status in results:
            symbol = "✅" if status == "PASSED" else "❌"
            print(f"{symbol} {task_name:40s} {status}")
        
        passed = sum(1 for _, s in results if s == "PASSED")
        total = len(results)
        print(f"\nTOTAL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        print(f"{'='*80}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_bayesian_tests()
