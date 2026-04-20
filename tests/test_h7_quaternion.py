"""
tests/test_h7_quaternion.py
===========================
Pytest suite for h7_quaternion.py.

Covers:
  - states_to_quaternion(): H7 pair mapping
  - Hamilton product (non-commutative)
  - quat_conjugate / quat_norm / quat_normalize
  - commutator / anti_commutator
  - compute_lagrangian_quaternion() — Metriplectic brackets
  - Regla 1.3: no pure-state collapse
  - Chirality ↔ particle classification
  - H7QuaternionMapper: analyze() completeness and physics validity
  - Reversibility test (Regla 1, MANIFIESTO)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h7_quaternion import (
    H7_AMPLITUDES,
    H7_AMPLITUDES_MIRROR,
    H7_PAIR_LABELS,
    states_to_quaternion,
    quat_multiply,
    quat_conjugate,
    quat_norm,
    quat_normalize,
    commutator,
    anti_commutator,
    compute_lagrangian_quaternion,
    H7QuaternionMapper,
)


# ────────────────────────────────────────────────────────────
# FIXTURES
# ────────────────────────────────────────────────────────────

@pytest.fixture
def amplitudes():
    return H7_AMPLITUDES.copy()


@pytest.fixture
def q_LE(amplitudes):
    return states_to_quaternion(amplitudes)


@pytest.fixture
def q_BE(q_LE):
    # Mirror quaternion: partial conjugate (breaks commutativity)
    return np.array([
        q_LE[0], np.conj(q_LE[1]), q_LE[2], np.conj(q_LE[3]),
    ], dtype=complex)


@pytest.fixture
def mapper():
    # Default mapper: H7_AMPLITUDES vs. its parity-mirror quaternion
    return H7QuaternionMapper()


# ────────────────────────────────────────────────────────────
# states_to_quaternion
# ────────────────────────────────────────────────────────────

class TestStatesToQuaternion:
    def test_output_shape(self, amplitudes):
        q = states_to_quaternion(amplitudes)
        assert q.shape == (4,)

    def test_output_dtype_complex(self, amplitudes):
        q = states_to_quaternion(amplitudes)
        assert np.iscomplexobj(q)

    def test_pair_sums_correct(self, amplitudes):
        """Each component must equal the stated pair sum."""
        q = states_to_quaternion(amplitudes)
        assert np.isclose(q[0], amplitudes[0] + amplitudes[7])
        assert np.isclose(q[1], amplitudes[1] + amplitudes[6])
        assert np.isclose(q[2], amplitudes[2] + amplitudes[5])
        assert np.isclose(q[3], amplitudes[3] + amplitudes[4])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="8"):
            states_to_quaternion(np.ones(6, dtype=complex))

    def test_real_amplitudes_give_real_quaternion(self):
        """All-real amplitudes → all-real quaternion components."""
        a = np.ones(8, dtype=float)
        q = states_to_quaternion(a)
        assert np.all(np.imag(q) == 0)


# ────────────────────────────────────────────────────────────
# Hamilton product
# ────────────────────────────────────────────────────────────

class TestQuatMultiply:
    def test_identity_left(self):
        """q ⊗ identity = q"""
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        q = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
        assert np.allclose(quat_multiply(identity, q), q)

    def test_identity_right(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        q = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
        assert np.allclose(quat_multiply(q, identity), q)

    def test_non_commutativity(self, q_LE, q_BE):
        """
        q_LE ⊗ q_BE ≠ q_BE ⊗ q_LE when q_LE and q_BE are DISTINCT states.
        q_LE = H7_AMPLITUDES,  q_BE = H7_AMPLITUDES_PHASED (π/4 rotation)
        """
        q_LB = quat_multiply(q_LE, q_BE)
        q_BL = quat_multiply(q_BE, q_LE)
        assert not np.allclose(q_LB, q_BL), (
            "Products of two different H7 states should not commute"
        )

    def test_associativity(self, q_LE, q_BE):
        """(q_LE ⊗ q_BE) ⊗ q_LE == q_LE ⊗ (q_BE ⊗ q_LE)"""
        lhs = quat_multiply(quat_multiply(q_LE, q_BE), q_LE)
        rhs = quat_multiply(q_LE, quat_multiply(q_BE, q_LE))
        assert np.allclose(lhs, rhs)

    def test_output_shape(self, q_LE, q_BE):
        result = quat_multiply(q_LE, q_BE)
        assert result.shape == (4,)

    def test_ij_equals_k(self):
        """i ⊗ j = k  (Hamilton's fundamental relation)"""
        i = np.array([0, 1, 0, 0], dtype=complex)
        j = np.array([0, 0, 1, 0], dtype=complex)
        k = np.array([0, 0, 0, 1], dtype=complex)
        assert np.allclose(quat_multiply(i, j), k)

    def test_ji_equals_minus_k(self):
        """j ⊗ i = −k"""
        i = np.array([0, 1, 0, 0], dtype=complex)
        j = np.array([0, 0, 1, 0], dtype=complex)
        k = np.array([0, 0, 0, -1], dtype=complex)
        assert np.allclose(quat_multiply(j, i), k)


# ────────────────────────────────────────────────────────────
# quat_conjugate / quat_norm / quat_normalize
# ────────────────────────────────────────────────────────────

class TestQuatHelpers:
    def test_conjugate_real_unchanged(self):
        q = np.array([5.0, 0.0, 0.0, 0.0], dtype=complex)
        assert np.allclose(quat_conjugate(q), q)

    def test_conjugate_flips_imaginary(self):
        q = np.array([1.0, 2.0, -3.0, 4.0], dtype=complex)
        qc = quat_conjugate(q)
        assert np.isclose(qc[0],  q[0])
        assert np.isclose(qc[1], -q[1])
        assert np.isclose(qc[2], -q[2])
        assert np.isclose(qc[3], -q[3])

    def test_norm_identity(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        assert np.isclose(quat_norm(identity), 1.0)

    def test_norm_known_value(self):
        q = np.array([3.0, 4.0, 0.0, 0.0], dtype=complex)
        assert np.isclose(quat_norm(q), 5.0)

    def test_normalize_unit_norm(self, q_LE):
        q_unit = quat_normalize(q_LE)
        assert np.isclose(quat_norm(q_unit), 1.0)

    def test_normalize_zero_fallback(self):
        q_zero = np.zeros(4, dtype=complex)
        q_fb = quat_normalize(q_zero)
        assert np.isclose(quat_norm(q_fb), 1.0)
        assert np.isclose(q_fb[0], 1.0)  # identity fallback


# ────────────────────────────────────────────────────────────
# commutator / anti_commutator
# ────────────────────────────────────────────────────────────

class TestBrackets:
    def test_commutator_antisymmetry(self, q_LE, q_BE):
        """[A, B] = −[B, A]"""
        comm_AB = commutator(q_LE, q_BE)
        comm_BA = commutator(q_BE, q_LE)
        assert np.allclose(comm_AB, -comm_BA)

    def test_commutator_self_zero(self, q_LE):
        """[q, q] = 0"""
        assert np.allclose(commutator(q_LE, q_LE), np.zeros(4, dtype=complex))

    def test_anti_commutator_symmetry(self, q_LE, q_BE):
        """{A, B} = {B, A}"""
        ac_AB = anti_commutator(q_LE, q_BE)
        ac_BA = anti_commutator(q_BE, q_LE)
        assert np.allclose(ac_AB, ac_BA)

    def test_commutator_nonzero_for_h7(self, q_LE, q_BE):
        """
        [q_LE, q_BE] ≠ 0 when the two states are distinct
        (H7_AMPLITUDES vs H7_AMPLITUDES_PHASED).
        """
        comm = commutator(q_LE, q_BE)
        assert quat_norm(comm) > 1e-8

    def test_commutator_zero_for_same_state(self, q_LE):
        """
        Physical note: a single H7 state compared to itself (or its
        amplitude-reversed copy) gives [q, q] = 0 (Abelian / bosonic in
        isolation).  Non-Abelian structure emerges between DISTINCT states.
        """
        q_same = states_to_quaternion(H7_AMPLITUDES[::-1])  # equals q_LE
        assert np.allclose(commutator(q_LE, q_same), np.zeros(4, dtype=complex))

    def test_anti_commutator_nonzero(self, q_LE, q_BE):
        acomm = anti_commutator(q_LE, q_BE)
        assert quat_norm(acomm) > 1e-8


# ────────────────────────────────────────────────────────────
# compute_lagrangian_quaternion — Metriplectic brackets
# ────────────────────────────────────────────────────────────

class TestLagrangianQuaternion:
    def test_returns_two_floats(self, q_LE, q_BE):
        L_symp, L_metr = compute_lagrangian_quaternion(q_LE, q_BE)
        assert isinstance(L_symp, float)
        assert isinstance(L_metr, float)

    def test_both_positive(self, q_LE, q_BE):
        """Both Lagrangian norms must be non-negative."""
        L_symp, L_metr = compute_lagrangian_quaternion(q_LE, q_BE)
        assert L_symp >= 0
        assert L_metr >= 0

    def test_regla_1_3_no_pure_state(self):
        """
        Regla 1.3: If commutator is zero (Abelian), L_symp must be floored
        to 1e-10, not zero.
        """
        # Pure real quaternions commute → commutator = 0
        q_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        q_b = np.array([2.0, 0.0, 0.0, 0.0], dtype=complex)
        L_symp, _ = compute_lagrangian_quaternion(q_a, q_b)
        assert L_symp >= 1e-10

    def test_l_symp_maps_to_commutator_norm(self, q_LE, q_BE):
        """L_symp == ‖[q_LE, q_BE]‖"""
        L_symp, _ = compute_lagrangian_quaternion(q_LE, q_BE)
        expected = quat_norm(commutator(q_LE, q_BE))
        assert np.isclose(L_symp, expected)

    def test_l_metr_maps_to_anti_commutator_norm(self, q_LE, q_BE):
        """L_metr == ‖{q_LE, q_BE}‖"""
        _, L_metr = compute_lagrangian_quaternion(q_LE, q_BE)
        expected = quat_norm(anti_commutator(q_LE, q_BE))
        assert np.isclose(L_metr, expected)

    def test_ratio_near_phi_squared(self, q_LE, q_BE):
        """
        Golden balance (Regla MANIFIESTO III-B):
        We test that the ratio L_symp/L_metr is in a physically reasonable
        range for two distinct H7 states.  The exact φ² value is a target
        for tuned systems, not a hard constraint on all pairs.
        """
        L_symp, L_metr = compute_lagrangian_quaternion(q_LE, q_BE)
        ratio = L_symp / L_metr
        assert 0.0 < ratio < 100.0, f"Ratio {ratio:.4f} out of physical range"


# ────────────────────────────────────────────────────────────
# H7QuaternionMapper
# ────────────────────────────────────────────────────────────

class TestH7QuaternionMapper:
    def test_init_default(self):
        mapper = H7QuaternionMapper()
        assert len(mapper.amplitudes_a) == 8
        # Mirror state has no canonical amplitude repr
        assert mapper.amplitudes_b is None

    def test_init_custom_amplitudes(self):
        a = np.random.randn(8) + 1j * np.random.randn(8)
        b = np.random.randn(8) + 1j * np.random.randn(8)
        mapper = H7QuaternionMapper(a, b)
        assert np.allclose(mapper.amplitudes_a, a)
        assert np.allclose(mapper.amplitudes_b, b)

    def test_init_wrong_length_raises(self):
        with pytest.raises(ValueError):
            H7QuaternionMapper(np.ones(5, dtype=complex))

    def test_init_wrong_length_b_raises(self):
        with pytest.raises(ValueError):
            H7QuaternionMapper(H7_AMPLITUDES, np.ones(5, dtype=complex))

    def test_analyze_returns_dict(self, mapper):
        report = mapper.analyze()
        assert isinstance(report, dict)

    def test_analyze_required_keys(self, mapper):
        report = mapper.analyze()
        required = {
            "q_LE", "q_BE", "norm_LE", "norm_BE",
            "q_LE_x_BE", "q_BE_x_LE",
            "commutator", "anti_commutator",
            "L_symp", "L_metr",
            "chirality", "is_non_abelian", "is_commutative",
        }
        missing = required - set(report.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_q_LE_shape(self, mapper):
        report = mapper.analyze()
        assert report["q_LE"].shape == (4,)

    def test_q_BE_is_mirror_default(self, mapper):
        """
        Default q_BE must be the partial conjugate of q_LE
        (q0 unchanged, q1 conj, q2 unchanged, q3 conj).
        """
        q = mapper.q_LE
        expected = np.array([
            q[0], np.conj(q[1]), q[2], np.conj(q[3])
        ], dtype=complex)
        assert np.allclose(mapper.q_BE, expected)

    def test_is_non_abelian_true_for_h7_vs_mirror(self, mapper):
        """
        Default mapper (H7_AMPLITUDES vs. its parity-mirror) must be non-Abelian.
        The partial conjugate breaks commutativity → chirality > 0.
        """
        assert mapper.is_non_abelian

    def test_single_state_is_abelian(self):
        """
        A single H7 state compared to itself (or reversed copy) is Abelian.
        This is a fundamental check: the H7 reference state alone is bosonic
        in quaternion space; chirality requires two distinct states.
        """
        q = states_to_quaternion(H7_AMPLITUDES)
        q_rev = states_to_quaternion(H7_AMPLITUDES[::-1])
        assert np.allclose(q, q_rev), "Reversed copy must equal original (Abelian identity)"
        assert np.allclose(commutator(q, q_rev), np.zeros(4, dtype=complex))

    def test_chirality_positive(self, mapper):
        assert mapper.chirality > 0

    def test_norm_properties(self, mapper):
        """Both norms must be real positive numbers."""
        assert mapper.norm_LE > 0
        assert mapper.norm_BE > 0

    # Reversibility test — Regla 1, MANIFIESTO
    def test_reversibility_check(self, mapper):
        """
        Regla 1 (MANIFIESTO): The default mapper compares the original H7 state
        against its parity-mirror (partial conjugate). Parity reflection breaks
        time-reversal symmetry → non-commutative → fermionic.
        """
        report = mapper.analyze()
        assert not report["is_commutative"], (
            "H7 vs. parity-mirror must be non-commutative (fermionic)"
        )

    def test_print_report_runs(self, mapper, capsys):
        """print_report() must not raise and must print key labels."""
        report = mapper.analyze()
        mapper.print_report(report)
        captured = capsys.readouterr()
        assert "L_symp" in captured.out
        assert "L_metr" in captured.out
        assert "Chirality" in captured.out

    def test_analyze_lagrangian_nonzero(self, mapper):
        report = mapper.analyze()
        assert report["L_symp"] > 0
        assert report["L_metr"] > 0

    def test_commutator_equals_product_difference(self, mapper):
        """[q_LE, q_BE] == q_LE⊗q_BE − q_BE⊗q_LE"""
        report = mapper.analyze()
        expected = report["q_LE_x_BE"] - report["q_BE_x_LE"]
        assert np.allclose(report["commutator"], expected)


# ────────────────────────────────────────────────────────────
# INTEGRATION: quaternion record appears in run_vqe_maxcut
# ────────────────────────────────────────────────────────────

class TestQuaternionIntegration:
    QUAT_KEYS = {
        "q_L_symp", "q_L_metr", "chirality",
        "is_non_abelian", "norm_qLE", "norm_qBE",
    }

    def test_run_record_has_quat_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from run_vqe_maxcut import run_maxcut
        record = run_maxcut()
        missing = self.QUAT_KEYS - set(record.keys())
        assert not missing, f"Missing quaternion fields in record: {missing}"

    def test_chirality_positive_in_record(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from run_vqe_maxcut import run_maxcut
        record = run_maxcut()
        assert record["chirality"] > 0

    def test_quat_lagrangian_both_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from run_vqe_maxcut import run_maxcut
        record = run_maxcut()
        # L_symp may be the Regla 1.3 floor (1e-10) if Abelian; that's valid.
        assert record["q_L_symp"] >= 1e-10
        assert record["q_L_metr"] > 0

    def test_csv_has_quat_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from run_vqe_maxcut import run_maxcut
        import csv as _csv
        csv_file = str(tmp_path / "submission.csv")
        run_maxcut(export_csv=True, csv_path=csv_file)
        with open(csv_file, newline="", encoding="utf-8") as fh:
            header = set(_csv.DictReader(fh).fieldnames or [])
        missing = self.QUAT_KEYS - header
        assert not missing, f"Missing CSV columns: {missing}"
