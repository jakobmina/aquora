import numpy as np
from typing import Tuple, Dict, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings

class EnergyProfile(Enum):
    """Predefined energy normalization profiles."""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    METRIPLEX = "metriplex"
    CUSTOM = "custom"

@dataclass
class MetriplexConfig:
    """Configuration for metriplex oracle."""
    momentum_range: Tuple[int, int] = (1, 6)
    energy_profile: EnergyProfile = EnergyProfile.METRIPLEX
    normalization_target: float = 0.45
    collision_groups: Optional[Dict[str, List[int]]] = None

    def __post_init__(self):
        if self.collision_groups is None:
            self.collision_groups = {
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            }

class MetriplexOracle:
    """
    Metriplex momentum oracle for second quantization.

    Maps momentum states to energy vectors establishing a hidden 2-to-1
    function. Simon's algorithm discovers: momentum pairs differing by 3
    collide. (Symplectic component: phase shifts; Metric component: energy
    relaxation toward normalization_target attractor.)
    """
    def __init__(self, config: MetriplexConfig = None):
        if config is None:
            config = MetriplexConfig()
        self.config = config
        self.p_min, self.p_max = config.momentum_range
        self._build_energy_map()
        self._build_collision_map()

    def _build_energy_map(self):
        self.energy_map = {}
        for p in range(self.p_min, self.p_max + 1):
            if self.config.energy_profile == EnergyProfile.LINEAR:
                raw_energy = p / self.p_max
            elif self.config.energy_profile == EnergyProfile.QUADRATIC:
                raw_energy = (p / self.p_max) ** 2
            elif self.config.energy_profile == EnergyProfile.METRIPLEX:
                alpha = 1.2
                raw_energy = self.config.normalization_target * (p / self.p_max) ** alpha
            else:
                raise ValueError(f"Unknown energy profile: {self.config.energy_profile}")
            self.energy_map[p] = raw_energy

        mean_energy = np.mean([self.energy_map[p] for p in range(self.p_min, self.p_max + 1)])
        if abs(mean_energy - self.config.normalization_target) > 0.01:
            warnings.warn(
                f"Mean energy {mean_energy:.4f} deviates from target "
                f"{self.config.normalization_target:.4f}"
            )

    def _build_collision_map(self):
        self.collision_map = {}
        self.output_groups = {}
        for group_name, momenta in self.config.collision_groups.items():
            for p in momenta:
                self.collision_map[p] = group_name
            group_index = list(self.config.collision_groups.keys()).index(group_name)
            n_groups = len(self.config.collision_groups)
            output_vec = np.zeros(n_groups)
            output_vec[group_index] = 1.0
            self.output_groups[group_name] = output_vec

    def _compute_symmetry_string(self) -> int:
        group_lists = list(self.config.collision_groups.values())
        xor_accumulator = 0
        for group in group_lists:
            for i, p1 in enumerate(group):
                for p2 in group[i+1:]:
                    xor_accumulator |= (p1 ^ p2)
        return xor_accumulator

    def forward(self, momentum: int) -> Tuple[str, np.ndarray, float]:
        """Evaluate oracle at given momentum state."""
        if momentum < self.p_min or momentum > self.p_max:
            raise ValueError(f"Momentum {momentum} out of range [{self.p_min}, {self.p_max}]")
        group = self.collision_map[momentum]
        output_vec = self.output_groups[group]
        energy = self.energy_map[momentum]
        return group, output_vec, energy

    def collide_pair(self, p1: int, p2: int) -> bool:
        return self.collision_map[p1] == self.collision_map[p2]

    def get_collision_partners(self, p: int) -> List[int]:
        group = self.collision_map[p]
        return self.config.collision_groups[group]

    def symmetry_string(self) -> int:
        return self._compute_symmetry_string()

    def to_hilbert_oracle(self, fock_basis) -> Callable:
        def quantum_oracle(state_vector: np.ndarray) -> np.ndarray:
            result = state_vector.copy()
            for idx, occupation in enumerate(fock_basis.basis_states):
                effective_momentum = self._occupation_to_momentum(occupation)
                energy = self.energy_map[effective_momentum]
                phase_shift = np.exp(2j * np.pi * energy)
                result[idx] *= phase_shift
            return result
        return quantum_oracle

    def _occupation_to_momentum(self, occupation: Tuple[int, ...]) -> int:
        total_occ = sum(occupation)
        range_size = self.p_max - self.p_min + 1
        p = (total_occ % range_size) + self.p_min
        return p

    def get_oracle_info(self) -> Dict:
        return {
            'momentum_range': self.config.momentum_range,
            'n_groups': len(self.config.collision_groups),
            'collision_groups': {k: v for k, v in self.config.collision_groups.items()},
            'symmetry_string': self.symmetry_string(),
            'energy_profile': self.config.energy_profile.value,
            'normalization_target': self.config.normalization_target,
            'energy_map': {p: self.energy_map[p] for p in range(self.p_min, self.p_max + 1)},
            'collision_structure': {p: self.collision_map[p] for p in range(self.p_min, self.p_max + 1)}
        }

class H7Conservation:
    """
    Mechanism I: H7 Entanglement Conservation
    """
    CONSERVATION_CONSTANT = 7

    @staticmethod
    def partner_state(state: int) -> int:
        if not (0 <= state <= 7):
            raise ValueError("State must be in [0, 7]")
        return H7Conservation.CONSERVATION_CONSTANT ^ state

    @staticmethod
    def verify_pairing(state_a: int, state_b: int) -> bool:
        return state_b == H7Conservation.partner_state(state_a)

    @staticmethod
    def pairing_table() -> Dict[int, int]:
        return {i: H7Conservation.partner_state(i) for i in range(8)}

    @staticmethod
    def verify_conservation_invariant(state_vector: np.ndarray, threshold: float = 1e-6) -> bool:
        if len(state_vector) != 8:
            raise ValueError("State vector must be 8-dimensional (3-qubit Hilbert space)")
        for i in range(8):
            if abs(state_vector[i]) > threshold:
                partner = H7Conservation.partner_state(i)
                if abs(state_vector[partner]) < threshold:
                    return False
        return True

class OccupationMode(Enum):
    """Enumeration of occupation number modes in Fock space."""
    BOSONIC = "bosonic"
    FERMIONIC = "fermionic"

@dataclass
class FockConfig:
    """Configuration for Fock space instantiation."""
    n_modes: int = 3
    n_max: int = 3
    mode_type: OccupationMode = OccupationMode.BOSONIC
    use_gray_code: bool = True

class FockBasis:
    """
    Fock space representation with second quantization operators.
    dim(H_Fock) = (n_max + 1)^n_modes
    """
    def __init__(self, config: FockConfig = None):
        if config is None:
            config = FockConfig()
        self.config = config
        self.n_modes = config.n_modes
        self.n_max = config.n_max
        self.dim = (config.n_max + 1) ** config.n_modes
        self._build_basis()
        self._precompute_operators()

    def _build_basis(self):
        basis_states = []
        for occupation in np.ndindex(*[self.n_max + 1] * self.n_modes):
            basis_states.append(occupation)
        self.basis_states = np.array(basis_states)
        assert len(self.basis_states) == self.dim
        self.state_to_index = {tuple(state): i for i, state in enumerate(self.basis_states)}
        self.index_to_state = {i: tuple(state) for i, state in enumerate(self.basis_states)}

    def _precompute_operators(self):
        self.creation_ops = {}
        self.annihilation_ops = {}
        for mode in range(self.n_modes):
            self.creation_ops[mode] = self._build_creation_op(mode)
            self.annihilation_ops[mode] = self._build_annihilation_op(mode)

    def _build_creation_op(self, mode: int) -> np.ndarray:
        op = np.zeros((self.dim, self.dim), dtype=complex)
        for i, state in enumerate(self.basis_states):
            state_list = list(state)
            if state_list[mode] < self.n_max:
                new_occupation = state_list[mode] + 1
                amplitude = np.sqrt(new_occupation)
                state_list[mode] = new_occupation
                j = self.state_to_index[tuple(state_list)]
                op[j, i] = amplitude
        return op

    def _build_annihilation_op(self, mode: int) -> np.ndarray:
        op = np.zeros((self.dim, self.dim), dtype=complex)
        for i, state in enumerate(self.basis_states):
            state_list = list(state)
            if state_list[mode] > 0:
                old_occupation = state_list[mode]
                amplitude = np.sqrt(old_occupation)
                state_list[mode] = old_occupation - 1
                j = self.state_to_index[tuple(state_list)]
                op[j, i] = amplitude
        return op

    def get_creation_op(self, mode: int) -> np.ndarray:
        return self.creation_ops[mode].copy()

    def get_annihilation_op(self, mode: int) -> np.ndarray:
        return self.annihilation_ops[mode].copy()

    def number_operator(self, mode: int) -> np.ndarray:
        a_dag = self.get_creation_op(mode)
        a = self.get_annihilation_op(mode)
        return a_dag @ a

    def total_number_operator(self) -> np.ndarray:
        N_total = np.zeros((self.dim, self.dim), dtype=complex)
        for mode in range(self.n_modes):
            N_total += self.number_operator(mode)
        return N_total

    def state_vector(self, occupation: Tuple[int, ...]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=complex)
        idx = self.state_to_index[tuple(occupation)]
        vec[idx] = 1.0
        return vec

class FockStateVector:
    """Quantum state in Fock basis with helper methods."""
    def __init__(self, fock_basis: FockBasis, vector: np.ndarray = None):
        self.fock = fock_basis
        if vector is None:
            self.vec = np.zeros(fock_basis.dim, dtype=complex)
        else:
            if len(vector) != fock_basis.dim:
                raise ValueError(f"Vector dimension {len(vector)} != {fock_basis.dim}")
            self.vec = vector.astype(complex)

    def normalize(self):
        norm = np.linalg.norm(self.vec)
        if norm > 1e-10:
            self.vec /= norm
        return self
