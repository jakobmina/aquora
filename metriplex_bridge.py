import math
import numpy as np
from typing import Tuple, Dict, Optional

from h_logic import MetriplexOracle, MetriplexConfig, H7Conservation, FockBasis, FockConfig
from endian import TopologicalBigEndianEncoder, BigEndianHexadecimalEncoder

PHI: float = (1.0 + (math.sqrt(5.0))) / 2.0
PI: float = math.pi
GOLDEN_RATIO_SQUARED: float = PHI ** 2

def golden_operator(n: int, phi: float = PHI) -> float:
    """O_n = cos(π·n) · cos(π·φ·n)"""
    parity = math.cos(PI * n)
    quasiperiod = math.cos(PI * phi * n)
    return parity * quasiperiod

def o_n_to_phase_fragment(o_n_value: float) -> int:
    fragment = round((o_n_value + 1.0) * 3.5)
    return int(np.clip(fragment, 0, 7))

def phase_fragment_to_o_n(fragment: int) -> float:
    return (fragment / 3.5) - 1.0

class MetriplexEndianBridge:
    def __init__(self, oracle: Optional[MetriplexOracle] = None, fock: Optional[FockBasis] = None):
        self.oracle = oracle or MetriplexOracle(MetriplexConfig())
        self.fock = fock or FockBasis(FockConfig(n_modes=3, n_max=2))
        self._encoder = TopologicalBigEndianEncoder
        self._hex = BigEndianHexadecimalEncoder

    def compute_lagrangian(self) -> Tuple[float, float]:
        L_symp = 0.0
        L_metr = 0.0
        for entry in self._encoder.topology_entries:
            p = entry['index']
            if p < self.oracle.p_min or p > self.oracle.p_max:
                continue
            _, _, energy = self.oracle.forward(p)
            if entry['winding'] == 0:
                L_symp += energy
            elif entry['winding'] == 2:
                L_metr += energy
        return L_symp, L_metr

    def encode_fock_state(self, occupation: Tuple[int, ...], fmt: str = 'uint16') -> str:
        p = self.oracle._occupation_to_momentum(occupation)
        group, _, energy = self.oracle.forward(p)
        if group == 'A':
            winding, mapping, ternary = 0, 0, 1
        else:
            winding, mapping, ternary = 2, 1, -1

        h7_state = (p - 1) % 8
        o_n_val = golden_operator(h7_state)
        phase_frag = o_n_to_phase_fragment(o_n_val)

        packed = self._encoder.pack_topology(
            index=p,
            pair=7 - p if 1 <= 7 - p <= 6 else p,
            winding=winding,
            mapping=mapping,
            ternary_weight=ternary,
            discrete_phase_fragment=phase_frag,
        )
        return self._to_hex(packed, fmt)

    def _to_hex(self, value, fmt: str) -> str:
        masks = {
            'uint8':   (0xFF, self._hex.to_hex_uint8),
            'uint16':  (0xFFFF, self._hex.to_hex_uint16),
            'uint32':  (0xFFFFFFFF, self._hex.to_hex_uint32),
            'uint64':  (0xFFFFFFFFFFFFFFFF, self._hex.to_hex_uint64),
            'uint128': ((1 << 128) - 1, self._hex.to_hex_uint128),
        }
        mask, encoder = masks[fmt]
        return encoder(int(value) & mask)

    def full_state_report(self, occupation: Tuple[int, ...]) -> Dict:
        """
        Generates a physically transparent report of the state, including
        classification for Second Quantization and Vacuum Overlaps.
        """
        p = self.oracle._occupation_to_momentum(occupation)
        group, output_vec, energy = self.oracle.forward(p)
        
        # Mapping to H7 phase space (0-7)
        h7_state = (p - 1) % 8
        o_n_val = golden_operator(h7_state)
        
        # Second Quantization: n odd -> Fermionic, n even -> Bosonic
        particle_type = "fermionic" if h7_state % 2 != 0 else "bosonic"
        
        # Non-Local Vacuum Overlap (Irreversibility measure)
        # W = O(n) + O(7-n)
        complement_n = 7 - h7_state
        vacuum_overlap = o_n_val + golden_operator(complement_n)
        
        hex16 = self.encode_fock_state(occupation, fmt='uint16')
        L_symp, L_metr = self.compute_lagrangian()

        return {
            'occupation': occupation,
            'momentum_p': p,
            'particle_type': particle_type,
            'vacuum_overlap': vacuum_overlap,
            'o_n': o_n_val,
            'hex_uint16': hex16,
            'L_symp': L_symp,
            'L_metr': L_metr,
            'ratio': L_symp / L_metr if L_metr > 0 else 0
        }
