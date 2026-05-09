import struct
from typing import Dict, Tuple, Optional, List, Union

class HexadecimalSpecifications:
    """Defines exact hexadecimal formats according to user table"""
    FORMATS = {
        'uint8': {
            'bits': 8, 'bytes': 1, 'hex_chars': 2,
            'min_value': 0, 'max_value': 255, 'signed': False
        },
        'uint16': {
            'bits': 16, 'bytes': 2, 'hex_chars': 4,
            'min_value': 0, 'max_value': 65535, 'signed': False
        },
        'uint32': {
            'bits': 32, 'bytes': 4, 'hex_chars': 8,
            'min_value': 0, 'max_value': 4294967295, 'signed': False
        },
        'uint64': {
            'bits': 64, 'bytes': 8, 'hex_chars': 16,
            'min_value': 0, 'max_value': 18446744073709551615, 'signed': False
        },
        'uint128': {
            'bits': 128, 'bytes': 16, 'hex_chars': 32,
            'min_value': 0, 'max_value': 2**128 - 1, 'signed': False
        },
        'int8': {
            'bits': 8, 'bytes': 1, 'hex_chars': 2,
            'min_value': -128, 'max_value': 127, 'signed': True
        },
    }

class BigEndianHexadecimalEncoder:
    """Encodes values to hexadecimal with big-endian byte ordering."""
    @staticmethod
    def to_hex_uint8(value: int) -> str:
        return f"{value:02X}"

    @staticmethod
    def to_hex_uint16(value: int) -> str:
        return f"{value:04X}"

    @staticmethod
    def to_hex_uint32(value: int) -> str:
        return f"{value:08X}"

    @staticmethod
    def to_hex_uint64(value: int) -> str:
        return f"{value:016X}"

    @staticmethod
    def to_hex_uint128(value: int) -> str:
        return f"{value:032X}"

    @staticmethod
    def from_hex_uint16(hex_str: str) -> int:
        return int(hex_str, 16)


class TopologicalBigEndianEncoder:
    """
    Packs topology and converts to the exact hexadecimal formats of the table.
    """
    TERNARY_TO_BITS = {-1: 0b00, 0: 0b01, +1: 0b10}
    
    topology_entries = [
        {'index': 1, 'pair': 6, 'winding': 0, 'mapping': 0, 'ternary_weight': 1,  'discrete_phase_fragment': 0},
        {'index': 5, 'pair': 2, 'winding': 0, 'mapping': 0, 'ternary_weight': 1,  'discrete_phase_fragment': 1},
        {'index': 3, 'pair': 4, 'winding': 0, 'mapping': 0, 'ternary_weight': 1,  'discrete_phase_fragment': 6},
        {'index': 4, 'pair': 3, 'winding': 2, 'mapping': 1, 'ternary_weight': -1, 'discrete_phase_fragment': 5},
        {'index': 5, 'pair': 2, 'winding': 2, 'mapping': 1, 'ternary_weight': -1, 'discrete_phase_fragment': 2},
        {'index': 6, 'pair': 1, 'winding': 2, 'mapping': 1, 'ternary_weight': -1, 'discrete_phase_fragment': 3},
        {'index': 2, 'pair': 3, 'winding': 0, 'mapping': 0, 'ternary_weight': 0,  'discrete_phase_fragment': 4},
    ]

    @staticmethod
    def pack_topology(
        index: int, pair: int, winding: int, mapping: int,
        ternary_weight: int, discrete_phase_fragment: int
    ) -> int:
        value = 0
        value |= (index - 1) << 0
        value |= (pair - 1) << 3
        value |= (winding // 2) << 6
        value |= mapping << 8
        value |= TopologicalBigEndianEncoder.TERNARY_TO_BITS[ternary_weight] << 9
        value |= discrete_phase_fragment << 11
        return value

    @staticmethod
    def unpack_topology(value: int) -> Dict:
        index = ((value >> 0) & 0b111) + 1
        pair = ((value >> 3) & 0b111) + 1
        winding = ((value >> 6) & 0b1) * 2
        mapping = (value >> 8) & 0b1
        weight_bits = (value >> 9) & 0b11
        ternary_weight = {0b00: -1, 0b01: 0, 0b10: 1}.get(weight_bits, 0)
        discrete_phase_fragment = (value >> 11) & 0b111
        cycle_phase_derived = (discrete_phase_fragment / 3.5) - 1.0

        return {
            'index': index, 'pair': pair, 'winding': winding,
            'mapping': mapping, 'ternary_weight': ternary_weight,
            'discrete_phase_fragment': discrete_phase_fragment,
            'cycle_phase_derived': cycle_phase_derived,
        }
