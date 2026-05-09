import pytest
import numpy as np
from h7_metriplectic_os.utf8_exp3 import encode_char, decode_char, encode_string, decode_string

def test_antagonist_pairs():
    """Verify that 'a' and 'A' generate conjugate properties."""
    psi_lower = encode_char('a')
    psi_upper = encode_char('A')
    
    # Ambos deben ser estados válidos (suma = 1)
    assert np.isclose(np.sum(psi_lower), 1.0)
    assert np.isclose(np.sum(psi_upper), 1.0)
    
    # La escala debe ser idéntica para pares antagonistas
    assert np.isclose(psi_lower[0] + psi_lower[7], psi_upper[0] + psi_upper[7])
    
    # Extraer diferencias de torsión (x = psi_1 - psi_6)
    x_lower = psi_lower[1] - psi_lower[6]
    x_upper = psi_upper[1] - psi_upper[6]
    
    y_lower = psi_lower[2] - psi_lower[5]
    y_upper = psi_upper[2] - psi_upper[5]
    
    z_lower = psi_lower[3] - psi_lower[4]
    z_upper = psi_upper[3] - psi_upper[4]
    
    # Para el caracter 'a' y 'A' (bits 2,3,4 podrían no todos ser 0)
    # 97 = 0110 0001 -> b2=0, b3=0, b4=0
    # The torsion for 'a' might be 0, wait, if torsion is 0, we fallback to b5=1.
    # Let's test letters with non-zero torsion, e.g. 'c' (99 = 0110 0011 -> b0=1, b1=1, b2=0, b3=0, b4=0)
    # Wait, 'c' is 99, 99%32 = 3. b2=0, b3=0, b4=0. Still 0!
    # Let's test 'p' (112 = 0111 0000 -> b4=1). 
    
    psi_p_lower = encode_char('p')
    psi_p_upper = encode_char('P')
    
    z_p_lower = psi_p_lower[3] - psi_p_lower[4]
    z_p_upper = psi_p_upper[3] - psi_p_upper[4]
    
    # Torsion of 'P' should be EXACTLY negative of 'p'
    assert np.isclose(z_p_lower, -z_p_upper)
    assert z_p_lower > 0  # Lowercase has positive torsion
    assert z_p_upper < 0  # Uppercase has negative torsion
    
def test_full_decode_encode():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?@#"
    
    for c in chars:
        encoded = encode_char(c)
        decoded = decode_char(encoded)
        assert decoded == c, f"Failed for char '{c}' (ord {ord(c)})"

def test_string_codec():
    msg = "Aquora H7 - Metriplectic Protocol"
    encoded = encode_string(msg)
    decoded = decode_string(encoded)
    assert decoded == msg
