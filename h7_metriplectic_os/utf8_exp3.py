import numpy as np

def encode_char(char: str) -> np.ndarray:
    """
    Codifica un carácter ASCII (7 bits) en un estado cuántico de 3 qubits (8 amplitudes).
    Usa el concepto de 'Pares Antagónicos' (Conjugación Cuaterniónica) propuesto:
    - Minúsculas (Base): Torsión Positiva (Cuaternión q)
    - Mayúsculas (Antagonista): Torsión Negativa (Cuaternión Conjugado q*)
    """
    val = ord(char)
    if val > 127:
        raise ValueError("utf-8exp3 prototype currently supports ASCII (0-127)")
        
    # Descomposición a 7 bits: b6 b5 b4 b3 b2 b1 b0
    b0 = (val >> 0) & 1
    b1 = (val >> 1) & 1
    b2 = (val >> 2) & 1
    b3 = (val >> 3) & 1
    b4 = (val >> 4) & 1
    
    # El bit 5 (32) diferencia mayúsculas (65='A') de minúsculas (97='a')
    # b5 = 0 -> Mayúscula (Antagonista)
    # b5 = 1 -> Minúscula (Base)
    b5 = (val >> 5) & 1 
    b6 = (val >> 6) & 1
    
    # Empaquetamos bits en las magnitudes del cuaternión
    # Escalar (Parte Real): w_mag (0 a 7)
    w_mag = b0 + (b1 << 1) + (b6 << 2)
    
    # Magnitudes Imaginarias
    # Sumamos 0.5 a x_mag para asegurar que siempre haya una Torsión base (quiralidad detectable)
    # incluso si b2, b3, b4 son 0.
    x_mag = b2 + 0.5
    y_mag = b3
    z_mag = b4
    
    # QUIRALIDAD (Antagonismo)
    # Si b5 == 0 (Mayúscula), conjugamos -> fase negativa
    # Si b5 == 1 (Minúscula), base -> fase positiva
    sign = 1.0 if b5 == 1 else -1.0
    
    # Construcción de Amplitudes de Probabilidad
    psi = np.zeros(8)
    
    # w = psi_0 + psi_7
    psi[0] = w_mag / 2.0
    psi[7] = w_mag / 2.0
    
    # Aplicamos el Tensor de Torsión: x = psi_1 - psi_6
    # Usamos 2.0 como divisor para mantener las probabilidades positivas
    psi[1] = (2.0 + sign * x_mag) / 4.0
    psi[6] = (2.0 - sign * x_mag) / 4.0
    
    # y = psi_2 - psi_5
    psi[2] = (1.0 + sign * y_mag) / 2.0
    psi[5] = (1.0 - sign * y_mag) / 2.0
    
    # z = psi_3 - psi_4
    psi[3] = (1.0 + sign * z_mag) / 2.0
    psi[4] = (1.0 - sign * z_mag) / 2.0
    
    # Normalización del estado cuántico (Σ P(i) = 1)
    total_prob = np.sum(psi)
    if total_prob > 0:
        psi = psi / total_prob
        
    return psi

def decode_char(psi: np.ndarray) -> str:
    """
    Decodifica un estado de 3 qubits (8 amplitudes) de vuelta a un carácter ASCII,
    midiendo la asimetría y el signo del conjugado.
    """
    # 1. Recuperar la escala de normalización original
    # Sabemos que la suma de (psi_1 + psi_6) antes de normalizar era 1.0
    S_x = psi[1] + psi[6]
    if S_x < 1e-10:
        return '?' # Corrupción cuántica
        
    scale = 1.0 / S_x
    
    # 2. Extraer el Escalar (w)
    w_mag = int(round((psi[0] + psi[7]) * scale))
    
    # 3. Medir el Tensor de Torsión (Asimetrías imaginarias)
    x_diff = (psi[1] - psi[6]) * scale
    y_diff = (psi[2] - psi[5]) * scale
    z_diff = (psi[3] - psi[4]) * scale
    
    # x_diff = sign * x_mag / 2.0 => sign * x_mag = 2.0 * x_diff
    # y_diff = sign * y_mag
    # z_diff = sign * z_mag
    
    sign_x_mag = 2.0 * x_diff
    sign_y_mag = y_diff
    sign_z_mag = z_diff
    
    # 4. Determinar la Quiralidad (Signo del Conjugado)
    # Como x_mag siempre es >= 0.5, sign_x_mag SIEMPRE tendrá el signo correcto!
    if sign_x_mag < 0:
        sign = -1.0
        b5 = 0
    else:
        sign = 1.0
        b5 = 1
        
    # Extraer magnitudes puras
    x_mag = sign_x_mag * sign
    y_mag = sign_y_mag * sign
    z_mag = sign_z_mag * sign
    
    # Restar la torsión base a x_mag
    b2 = int(round(x_mag - 0.5))
    b3 = int(round(y_mag))
    b4 = int(round(z_mag))
    
    # Extraer bits 0, 1, 6 de w_mag
    b0 = w_mag & 1
    b1 = (w_mag >> 1) & 1
    b6 = (w_mag >> 2) & 1
    
    # Reconstruir ASCII
    val = (b6 << 6) | (b5 << 5) | (b4 << 4) | (b3 << 3) | (b2 << 2) | (b1 << 1) | b0
    return chr(val)

def encode_string(text: str) -> np.ndarray:
    return np.array([encode_char(c) for c in text])

def decode_string(states: np.ndarray) -> str:
    return "".join([decode_char(psi) for psi in states])
