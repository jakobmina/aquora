import numpy as np

class H7NativeCodec:
    """
    H7 Native Codec.
    Treats standard ASCII/UTF-8 bytes as native H7 quantum states.
    Demonstrates that the structure of the English alphabet in ASCII
    is fundamentally an H7 Metriplectic system.
    """
    
    # El Bit 5 (0x20 = 32) es el Operador de Conjugación en ASCII
    CONJUGATION_OPERATOR = 0x20
    
    @staticmethod
    def apply_conjugation(char: str) -> str:
        """
        Aplica el operador de conjugación (Pauli-X en el qubit de Quiralidad).
        Convierte de Estado Base a Estado Conjugado (ej. 'a' <-> 'A').
        Esta es una operación atómica a nivel de bits.
        """
        if not char.isalpha():
            return char
            
        byte_val = ord(char)
        # Torsión Métrica: XOR con 0x20 invierte el Bit 5
        conjugated_byte = byte_val ^ H7NativeCodec.CONJUGATION_OPERATOR
        return chr(conjugated_byte)
        
    @staticmethod
    def is_conjugated(char: str) -> bool:
        """
        Verifica si el estado está conjugado (Mayúscula = Bit 5 OFF = 0)
        o en estado base (Minúscula = Bit 5 ON = 1).
        Para las letras ASCII, si el bit 0x20 es 0, es mayúscula.
        """
        if not char.isalpha():
            return False
            
        byte_val = ord(char)
        return (byte_val & H7NativeCodec.CONJUGATION_OPERATOR) == 0

    @staticmethod
    def compute_lagrangian(text: str) -> dict:
        """
        Calcula la estabilidad física (Lagrangiano) del texto.
        - L_symp (Conservativo): Densidad de los bits base (0-4 y 6-7).
        - L_metr (Disipativo): Tasa de estados conjugados (Mayúsculas vs Total de letras).
        """
        if not text:
            return {"L_symp": 0.0, "L_metr": 0.0, "stability_ratio": 0.0}
            
        total_letters = 0
        conjugated_count = 0
        symp_magnitude = 0
        
        for char in text:
            byte_val = ord(char)
            # Acumulamos la magnitud de los bits que NO son de quiralidad
            # Filtramos el Bit 5 usando una máscara AND con NOT(0x20) = 0xDF
            base_bits = byte_val & 0xDF
            
            # Contamos cuántos bits están encendidos en la base (peso de Hamming simple)
            symp_magnitude += bin(base_bits).count('1')
            
            if char.isalpha():
                total_letters += 1
                if H7NativeCodec.is_conjugated(char):
                    conjugated_count += 1
                    
        # L_symp es el promedio de energía en los bits base por caracter
        L_symp = symp_magnitude / len(text)
        
        # L_metr es la entropía generada por la conjugación (0.0 = todo base, 1.0 = todo conjugado)
        # Si no hay letras, la disipación es el ruido de fondo (asumimos 0)
        L_metr = (conjugated_count / total_letters) if total_letters > 0 else 0.0
        
        # Ratio de Estabilidad
        # Para evitar división por cero, sumamos un épsilon
        ratio = L_symp / (L_metr + 1e-8)
        
        return {
            "L_symp": L_symp,
            "L_metr": L_metr,
            "stability_ratio": ratio,
            "conjugated_density": L_metr
        }

    @staticmethod
    def process_informational_flow(text: str) -> str:
        """
        Ejemplo de procesamiento nativo: Un "Filtro de Coherencia".
        Obliga a que el flujo de información retorne a su estado base (Minúsculas),
        eliminando la entropía de conjugación aplicando el operador solo si está conjugado.
        """
        result = []
        for char in text:
            if H7NativeCodec.is_conjugated(char):
                result.append(H7NativeCodec.apply_conjugation(char))
            else:
                result.append(char)
        return "".join(result)
