import xyz_parse
input_data = """2
Sample H2 molecule
H 0.3710 0.0 0.0
H -0.3710 0.0 0.0"""
try:
    molecule = xyz_parse.Molecule.parse(input_data)
    print(f"Success: {repr(molecule)}")
except Exception as e:
    print(f"Error: {e}")
