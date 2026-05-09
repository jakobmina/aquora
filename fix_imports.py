import os
import re

modules = [
    "h7_sysdaemon", "h7_framework", "api", "h7_quaternion", "h7_cascade_qiskit",
    "h7_bayesian_oracle", "h7_cascade_maxcut", "h7_native_codec",
    "metriplex_bridge", "vertex_h7_bridge", "covariance_asymmetry",
    "endian", "h_logic", "utf8_exp3"
]

def fix_imports(filepath, is_internal=False):
    with open(filepath, 'r') as f:
        content = f.read()

    for mod in modules:
        # Match 'import module' -> 'import h7_metriplectic_os.module'
        if not is_internal:
            content = re.sub(rf'^import {mod}(\s|$)', rf'import h7_metriplectic_os.{mod}\1', content, flags=re.MULTILINE)
            content = re.sub(rf'^from {mod} import', rf'from h7_metriplectic_os.{mod} import', content, flags=re.MULTILINE)
        else:
            # For internal files within the package, we can use relative imports
            content = re.sub(rf'^import {mod}(\s|$)', rf'from . import {mod}\1', content, flags=re.MULTILINE)
            content = re.sub(rf'^from {mod} import', rf'from .{mod} import', content, flags=re.MULTILINE)
            
    with open(filepath, 'w') as f:
        f.write(content)

# Internal files
pkg_dir = 'h7_metriplectic_os'
for filename in os.listdir(pkg_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        fix_imports(os.path.join(pkg_dir, filename), is_internal=True)

# External files (root and tests)
for filename in ['main.py', 'run_vqe_maxcut.py', 'generate_submission.py']:
    if os.path.exists(filename):
        fix_imports(filename, is_internal=False)

tests_dir = 'tests'
if os.path.exists(tests_dir):
    for filename in os.listdir(tests_dir):
        if filename.endswith('.py'):
            fix_imports(os.path.join(tests_dir, filename), is_internal=False)

print("Imports fixed.")
