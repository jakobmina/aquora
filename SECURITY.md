# SECURITY POLICY — Smopsys Quantum & AI Independent Research Laboratory
## Aquora · H7 Metriplectic OS · DIT Framework

**Maintainer:** Jacobo Tlacaelel Mina Rodríguez  
**Repository:** `jakobmina/aquora`  
**Last updated:** May 2026  
**Classification:** Public Research / Open Source (MIT)

---

## Supported Versions

| Component | Version | Security Support |
|---|---|---|
| H7 Metriplectic OS | `0.1.x` | ✅ Active |
| MetriplexBridge | `0.1.x` | ✅ Active |
| C Kernel (`libmetriplex_core.so`) | `0.1.x` | ✅ Active |
| h7_sysdaemon | `0.1.x` | ✅ Active |
| Older releases | `< 0.1.0` | ❌ Not supported |

---

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately via:

- **Email:** `smopsys@gmail.com` (preferred)
- **GitHub:** Use [Private Security Advisories](https://github.com/jakobmina/aquora/security/advisories/new)

Include in your report:

1. Component affected (`h7_sysdaemon`, `core_physics/`, API, etc.)
2. Description of the vulnerability and its potential impact
3. Steps to reproduce
4. Suggested fix if you have one

**Response SLA:** acknowledgment within 72 hours, patch or mitigation within 14 days for critical issues.

---

## Credential & Secret Management

This repository interacts with external services (Google Cloud Vertex AI, q3as quantum backend, Firebase). The following rules are **mandatory** and enforced via `.gitignore`:

### Never commit:
```
credentials.json          # Google Cloud / Vertex AI service account key
*.pem / *.key             # TLS/SSH private keys
.env                      # Environment variable files with secrets
firebase.json (with keys) # Firebase private config
q3as auth tokens          # q3as backend credentials
```

### Correct pattern for credentials:
```python
# WRONG — hardcoded in source
client = Client(Credentials("my-secret-key-here"))

# CORRECT — loaded from file outside repo or from env
client = Client(Credentials.load(os.environ.get("Q3AS_CREDENTIALS_PATH", "credentials.json")))
```

### Environment variables for CI/CD:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
export Q3AS_CREDENTIALS_PATH=/path/to/q3as_credentials.json
export FIREBASE_PROJECT_ID=prone1-429307
```

The `.firebaserc` file in this repo contains only the project ID (`prone1-429307`), which is not a secret. It is safe to commit.

---

## C Kernel Security (`core_physics/`)

The compiled shared library `libmetriplex_core.so` / `h7_kernel.so` is loaded at runtime via `ctypes`. This introduces native code execution risks.

### Rules:

**Build from source only.** Never load a pre-compiled `.so` from an untrusted source:
```bash
# Always compile yourself
cd core_physics && make clean && make
```

**Verify the compiled binary** before loading in production:
```bash
sha256sum core_physics/libmetriplex_core.so
# Compare against known-good hash from the release notes
```

**Pointer safety.** The C kernel uses raw `ctypes` pointers. The Python wrappers in `core_physics/h7_wrapper.py` validate array dimensions before passing to C. Do not bypass these wrappers to call `_kernel` directly.

**Memory bounds.** The `invert_matrix_4x4` function in `h7_kernel.c` uses a fixed 4×4 assumption. Passing matrices of wrong shape via `ctypes` will cause undefined behavior (buffer overread). The Python wrapper enforces this with explicit padding to 4×4.

---

## Daemon Security (`h7_sysdaemon.py`)

The daemon reads real hardware telemetry (`psutil`) and runs continuously. When deployed as a `systemd` service:

**Recommended systemd hardening:**
```ini
[Service]
User=h7daemon
Group=h7daemon
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/h7
CapabilityBoundingSet=
AmbientCapabilities=
```

**Do NOT run as root.** The daemon only needs read access to `/proc` (via `psutil`). A dedicated low-privilege user is sufficient.

**Tick rate as an attack surface.** Very high tick rates (`tick_rate < 0.1`) combined with disk I/O monitoring can cause excessive system load. The minimum recommended tick rate is `0.2` Hz in production.

---

## API Security (`api.py` — FastAPI)

The FastAPI server exposes the H7 experiment dashboard. It is designed as a **local research tool**, not a public internet service.

### Current security posture:
- No authentication on any endpoint
- No rate limiting
- Binds to `0.0.0.0` (all interfaces)

### Required hardening before any network exposure:

**1. Restrict binding to localhost:**
```python
uvicorn.run(app, host="127.0.0.1", port=port)
```

**2. Add API key middleware if exposing externally:**
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.environ.get("H7_API_KEY"):
        raise HTTPException(status_code=403)
```

**3. The `/api/intent` endpoint calls Vertex AI with user-supplied prompt text.** Treat this as an LLM injection surface. Do not expose it to untrusted users without input sanitization.

---

## Dependency Security

### Python dependencies — known risk areas:

| Package | Risk | Mitigation |
|---|---|---|
| `qiskit` | Large attack surface, complex C extensions | Pin to known-good version in `requirements.txt` |
| `ctypes` (stdlib) | Direct memory access | Only call via validated wrappers in `h7_wrapper.py` |
| `psutil` | Reads sensitive system info | Acceptable for daemon; do not log raw values externally |
| `vertexai` | GCP credentials in memory | Use short-lived service account tokens; revoke if leaked |
| `fastapi` + `uvicorn` | Network exposure | Keep local; add auth before any external deployment |

### Pinning dependencies:
```bash
pip freeze > requirements-lock.txt
```
Use `requirements-lock.txt` in reproducible deployments. The `pyproject.toml` specifies minimum versions only — these are not sufficient for production security.

### Checking for known CVEs:
```bash
pip install pip-audit
pip-audit -r requirements-lock.txt
```

---

## Firebase / GCP Security

The project uses Firebase project `prone1-429307` (visible in `.firebaserc`). This project ID is not a secret but serves as a target identifier.

- **Rotate service account keys** every 90 days or immediately if `credentials.json` is accidentally committed.
- **Least privilege:** The service account used for this project should have only the permissions required (e.g., `Vertex AI User`, not `Project Owner`).
- **Audit logs:** Enable GCP audit logging for Vertex AI API calls from this project.

If `credentials.json` is ever accidentally committed:
```bash
# Immediately:
# 1. Revoke the key in GCP Console → IAM → Service Accounts
# 2. Remove from git history:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch credentials.json" \
  --prune-empty --tag-name-filter cat -- --all
git push origin --force --all
# 3. Generate a new key
```

---

## Data Classification

| Data type | Classification | Storage rule |
|---|---|---|
| H7 framework source code | Public | Commit freely |
| VQE experiment results (JSON) | Public research | Commit freely |
| `credentials.json` | **SECRET** | Never commit |
| CL1 / Cortical Labs HDF5 exports | Confidential | Not in this repo |
| Submission CSVs (`submission.csv`) | Public research | Commit freely |
| `.env` files | **SECRET** | Never commit |

---

## Quantum Backend Security (q3as)

The `run_vqe_maxcut.py` module submits jobs to the q3as quantum backend. Job submissions include the Hamiltonian edge weights derived from H7 constants.

- The edge weights in submitted jobs (e.g., `0.695864585574`) are derived from public H7 constants and contain no sensitive information.
- Credentials for q3as are loaded from `credentials_path` parameter. Default is `"credentials.json"` — **keep this file outside version control**.
- Job results returned from q3as are stored in `submission.csv`. These are public research results and safe to commit.

---

## Known Limitations (Non-Goals)

This project is a **research laboratory tool**, not a production system. The following are known limitations that are out of scope for this security policy:

- No multi-user authentication
- No audit trail for experiment runs
- No data encryption at rest for result files
- The `h7_sysdaemon.py` is not hardened for adversarial environments

These limitations are acceptable for the current research context. Before any production deployment or public cloud hosting, a full security review should be conducted.

---

## Security Checklist for Contributors

Before opening a PR, verify:

- [ ] No credentials, API keys, or tokens in any committed file
- [ ] `.gitignore` is not modified to unblock secret files
- [ ] Any new `ctypes` calls go through validated Python wrappers
- [ ] New API endpoints on `api.py` do not expose system-level information
- [ ] Dependencies added to `pyproject.toml` have been checked with `pip-audit`
- [ ] New C code in `core_physics/` does not introduce unchecked buffer operations

---

*smokApp Quantum & AI Independent Research Laboratory — Tlaxcala, México*  
*Framework: Aquora · H7 Metriplectic Hierarchy OS*
