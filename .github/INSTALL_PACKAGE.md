# Installing h7-metriplectic-os Package from GitHub Packages

## Prerequisites

You need to authenticate with GitHub Packages to install this package.

### Step 1: Create a Personal Access Token (PAT)

1. Go to [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Give it a name like "gh-packages-token"
4. Select scopes:
   - ✅ `read:packages` (to download packages)
   - ✅ `repo` (optional, for repository access)
5. Generate and **copy the token** (you won't see it again)

### Step 2: Configure your machine

#### Option A: Using `.netrc` (Recommended for Unix/Linux/Mac)

Create or edit `~/.netrc`:

```
machine github.com
login YOUR_GITHUB_USERNAME
password YOUR_PERSONAL_ACCESS_TOKEN
```

Set permissions:
```bash
chmod 600 ~/.netrc
```

#### Option B: Using environment variables

```bash
export GITHUB_TOKEN=YOUR_PERSONAL_ACCESS_TOKEN
```

#### Option C: Using pip directly

```bash
pip install h7-metriplectic-os --index-url https://YOUR_USERNAME:YOUR_TOKEN@npm.pkg.github.com/simple/
```

### Step 3: Install the package

```bash
pip install h7-metriplectic-os
```

Or with a specific version:

```bash
pip install h7-metriplectic-os==0.1.0
```

## Troubleshooting

- **401 Unauthorized**: Check your token and username
- **404 Not Found**: Verify the package URL and version
- **Connection refused**: Ensure you have internet access

## Using in your projects

Once installed, import and use the package:

```python
import h7_metriplectic_os

print(h7_metriplectic_os.__version__)
```
