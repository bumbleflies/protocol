# CI/CD Pipeline Guide

## Overview

This project uses GitHub Actions for continuous integration and deployment. The pipeline ensures code quality, runs tests, and automates releases.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Push / Pull Request                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   CI Pipeline (ci.yml)                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Test Matrix │  │   Coverage   │  │     Lint     │      │
│  │              │  │              │  │              │      │
│  │ • Python 3.11│  │ • pytest-cov │  │ • flake8     │      │
│  │ • Python 3.12│  │ • Codecov    │  │ • black      │      │
│  │ • Python 3.13│  │ • 45% min    │  │ • mypy       │      │
│  │              │  │              │  │              │      │
│  │ • Ubuntu     │  │              │  │              │      │
│  │ • Windows    │  │              │  │              │      │
│  │ • macOS      │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────────────────────────┐     │
│  │   Security   │  │       Build Status Check         │     │
│  │              │  │                                  │     │
│  │ • safety     │  │  ✓ All tests passed             │     │
│  │ • bandit     │  │  ✓ Coverage above threshold     │     │
│  └──────────────┘  └──────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼ (if tag v*.*.*)
┌─────────────────────────────────────────────────────────────┐
│                Release Pipeline (release.yml)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     Test     │  │    Build     │  │   Release    │      │
│  │              │  │              │  │              │      │
│  │ • Run tests  │  │ • Create pkg │  │ • Create GH  │      │
│  │ • Coverage   │  │ • wheel      │  │   release    │      │
│  │              │  │ • sdist      │  │ • Changelog  │      │
│  │              │  │              │  │ • Artifacts  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Publish to PyPI (optional)             │       │
│  │                                                  │       │
│  │  • Requires PYPI_API_TOKEN secret              │       │
│  │  • Automatic on tag push                       │       │
│  └──────────────────────────────────────────────────┘       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Workflows

### 1. Continuous Integration (ci.yml)

**Purpose:** Validate code quality and functionality on every push/PR

**Stages:**

#### Stage 1: Test Matrix
- **Platforms:** Ubuntu, Windows, macOS
- **Python:** 3.11, 3.12, 3.13
- **Total:** 9 combinations (3 OS × 3 Python)
- **Duration:** ~5-10 minutes

```bash
# What runs:
pip install -r requirements.txt
pip install -r test_requirements.txt
pytest tests/ -v
```

#### Stage 2: Coverage Analysis
- **Target:** Minimum 45% coverage
- **Reports:** XML, HTML, Terminal
- **Integration:** Codecov (optional)
- **Duration:** ~2-3 minutes

```bash
# What runs:
pytest tests/ --cov=tasks --cov=pipeline --cov-report=xml
coverage report --fail-under=45
```

#### Stage 3: Code Quality
- **Tools:** flake8, black, mypy
- **Mode:** Non-blocking (warnings only)
- **Duration:** ~1-2 minutes

```bash
# What runs:
flake8 tasks/ pipeline/
black --check tasks/ pipeline/ tests/
mypy tasks/ pipeline/
```

#### Stage 4: Security Scan
- **Tools:** safety, bandit
- **Checks:** Dependency vulnerabilities, code security issues
- **Duration:** ~1-2 minutes

```bash
# What runs:
safety check --file requirements.txt
bandit -r tasks/ pipeline/
```

#### Stage 5: Build Status
- **Purpose:** Aggregate all job results
- **Fails if:** Tests fail OR coverage below threshold

### 2. Release Pipeline (release.yml)

**Trigger:** Tag push matching `v*.*.*`

**Stages:**

#### Stage 1: Pre-release Testing
- Run full test suite
- Check coverage threshold
- **Blocks release if tests fail**

#### Stage 2: Build Distribution
- Generate wheel and source distribution
- Create setup.py dynamically
- Upload artifacts

#### Stage 3: Create GitHub Release
- Generate changelog from git history
- Create release with notes
- Attach distribution files
- Add installation instructions

#### Stage 4: Publish to PyPI (Optional)
- Requires `PYPI_API_TOKEN` secret
- Publishes package automatically
- Skips if already exists

### 3. Security Analysis (codeql.yml)

**Trigger:** Push, PR, or weekly schedule

**Features:**
- CodeQL advanced security scanning
- Detects vulnerabilities
- Security alerts in GitHub
- Extended queries

### 4. Dependency Review (dependency-review.yml)

**Trigger:** Pull requests only

**Features:**
- Reviews dependency changes
- Checks for vulnerabilities
- Comments on PR
- Blocks dangerous changes

## Setup Guide

### Prerequisites

1. **GitHub Repository Setup**
   ```bash
   # Initialize git if not already done
   git init
   git add .
   git commit -m "Initial commit with CI/CD"
   git branch -M main
   git remote add origin https://github.com/bumbleflies/protocol.git
   git push -u origin main
   ```

2. **Enable GitHub Actions**
   - Actions are enabled by default
   - Go to: Settings → Actions → General
   - Ensure "Allow all actions" is selected

### Required Secrets

#### For Codecov (Optional)
```bash
# Get token from https://codecov.io/
# Add to: Settings → Secrets → Actions → New secret
Name: CODECOV_TOKEN
Value: <your-codecov-token>
```

#### For PyPI Publishing (Optional)
```bash
# Generate API token at https://pypi.org/manage/account/token/
# Add to: Settings → Secrets → Actions → New secret
Name: PYPI_API_TOKEN
Value: <your-pypi-token>
```

### Branch Protection

1. Navigate to: **Settings → Branches**
2. Click: **Add rule**
3. Branch name pattern: `main`
4. Enable:
   - ☑ Require pull request before merging
   - ☑ Require status checks to pass
   - ☑ Require branches to be up to date
5. Select required checks:
   - `Test on Python 3.12 (ubuntu-latest)`
   - `Code Coverage`
   - `Build Status`
6. Save changes

## Usage

### Running CI Locally

Before pushing, run checks locally:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r test_requirements.txt

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=tasks --cov=pipeline

# Run linting
flake8 tasks/ pipeline/
black --check tasks/ pipeline/ tests/
mypy tasks/ pipeline/

# Run security checks
safety check --file requirements.txt
bandit -r tasks/ pipeline/
```

### Creating a Release

#### Automatic Release (Recommended)

```bash
# 1. Update version in pyproject.toml
# 2. Commit all changes
git add .
git commit -m "Prepare release v1.0.0"
git push origin main

# 3. Create annotated tag
git tag -a v1.0.0 -m "Release v1.0.0

Features:
- SOLID principles refactoring
- Configuration-based pipeline
- Comprehensive test suite

Fixes:
- Platform compatibility
- Resource cleanup

Documentation:
- Added CI/CD pipeline
- Updated architecture docs"

# 4. Push tag to trigger release
git push origin v1.0.0

# GitHub Actions will automatically:
# - Run tests
# - Build packages
# - Create GitHub release
# - Publish to PyPI (if configured)
```

#### Manual Release

```bash
# Build locally
python -m build

# Check distribution
twine check dist/*

# Upload to TestPyPI (test first)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### Monitoring Workflows

#### View Workflow Runs

1. Go to: **Actions** tab in GitHub
2. Click on a workflow name
3. Select a run to see details
4. Click on jobs to see step logs

#### Download Artifacts

1. Open a completed workflow run
2. Scroll to **Artifacts** section
3. Available artifacts:
   - `test-results-*` - JUnit XML test results
   - `coverage-report` - HTML coverage report
   - `security-reports` - Security scan results
   - `dist` - Distribution packages (releases only)

## Troubleshooting

### Common Issues

#### 1. Tests Fail Only in CI

**Symptoms:** Tests pass locally but fail in GitHub Actions

**Causes:**
- Path separator differences (Windows vs Unix)
- Timezone differences
- Missing environment variables
- Different Python version

**Solutions:**
```python
# Use pathlib for cross-platform paths
from pathlib import Path
path = Path("dir") / "file.txt"

# Use tempfile for temp directories
import tempfile
temp_dir = Path(tempfile.gettempdir())

# Mock environment variables in tests
@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
```

#### 2. Coverage Below Threshold

**Symptoms:** Coverage check fails with "coverage below 45%"

**Solutions:**
```bash
# Option 1: Add more tests
pytest tests/ -v --cov=tasks --cov=pipeline --cov-report=term-missing

# Option 2: Adjust threshold (in ci.yml)
coverage report --fail-under=40  # Lower threshold

# Option 3: Exclude uncovered code
# Add to pyproject.toml [tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
]
```

#### 3. Release Workflow Doesn't Trigger

**Symptoms:** Tag pushed but no release created

**Causes:**
- Tag format doesn't match `v*.*.*`
- Workflow file has syntax errors
- Insufficient permissions

**Solutions:**
```bash
# Check tag format (must start with 'v')
git tag -a v1.0.0 -m "Release 1.0.0"  # ✓ Correct
git tag -a 1.0.0 -m "Release 1.0.0"   # ✗ Wrong

# Check workflow syntax
# Use: https://rhysd.github.io/actionlint/

# Check repository permissions
# Settings → Actions → General → Workflow permissions
# Enable: "Read and write permissions"
```

#### 4. PyPI Upload Fails

**Symptoms:** "Package already exists" or authentication error

**Solutions:**
```bash
# Increment version in pyproject.toml
version = "1.0.1"  # Not 1.0.0

# Check PYPI_API_TOKEN secret is set
# Settings → Secrets → Actions → PYPI_API_TOKEN

# Test with TestPyPI first
# Add separate secret: TEST_PYPI_API_TOKEN
```

### Getting Help

1. **Check Workflow Logs**
   - Click on failed job
   - Expand failed step
   - Look for error messages

2. **Enable Debug Logging**
   ```bash
   # Add to workflow file:
   env:
     ACTIONS_STEP_DEBUG: true
     ACTIONS_RUNNER_DEBUG: true
   ```

3. **Run Locally**
   ```bash
   # Use act to run GitHub Actions locally
   brew install act  # macOS
   # or
   choco install act-cli  # Windows

   act push  # Simulate push event
   ```

## Best Practices

### Commit Messages

Use conventional commits:
```
feat: add new OCR provider
fix: resolve platform compatibility issue
docs: update CI/CD documentation
test: add tests for registry
refactor: improve task processor architecture
chore: update dependencies
```

### Versioning

Follow semantic versioning (SemVer):
- `v1.0.0` - Major: Breaking changes
- `v1.1.0` - Minor: New features, backward compatible
- `v1.0.1` - Patch: Bug fixes, backward compatible

### Pre-commit Checks

Install pre-commit hooks:
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Pull Request Process

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit
3. Run tests locally: `pytest tests/ -v`
4. Push branch: `git push origin feature/my-feature`
5. Create PR on GitHub
6. Wait for CI checks to pass
7. Request review
8. Merge after approval

## Metrics & Monitoring

### Key Metrics

- **Test Pass Rate:** Target 100%
- **Code Coverage:** Minimum 45%, target 80%
- **Build Time:** Target < 15 minutes
- **Security Issues:** Target 0 high/critical

### Monitoring

- **GitHub Actions Dashboard:** View workflow status
- **Codecov:** Track coverage trends
- **Dependabot:** Monitor dependency updates
- **CodeQL:** Track security vulnerabilities

## Resources

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Python Packaging Guide](https://packaging.python.org/)
