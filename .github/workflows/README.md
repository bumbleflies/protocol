# GitHub Actions CI/CD Pipeline

This directory contains GitHub Actions workflows for automated testing, quality checks, and releases.

## Workflows

### 1. CI - Test & Quality (`ci.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches

**Jobs:**

#### Test Matrix
- Runs on: Ubuntu, Windows, macOS
- Python versions: 3.11, 3.12, 3.13
- Executes full test suite with pytest
- Uploads test results as artifacts

#### Coverage
- Generates code coverage report
- Uploads to Codecov (optional)
- Requires minimum 45% coverage
- Generates HTML coverage report

#### Lint
- Runs `flake8` for code style
- Runs `black` for formatting check
- Runs `mypy` for type checking
- All checks are non-blocking (continue-on-error)

#### Security
- Runs `safety` for dependency vulnerabilities
- Runs `bandit` for security issues
- Uploads security reports

#### Build Status
- Final check that aggregates all job results
- Fails if tests or coverage checks fail

**Usage:**
```bash
# Workflow runs automatically on push/PR
# To see results, check the Actions tab in GitHub
```

### 2. Release (`release.yml`)

**Triggers:**
- Push tags matching `v*.*.*` (e.g., `v1.0.0`, `v2.1.3`)

**Jobs:**

#### Test Before Release
- Runs full test suite
- Checks coverage threshold
- Blocks release if tests fail

#### Build Distribution
- Creates Python distribution packages (wheel & sdist)
- Generates setup.py automatically
- Uploads build artifacts

#### Create GitHub Release
- Generates changelog from git history
- Creates GitHub release with notes
- Attaches distribution files
- Includes installation instructions

#### Publish to PyPI (Optional)
- Publishes package to PyPI
- Requires `PYPI_API_TOKEN` secret
- Skips if package already exists

**Usage:**
```bash
# Create a new release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# This will automatically:
# 1. Run tests
# 2. Build distribution
# 3. Create GitHub release
# 4. Publish to PyPI (if configured)
```

### 3. CodeQL Security Analysis (`codeql.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches
- Scheduled: Every Monday at midnight

**Features:**
- Advanced security scanning
- Detects common vulnerabilities
- Provides security alerts
- Runs extended security queries

### 4. Dependency Review (`dependency-review.yml`)

**Triggers:**
- Pull requests to `main`, `master`, or `develop` branches

**Features:**
- Reviews dependency changes in PRs
- Checks for known vulnerabilities
- Comments summary on PR
- Fails on moderate or higher severity issues

## Setup Instructions

### 1. Required Secrets

Add these secrets in GitHub repository settings:

**For PyPI Publishing (optional):**
- `PYPI_API_TOKEN`: Your PyPI API token
  - Get from: https://pypi.org/manage/account/token/

**For Codecov (optional):**
- `CODECOV_TOKEN`: Your Codecov upload token
  - Get from: https://codecov.io/

### 2. Branch Protection Rules

Recommended settings for `main` branch:

1. Go to Settings → Branches → Branch protection rules
2. Add rule for `main`:
   - ✅ Require pull request before merging
   - ✅ Require status checks to pass before merging
   - Required checks:
     - `Test on Python 3.12 (ubuntu-latest)`
     - `Code Coverage`
     - `Build Status`
   - ✅ Require branches to be up to date before merging

### 3. Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/bumbleflies/protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/bumbleflies/protocol/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bumbleflies/protocol/branch/main/graph/badge.svg)](https://codecov.io/gh/bumbleflies/protocol)
[![CodeQL](https://github.com/bumbleflies/protocol/actions/workflows/codeql.yml/badge.svg)](https://github.com/bumbleflies/protocol/actions/workflows/codeql.yml)
[![Release](https://github.com/bumbleflies/protocol/actions/workflows/release.yml/badge.svg)](https://github.com/bumbleflies/protocol/actions/workflows/release.yml)
```

### 4. Release Process

#### Automatic Release (Recommended)

```bash
# 1. Ensure all changes are committed and pushed
git add .
git commit -m "Prepare release v1.0.0"
git push origin main

# 2. Create and push tag
git tag -a v1.0.0 -m "Release version 1.0.0

- Feature: SOLID principles refactoring
- Feature: Configuration-based pipeline
- Fix: Platform compatibility issues
- Docs: Comprehensive documentation"

git push origin v1.0.0

# 3. GitHub Actions will automatically:
#    - Run all tests
#    - Build distribution packages
#    - Create GitHub release with changelog
#    - Publish to PyPI (if configured)
```

#### Manual Release (Alternative)

```bash
# Build locally
python -m build

# Test the distribution
pip install dist/flipchart_ocr_pipeline-1.0.0-py3-none-any.whl

# Upload to PyPI
twine upload dist/*
```

## Workflow Status

### Check Workflow Runs

1. Go to: https://github.com/bumbleflies/protocol/actions
2. Click on a workflow to see runs
3. Click on a run to see job details
4. Click on a job to see step logs

### Download Artifacts

1. Go to workflow run
2. Scroll to "Artifacts" section
3. Download:
   - `test-results-*` - JUnit test results
   - `coverage-report` - HTML coverage report
   - `security-reports` - Security scan results
   - `dist` - Distribution packages

## Troubleshooting

### Tests Fail on Windows

**Issue:** Path separators or encoding issues

**Solution:**
- Ensure all paths use `pathlib.Path`
- Use `tempfile.gettempdir()` instead of `/tmp/`

### Coverage Below Threshold

**Issue:** Coverage check fails with < 45%

**Solution:**
- Add more tests for uncovered code
- Or adjust threshold in `ci.yml`: `--fail-under=45`

### Release Workflow Fails

**Issue:** Tag format doesn't match

**Solution:**
- Use format: `vX.Y.Z` (e.g., `v1.0.0`)
- Ensure tag starts with lowercase `v`

### PyPI Upload Fails

**Issue:** Package name already exists or credentials invalid

**Solution:**
- Check `PYPI_API_TOKEN` secret is set correctly
- Ensure package version is incremented
- Use unique package name in setup.py

## Advanced Configuration

### Customize Test Matrix

Edit `ci.yml` matrix section:

```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  python-version: ['3.11', '3.12', '3.13']
```

### Adjust Coverage Threshold

Edit `ci.yml` coverage job:

```yaml
- name: Check coverage threshold
  run: |
    coverage report --fail-under=50  # Change from 45 to 50
```

### Add Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

Install: `pre-commit install`

## Best Practices

1. **Write Tests First**: Ensure new features have tests
2. **Check Locally**: Run `pytest tests/` before pushing
3. **Semantic Versioning**: Follow `vMAJOR.MINOR.PATCH`
4. **Descriptive Commits**: Write clear commit messages
5. **Review PRs**: Wait for CI checks before merging
6. **Tag Releases**: Use annotated tags with descriptions
7. **Monitor Security**: Review Dependabot alerts regularly

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [Python Packaging Guide](https://packaging.python.org/)
