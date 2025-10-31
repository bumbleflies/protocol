# Semantic Release Documentation

This project uses [Python Semantic Release](https://python-semantic-release.readthedocs.io/) for automated versioning and releasing.

## Overview

Semantic Release automates the entire package release workflow including:
- Determining the next version number based on commit messages
- Generating release notes/changelog
- Updating version numbers in code
- Creating git tags
- Creating GitHub releases
- Publishing to PyPI (optional)

## Commit Message Format

Semantic Release uses [Conventional Commits](https://www.conventionalcommits.org/) to determine version bumps:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types and Version Bumps

| Type | Version Bump | Description | Example |
|------|--------------|-------------|---------|
| `feat` | **MINOR** (0.x.0) | New feature | `feat: add flipchart detection algorithm` |
| `fix` | **PATCH** (0.0.x) | Bug fix | `fix: handle missing API key gracefully` |
| `perf` | **PATCH** (0.0.x) | Performance improvement | `perf: optimize image processing` |
| `docs` | None | Documentation only | `docs: update README with API key instructions` |
| `style` | None | Code style changes | `style: format with black` |
| `refactor` | None | Code refactoring | `refactor: extract OCR provider interface` |
| `test` | None | Add or update tests | `test: add tests for main.py` |
| `chore` | None | Maintenance tasks | `chore: update dependencies` |
| `ci` | None | CI configuration | `ci: add semantic-release workflow` |
| `build` | None | Build system changes | `build: update pyproject.toml` |

### Breaking Changes

To trigger a **MAJOR** version bump (x.0.0), add `BREAKING CHANGE:` in the commit body or footer:

```
feat: redesign API interface

BREAKING CHANGE: The `process()` method now returns a Result object instead of FileTask
```

Or use the `!` notation:

```
feat!: redesign API interface
```

### Examples

```bash
# Patch release (1.0.0 → 1.0.1)
git commit -m "fix: handle empty OCR results correctly"

# Minor release (1.0.0 → 1.1.0)
git commit -m "feat: add support for PDF annotation filtering"

# Major release (1.0.0 → 2.0.0)
git commit -m "feat!: redesign pipeline configuration

BREAKING CHANGE: Configuration format changed from JSON to YAML"

# No release (documentation only)
git commit -m "docs: add semantic release documentation"
```

## How It Works

### Automatic Releases (Recommended)

When you push to the `main`/`master` branch, the semantic-release workflow automatically:

1. Runs tests to ensure code quality
2. Analyzes commit messages since the last release
3. Determines the next version number
4. Updates `pyproject.toml` with the new version
5. Generates/updates `CHANGELOG.md`
6. Creates a git tag (e.g., `v1.2.0`)
7. Creates a GitHub release with notes
8. Optionally publishes to PyPI

**Configuration:** [`.github/workflows/semantic-release.yml`](.github/workflows/semantic-release.yml)

### Manual Releases

You can also trigger releases manually:

```bash
# Preview what would be released (dry-run)
semantic-release --noop version

# Create a release locally
semantic-release version

# Publish the release
semantic-release publish
```

## Configuration

Semantic Release is configured in [`pyproject.toml`](pyproject.toml):

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "python -m build"
commit_message = "chore(release): {version}"
tag_format = "v{version}"
major_on_zero = true  # Allow 0.x.y versions before 1.0.0

[tool.semantic_release.branches.main]
match = "(main|master)"
prerelease = false

[tool.semantic_release.changelog.default_templates]
changelog_file = "CHANGELOG.md"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["build", "chore", "ci", "docs", "feat", "fix", "perf", "style", "refactor", "test"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]
```

## GitHub Actions Setup

### Required Secrets

For PyPI publishing, add this secret to your GitHub repository:
- `PYPI_API_TOKEN`: Your PyPI API token from https://pypi.org/manage/account/token/

### Permissions

The workflow uses `GITHUB_TOKEN` (automatically provided) with these permissions:
- `contents: write` - Create tags and releases
- `issues: write` - Update related issues
- `pull-requests: write` - Update related PRs

## Workflow Files

### 1. Semantic Release Workflow (`.github/workflows/semantic-release.yml`)

**Triggered by:** Pushes to `main`/`master` branch
**Purpose:** Automatic versioning and releasing

**Steps:**
1. Run tests
2. Analyze commits and determine next version
3. Update version files
4. Generate changelog
5. Create git tag
6. Create GitHub release
7. Publish to PyPI (if configured)

### 2. Release Workflow (`.github/workflows/release.yml`)

**Triggered by:** Version tags (e.g., `v1.2.3`)
**Purpose:** Build and distribute releases created by semantic-release

**Steps:**
1. Run tests
2. Build distribution packages
3. Create GitHub release with artifacts
4. Publish to PyPI (if configured)

## Best Practices

### 1. Write Clear Commit Messages

```bash
# ❌ Bad
git commit -m "fix stuff"
git commit -m "update code"

# ✅ Good
git commit -m "fix: handle missing NVIDIA_API_KEY gracefully"
git commit -m "feat: add automatic OCR skip when API key is missing"
```

### 2. Use Conventional Commits Consistently

- Always start with a type: `feat`, `fix`, `docs`, etc.
- Use lowercase for type and subject
- Keep subject line under 72 characters
- Use imperative mood: "add" not "added" or "adds"

### 3. Group Related Changes

```bash
# Multiple related changes in one commit
git add main.py tests/test_main.py README.md
git commit -m "feat: add graceful API key error handling

- Auto-skip OCR tasks when NVIDIA_API_KEY is missing
- Show helpful error message with setup instructions
- Add tests for missing API key scenario
- Update README with API key documentation"
```

### 4. Test Before Pushing

```bash
# Run tests locally
pytest tests/ -v

# Preview what would be released
semantic-release --noop version
```

### 5. Skip CI for Non-Releasable Commits

Use `[skip ci]` in commit messages to skip CI for documentation-only changes:

```bash
git commit -m "docs: fix typo in README [skip ci]"
```

## Troubleshooting

### No Release Created

**Problem:** Pushed to main but no release was created
**Solutions:**
- Check that commits follow conventional commit format
- Verify the commit contains a type that triggers releases (`feat`, `fix`, `perf`)
- Check GitHub Actions logs for errors
- Ensure you're not in a `chore(release):` commit cycle

### Version Not Incremented

**Problem:** Release created but version stayed the same
**Solutions:**
- Ensure commits since last release contain `feat` or `fix` types
- Check that `major_on_zero = true` if expecting 0.x.y versions
- Verify `pyproject.toml` version is being updated correctly

### PyPI Publishing Failed

**Problem:** Release created but PyPI upload failed
**Solutions:**
- Verify `PYPI_API_TOKEN` secret is configured in repository settings
- Check token has permission to upload to the package
- Ensure package name in `pyproject.toml` is available on PyPI
- Review GitHub Actions logs for specific error message

## Examples

### Feature Release

```bash
# Develop feature
git checkout -b feature/nvidia-ocr
# ... make changes ...
git add .
git commit -m "feat: add NVIDIA OCR provider abstraction"
git push origin feature/nvidia-ocr

# Create PR and merge to main
# → Semantic release automatically creates v1.1.0
```

### Bug Fix Release

```bash
git checkout -b fix/api-key-handling
# ... make changes ...
git add main.py tests/test_main.py
git commit -m "fix: handle missing NVIDIA_API_KEY gracefully

- Auto-skip OCR tasks when API key is not set
- Display helpful error message with setup instructions
- Add tests for missing API key scenario"
git push origin fix/api-key-handling

# Merge to main
# → Semantic release automatically creates v1.0.1
```

### Breaking Change Release

```bash
git checkout -b refactor/pipeline-config
# ... make changes ...
git add .
git commit -m "feat!: migrate from JSON to YAML configuration

BREAKING CHANGE: Pipeline configuration now uses YAML format.
Update your config files from config.json to pipeline_config.yaml.
See migration guide in CLAUDE.md."
git push origin refactor/pipeline-config

# Merge to main
# → Semantic release automatically creates v2.0.0
```

## Additional Resources

- [Python Semantic Release Documentation](https://python-semantic-release.readthedocs.io/)
- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
