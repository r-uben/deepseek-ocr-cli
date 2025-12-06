# Publishing Guide for deepseek-ocr-cli

## ✅ Completed Steps

1. ✅ Cleaned up investigation files
2. ✅ Updated README with badges and performance info
3. ✅ Updated pyproject.toml with repository metadata
4. ✅ Initialized git repository
5. ✅ Created initial commit

## 📦 Next Steps

### 1. Create GitHub Repository

**Option A: Using GitHub CLI (recommended)**
```bash
cd "/Users/rubenffuertes/Library/CloudStorage/GoogleDrive-fernandezfuertesruben@gmail.com/My Drive/Toolkits/deepseek-ocr-cli"

# Create public repository
gh repo create deepseek-ocr-cli --public --source=. --remote=origin --push

# Add description
gh repo edit --description "CLI tool for OCR using DeepSeek-OCR model via Ollama. Local processing with zero cloud dependencies."

# Add topics
gh repo edit --add-topic ocr,deepseek,cli,pdf,document-processing,ollama,python
```

**Option B: Manual (if gh not available)**
1. Go to https://github.com/new
2. Repository name: `deepseek-ocr-cli`
3. Description: "CLI tool for OCR using DeepSeek-OCR model via Ollama. Local processing with zero cloud dependencies."
4. Public repository
5. **Don't** initialize with README (we already have one)
6. Click "Create repository"
7. Then run:
```bash
cd "/Users/rubenffuertes/Library/CloudStorage/GoogleDrive-fernandezfuertesruben@gmail.com/My Drive/Toolkits/deepseek-ocr-cli"
git remote add origin https://github.com/r-uben/deepseek-ocr-cli.git
git branch -M main
git push -u origin main
```

### 2. Create Release Tag

```bash
cd "/Users/rubenffuertes/Library/CloudStorage/GoogleDrive-fernandezfuertesruben@gmail.com/My Drive/Toolkits/deepseek-ocr-cli"

# Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0: Initial public release"
git push origin v0.2.0

# Or use GitHub CLI
gh release create v0.2.0 --title "v0.2.0" --notes "Initial public release

Features:
- Local OCR processing via Ollama
- Support for PDFs and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- Batch processing with progress tracking
- Clean markdown output with table conversion
- 10-minute timeout for complex pages
- Rich CLI with progress bars"
```

### 3. Publish to PyPI (Optional)

**Prerequisites:**
- PyPI account at https://pypi.org/account/register/
- Poetry installed

**Steps:**
```bash
cd "/Users/rubenffuertes/Library/CloudStorage/GoogleDrive-fernandezfuertesruben@gmail.com/My Drive/Toolkits/deepseek-ocr-cli"

# Build package
poetry build

# Configure PyPI credentials (first time only)
poetry config pypi-token.pypi YOUR_PYPI_TOKEN

# Publish to PyPI
poetry publish

# Or publish to TestPyPI first (recommended)
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry publish -r testpypi

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ deepseek-ocr-cli
```

**Get PyPI Token:**
1. Go to https://pypi.org/manage/account/token/
2. Create new token with name "deepseek-ocr-cli"
3. Copy token (starts with `pypi-`)
4. Store securely

### 4. Add GitHub Actions (Optional)

Create `.github/workflows/tests.yml`:
```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH

    - name: Install dependencies
      run: |
        poetry install

    - name: Run tests
      run: |
        poetry run pytest

    - name: Run linters
      run: |
        poetry run black --check .
        poetry run ruff check .
```

### 5. Update Installation Instructions

Once published to PyPI, update README installation section:

```markdown
### 3. Install the CLI

**From PyPI (recommended):**
```bash
pip install deepseek-ocr-cli
```

**From source:**
```bash
git clone https://github.com/r-uben/deepseek-ocr-cli.git
cd deepseek-ocr-cli
poetry install
```
\`\`\`

---

## 🎯 Quick Command Summary

```bash
# 1. Create GitHub repo (if using gh CLI)
gh repo create deepseek-ocr-cli --public --source=. --remote=origin --push

# 2. Create release
gh release create v0.2.0 --title "v0.2.0" --notes "Initial public release"

# 3. Build and publish to PyPI
poetry build
poetry publish

# Done! ✅
```

---

## 📋 Post-Publication Checklist

- [ ] Repository created on GitHub
- [ ] Code pushed to GitHub
- [ ] Release tag created (v0.2.0)
- [ ] README badges working
- [ ] Published to PyPI (optional)
- [ ] Installation from PyPI works
- [ ] Repository topics added
- [ ] Star your own repo! ⭐

---

## 📊 Repository Stats to Track

After publication, monitor:
- GitHub stars ⭐
- PyPI downloads 📦
- Issues opened 🐛
- Pull requests 🔀

---

**Current Status:** Ready to publish! 🚀

All code is cleaned, documented, and committed. You just need to run the commands above.
