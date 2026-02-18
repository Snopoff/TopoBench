#!/bin/bash -l
set -euo pipefail

# ==============================================================================
# 🛠️  TopoBench Environment Setup Script (Py3.11 + Dynamic CUDA)
# ==============================================================================
# usage: bash uv_env_setup.sh [cpu|cu118|cu121]
# ==============================================================================

PLATFORM="${1:-cpu}"

# Visual Header
echo ""
echo "======================================================="
echo "🚀 Initializing TopoBench Environment ($PLATFORM)"
echo "======================================================="

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
TORCH_VER="2.3.0"

if [[ "$OSTYPE" == "darwin"* ]] && [[ "$PLATFORM" != "cpu" ]]; then
    echo "❌ Error: CUDA targets (cu118/cu121) are Linux-only. On macOS use: cpu."
    exit 1
fi

if [ "$PLATFORM" == "cpu" ]; then
    TARGET_INDEX="pytorch-cpu"
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html"
elif [ "$PLATFORM" == "cu118" ]; then
    TARGET_INDEX="pytorch-cu118"
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+cu118.html"
elif [ "$PLATFORM" == "cu121" ]; then
    TARGET_INDEX="pytorch-cu121"
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"
else
    echo "❌ Error: Invalid platform '$PLATFORM'. Use: cpu, cu118, or cu121."
    exit 1
fi

echo "⚙️  Updating pyproject.toml..."

# 1. Update the 'find-links' URL for PyG extensions
if [[ "$OSTYPE" == "darwin"* ]]; then
    # MacOS sed
    sed -i '' "s|find-links = \[\".*\"\]|find-links = [\"${PYG_URL}\"]|g" pyproject.toml
    # Update Linux Source Marker
    sed -i '' "s/index = \"pytorch-[a-z0-9]*\", marker = \"sys_platform == 'linux'/index = \"${TARGET_INDEX}\", marker = \"sys_platform == 'linux'/g" pyproject.toml
else
    # Linux sed
    sed -i "s|find-links = \[\".*\"\]|find-links = [\"${PYG_URL}\"]|g" pyproject.toml
    # Update Linux Source Marker
    sed -i "s/index = \"pytorch-[a-z0-9]*\", marker = \"sys_platform == 'linux'/index = \"${TARGET_INDEX}\", marker = \"sys_platform == 'linux'/g" pyproject.toml
fi

echo "✅ Set PyG Links to : ${PYG_URL}"
echo "✅ Set Torch Index to: ${TARGET_INDEX} (Linux only; macOS/Windows use PyPI)"

# ------------------------------------------------------------------------------
# Sync
# ------------------------------------------------------------------------------
echo ""
echo "🧹 Cleaning old lockfile..."
rm -f uv.lock

echo "📦 Syncing Environment (Python 3.11)..."
# Force Python 3.11 creation
uv sync --python 3.11 --all-extras

# ------------------------------------------------------------------------------
# Finalize
# ------------------------------------------------------------------------------
echo ""
echo "🔧 Configuring Git Hooks..."
if [[ ! -x ".venv/bin/python" ]]; then
    echo "❌ Error: .venv/bin/python is missing. The environment is incomplete."
    echo "   Recreate it with: rm -rf .venv uv.lock && uv sync --python 3.11 --all-extras"
    exit 1
fi

# Prefer the console script because some broken installs expose the package but
# fail with `python -m pre_commit`.
if [[ -x ".venv/bin/pre-commit" ]]; then
    .venv/bin/pre-commit --version >/dev/null
    .venv/bin/pre-commit install --install-hooks
elif .venv/bin/python -m pre_commit --version >/dev/null 2>&1; then
    .venv/bin/python -m pre_commit install --install-hooks
else
    echo "❌ Error: pre-commit is missing or broken in .venv."
    echo "   Ensure sync succeeded with lint extras: uv sync --python 3.11 --all-extras"
    exit 1
fi

echo ""
echo "🧪 Verifying Python/Torch runtime..."
if ! .venv/bin/python -c "import sys; import torch; print(f'✅ Python Ver    : {sys.version.split()[0]}'); print(f'✅ Torch Version : {torch.__version__}'); print(f'✅ CUDA Available: {torch.cuda.is_available()}'); print(f'✅ CUDA Version  : {torch.version.cuda}')"; then
    echo ""
    echo "❌ Torch import verification failed."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   On macOS, torch should come from PyPI (not the custom PyTorch index)."
        echo "   Re-run: bash uv_env_setup.sh cpu"
    fi
    exit 1
fi
echo "======================================================="
echo "🎉 Setup Complete!"
echo "======================================================="
