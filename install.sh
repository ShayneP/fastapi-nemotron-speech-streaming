#!/usr/bin/env bash
set -euo pipefail

echo "=== NeMo STT Server - Mac Installation ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'" 2>/dev/null || {
    echo "ERROR: Python 3.10+ is required"
    exit 1
}

# Create venv if not already in one
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Activated .venv"
else
    echo "Using existing virtualenv: $VIRTUAL_ENV"
fi

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU/MPS)
echo "Installing PyTorch..."
pip install torch torchaudio

# Install server dependencies
echo "Installing server dependencies..."
pip install -r requirements.txt

# Install NeMo toolkit with ASR support, handling triton issue on Mac
echo "Installing NeMo toolkit..."

# Try the simple path first
if pip install "nemo-toolkit[asr]" 2>/dev/null; then
    echo "NeMo installed successfully"
else
    echo "Standard install failed (likely triton issue on Mac), trying workaround..."

    # Clone and install from source with triton removed
    NEMO_TMP=$(mktemp -d)
    echo "Cloning NeMo to $NEMO_TMP..."
    git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$NEMO_TMP/NeMo"
    cd "$NEMO_TMP/NeMo"

    # Remove triton from all requirement files
    find . -name "*.txt" -path "*/requirements/*" -exec sed -i '' '/triton/d' {} + 2>/dev/null || true
    sed -i '' '/triton/d' requirements.txt 2>/dev/null || true

    # Also patch setup.cfg / pyproject.toml if triton appears
    find . -maxdepth 1 \( -name "setup.cfg" -o -name "pyproject.toml" \) -exec sed -i '' '/triton/d' {} + 2>/dev/null || true

    pip install ".[asr]"
    cd - > /dev/null

    echo "NeMo installed from source (triton excluded)"
    rm -rf "$NEMO_TMP"
fi

# Install Hugging Face hub for model download
pip install huggingface_hub

echo ""
echo "=== Installation complete ==="
echo ""
echo "Set this env var before running:"
echo "  export PYTORCH_ENABLE_MPS_FALLBACK=1"
echo ""
echo "Start the server:"
echo "  python server.py"
