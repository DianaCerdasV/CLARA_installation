#!/usr/bin/env bash
# Step 2: download every model the Qwen3 version of CLARA needs.
#
#   - qwen3-vl:8b (Ollama)             the "brain" that replaces Llama 3.2
#   - microsoft/Florence-2-base (HF)   finds the boxes of objects in the image
#   - SAM 2.1 base_plus               cuts out the object
#   - big-lama                         fills the hole left by the object
#
# Usage:  bash 02_download_models.sh ~/multimodal-chatbot
set -eo pipefail

CLARA_DIR="${1:-$HOME/multimodal-chatbot}"
CLARA_DIR="$(cd "$CLARA_DIR" && pwd)"
PM="$CLARA_DIR/Inpaint-Anything/pretrained_models"

# ---------------------------------------------------------------------------
# 1) Qwen3-VL through Ollama
# ---------------------------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    echo ">>> Downloading qwen3-vl:8b (about 6 GB)..."
    ollama pull qwen3-vl:8b
else
    echo ">>> Ollama is not installed on this PC: skipping qwen3-vl:8b."
    echo "    Fine if you use Ollama on another server (see use_remote_ollama.sh)."
    echo "    Otherwise install it first: curl -fsSL https://ollama.com/install.sh | sh"
fi

# ---------------------------------------------------------------------------
# 2) Florence-2 (downloaded once into the Hugging Face cache, no login needed)
# ---------------------------------------------------------------------------
echo ">>> Downloading microsoft/Florence-2-base..."
# A PYTHONPATH pointing at /usr/lib/python3/dist-packages makes conda load
# the system's old Python packages and crash. Conda envs never need it.
unset PYTHONPATH

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx qwen3; then
    echo "ERROR: the conda environment 'qwen3' does not exist yet."
    echo "Run step 1 first:  bash 01_create_envs.sh $CLARA_DIR"
    echo "(Ollama models already downloaded are kept, so re-running this is quick.)"
    exit 1
fi
conda activate qwen3
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('microsoft/Florence-2-base'))"
conda deactivate

# ---------------------------------------------------------------------------
# 3) SAM 2.1 checkpoints
# ---------------------------------------------------------------------------
mkdir -p "$PM/big-lama/models"
for f in sam2.1_hiera_base_plus.pt sam2.1_hiera_large.pt; do
    if [ -s "$PM/$f" ]; then
        echo ">>> $f already downloaded."
    else
        echo ">>> Downloading $f..."
        wget -c -O "$PM/$f" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$f"
    fi
done

# ---------------------------------------------------------------------------
# 4) big-lama: needs config.yaml + best.ckpt (the code reads big-lama/best.ckpt,
#    the README also asks for big-lama/models/best.ckpt, so we keep both)
# ---------------------------------------------------------------------------
if [ -s "$PM/big-lama/config.yaml" ] && [ -s "$PM/big-lama/best.ckpt" ]; then
    echo ">>> big-lama already downloaded."
else
    echo ">>> Downloading big-lama..."
    TMP_ZIP="$(mktemp --suffix=.zip)"
    wget -O "$TMP_ZIP" "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
    unzip -o "$TMP_ZIP" -d "$PM"   # creates $PM/big-lama/config.yaml and models/best.ckpt
    rm -f "$TMP_ZIP"
    cp "$PM/big-lama/models/best.ckpt" "$PM/big-lama/best.ckpt"
fi

echo
echo "All models downloaded."
echo "Next: bash 03_check_install.sh $CLARA_DIR"
