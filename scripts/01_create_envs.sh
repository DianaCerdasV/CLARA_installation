#!/usr/bin/env bash
# Step 1: create the two conda environments used by the Qwen3 version of CLARA
# (https://github.com/aipapadim/multimodal-chatbot).
#
#   qwen3            -> Qwen3 API (port 5555) + Gradio chat UI (port 7861)
#   inpaint-anything -> SAM 2.1 + LaMa object removal/moving API (port 4444)
#
# The names must stay exactly like this: start_pillar_chatbot.sh looks for them.
#
# Usage:  bash 01_create_envs.sh ~/multimodal-chatbot
set -eo pipefail

CLARA_DIR="${1:-$HOME/multimodal-chatbot}"
CLARA_DIR="$(cd "$CLARA_DIR" && pwd)"

# PyTorch build. cu121 works with most NVIDIA drivers from 2023 onwards.
# RTX 50xx (Blackwell) cards need a newer build, e.g.:
#   TORCH_SPEC="torch==2.7.1 torchvision==0.22.1" TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash 01_create_envs.sh ...
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1 torchvision==0.20.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if [ ! -f "$CLARA_DIR/start_pillar_chatbot.sh" ] || [ ! -d "$CLARA_DIR/qwen3_api" ]; then
    echo "ERROR: $CLARA_DIR does not look like the multimodal-chatbot repository."
    echo "Clone it first:  git clone https://github.com/aipapadim/multimodal-chatbot.git"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

create_env() {
    if conda env list | awk '{print $1}' | grep -qx "$1"; then
        echo ">>> Environment '$1' already exists, reusing it."
    else
        echo ">>> Creating environment '$1'..."
        conda create -n "$1" python=3.12 pip -y
    fi
}

# ---------------------------------------------------------------------------
# 1) qwen3: Qwen3 API + Florence-2 grounding + Gradio UI
# ---------------------------------------------------------------------------
create_env qwen3
conda activate qwen3
python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
# Versions pinned to the ones in the author's core/env-core.yml.
# transformers 4.47 is needed for Florence-2 (newer releases break its remote code).
python -m pip install \
    "setuptools<81" \
    transformers==4.47.0 huggingface_hub==0.27.1 accelerate==1.2.1 \
    timm==1.0.14 einops \
    flask flask-cors waitress ollama Pillow requests \
    gradio==5.12.0 gradio-client==1.5.4 fastapi==0.115.6 pydantic==2.10.5
conda deactivate

# ---------------------------------------------------------------------------
# 2) inpaint-anything: SAM 2.1 + LaMa
# ---------------------------------------------------------------------------
create_env inpaint-anything
conda activate inpaint-anything
python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
# Same pins as the original CLARA requirements/inpaint.txt (tested by the author).
# setuptools<81 keeps pkg_resources, which pytorch-lightning 2.3 still imports.
python -m pip install \
    "setuptools<81" wheel \
    numpy==1.26.4 Flask==3.1.3 Pillow==11.1.0 pycocotools==2.0.8 \
    opencv-python==4.10.0.84 omegaconf==2.3.0 hydra-core==1.3.2 \
    pytorch-lightning==2.3.0 torchmetrics==1.4.0.post0 kornia==0.5.0 \
    albumentations==0.5.2 scikit-image==0.25.0 scikit-learn==1.6.1 \
    easydict==1.13 webdataset==0.2.100 pandas==2.2.3 matplotlib==3.10.0 \
    joblib==1.4.2 timm==1.0.14 tabulate PyYAML packaging iopath tqdm
# Install the bundled SAM 2.1 package. It must be a normal install (no -e):
# an editable install gets shadowed by the Inpaint-Anything/sam2 folder.
# SAM2_BUILD_CUDA=0 skips an optional CUDA extension that often fails to compile.
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation "$CLARA_DIR/Inpaint-Anything/sam2"
conda deactivate

echo
echo "Done. Both environments are ready: qwen3, inpaint-anything"
echo "Next: bash 02_download_models.sh $CLARA_DIR"
