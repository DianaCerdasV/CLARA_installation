#!/usr/bin/env bash
# Step 3: check that everything is in place before starting CLARA.
# It does not change anything; it only prints OK / MISSING for each item.
#
# Usage:  bash 03_check_install.sh ~/multimodal-chatbot
set -o pipefail

CLARA_DIR="${1:-$HOME/multimodal-chatbot}"
CLARA_DIR="$(cd "$CLARA_DIR" && pwd)"
PM="$CLARA_DIR/Inpaint-Anything/pretrained_models"
FAIL=0

ok()   { echo "  [OK]      $1"; }
bad()  { echo "  [MISSING] $1"; FAIL=1; }

echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    bad "nvidia-smi not found (an NVIDIA GPU + driver is required)"
fi

echo "Ollama"
if command -v ollama >/dev/null 2>&1; then
    ok "ollama $(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    if ollama list 2>/dev/null | grep -q "qwen3-vl:8b"; then ok "qwen3-vl:8b model"; else bad "qwen3-vl:8b model (run: ollama pull qwen3-vl:8b)"; fi
else
    bad "ollama (run: curl -fsSL https://ollama.com/install.sh | sh)"
fi

echo "Model files"
for f in sam2.1_hiera_base_plus.pt big-lama/config.yaml big-lama/best.ckpt big-lama/models/best.ckpt; do
    if [ -s "$PM/$f" ]; then ok "$f"; else bad "Inpaint-Anything/pretrained_models/$f"; fi
done

echo "Conda environments"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda activate qwen3 2>/dev/null; then
    if (cd "$CLARA_DIR/core/src" && python -c "import multimodal_chatbot_ui" >/dev/null 2>&1); then ok "qwen3: UI imports"; else bad "qwen3: UI imports (re-run 01_create_envs.sh)"; fi
    if python -c "import torch, transformers, ollama, flask_cors, timm, einops; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        ok "qwen3: packages + CUDA"
    else
        bad "qwen3: packages or CUDA (python -c 'import torch; print(torch.cuda.is_available())' should print True)"
    fi
    if HF_HUB_OFFLINE=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Florence-2-base')" >/dev/null 2>&1; then ok "Florence-2-base downloaded"; else bad "Florence-2-base (re-run 02_download_models.sh)"; fi
    conda deactivate
else
    bad "conda env 'qwen3'"
fi

if conda activate inpaint-anything 2>/dev/null; then
    if (cd "$CLARA_DIR/Inpaint-Anything" && python -c "
import sys, os; sys.path.append(os.getcwd())
import torch; assert torch.cuda.is_available()
from sam2.build_sam import build_sam2
from lama_inpaint import inpaint_img_with_lama" >/dev/null 2>&1); then
        ok "inpaint-anything: SAM2 + LaMa imports + CUDA"
    else
        bad "inpaint-anything: imports or CUDA (re-run 01_create_envs.sh)"
    fi
    conda deactivate
else
    bad "conda env 'inpaint-anything'"
fi

if command -v lsof >/dev/null 2>&1; then ok "lsof"; else bad "lsof (run: sudo apt install lsof) - used by the start script"; fi

echo
if [ "$FAIL" = 0 ]; then
    echo "Everything looks good. Start CLARA with:"
    echo "  cd $CLARA_DIR && bash start_pillar_chatbot.sh"
else
    echo "Some items are missing - fix the [MISSING] lines above and run this check again."
fi
exit $FAIL
