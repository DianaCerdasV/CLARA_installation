#!/usr/bin/env bash
# Step 4 (optional): build the second version, "CLARA with clarification questions".
#
# It copies ~/multimodal-chatbot to ~/multimodal-chatbot-clara and replaces two
# files with the versions in clara_version/:
#   qwen3_api/ai_engine.py          + "clarify" mode (CLARA system prompt + full history)
#                                   + "locate" mode (Florence-2 boxes for object names)
#   core/src/multimodal_chatbot_ui.py   the dialogue UI; when Qwen answers with
#                                   'image-gen, result = {...}' it moves each object
#                                   into its container, one after another
#
# The original folder is not changed. Both versions use the same conda envs and
# the same model files (the copy links to the original pretrained_models folder).
#
# Usage:  bash 04_make_clara_version.sh ~/multimodal-chatbot [~/multimodal-chatbot-clara]
set -eo pipefail

SRC="${1:-$HOME/multimodal-chatbot}"
SRC="$(cd "$SRC" && pwd)"
DST="${2:-${SRC}-clara}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="$HERE/clara_version"
# Upstream commit the replacement files were written against.
BASE_COMMIT=086dae6366a54770fdc934d2d67ac5997d81754d
FILES="qwen3_api/ai_engine.py core/src/multimodal_chatbot_ui.py"

if [ ! -f "$SRC/start_pillar_chatbot.sh" ] || [ ! -d "$SRC/qwen3_api" ]; then
    echo "ERROR: $SRC does not look like the multimodal-chatbot repository."
    exit 1
fi
if [ ! -d "$SRC/Inpaint-Anything/pretrained_models" ]; then
    echo "ERROR: run 02_download_models.sh first (no pretrained_models folder in $SRC)."
    exit 1
fi
if [ -e "$DST" ]; then
    echo "ERROR: $DST already exists. Delete it first if you want to rebuild it:"
    echo "  rm -rf \"$DST\""
    exit 1
fi

# Warn if the authors changed these files after the version this was written for.
if git -C "$SRC" cat-file -e "$BASE_COMMIT" 2>/dev/null; then
    if ! git -C "$SRC" diff --quiet "$BASE_COMMIT" -- $FILES; then
        echo "WARNING: the authors changed $FILES since commit ${BASE_COMMIT:0:7}."
        echo "         The CLARA version replaces them, so those changes will not be in the copy."
    fi
fi

echo ">>> Copying $SRC -> $DST (without the large model files)..."
mkdir -p "$DST"
tar -C "$SRC" --exclude=./Inpaint-Anything/pretrained_models --exclude=./Inpaint-Anything/outputs \
    -cf - . | tar -C "$DST" -xf -
ln -s "$SRC/Inpaint-Anything/pretrained_models" "$DST/Inpaint-Anything/pretrained_models"

echo ">>> Adding the CLARA clarification files..."
for f in $FILES; do
    cp "$OVERLAY/$f" "$DST/$f"
done
# Keep the Ollama address of the original folder (e.g. a remote server set
# with use_remote_ollama.sh).
HOST_LINE="$(grep -m1 '^client = Client(host=' "$SRC/qwen3_api/ai_engine.py" || true)"
if [ -n "$HOST_LINE" ]; then
    sed -i "s#^client = Client(host=.*#$HOST_LINE#" "$DST/qwen3_api/ai_engine.py"
fi

echo
echo "Done. You now have two versions:"
echo "  $SRC        (no clarification questions)"
echo "  $DST  (CLARA clarification dialogue)"
echo
echo "Start the CLARA version with:"
echo "  cd $DST && bash start_pillar_chatbot.sh"
echo "Only one version can run at a time (they use the same ports)."
