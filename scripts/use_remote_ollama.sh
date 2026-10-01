#!/usr/bin/env bash
# Optional: use Ollama on another machine (e.g. a lab server) instead of this PC.
#
# The authors' code talks to Ollama at http://127.0.0.1:11434 (this PC).
# This script checks that the remote server answers and has qwen3-vl:8b,
# then changes that one line in qwen3_api/ai_engine.py to the server address.
#
# Usage:  bash use_remote_ollama.sh ~/multimodal-chatbot 192.168.1.50
#         bash use_remote_ollama.sh ~/multimodal-chatbot 192.168.1.50:11434
#         bash use_remote_ollama.sh ~/multimodal-chatbot local      (go back to this PC)
set -eo pipefail

CLARA_DIR="${1:?Usage: bash use_remote_ollama.sh <multimodal-chatbot folder> <server IP or 'local'>}"
SERVER="${2:?Usage: bash use_remote_ollama.sh <multimodal-chatbot folder> <server IP or 'local'>}"
CLARA_DIR="$(cd "$CLARA_DIR" && pwd)"
ENGINE="$CLARA_DIR/qwen3_api/ai_engine.py"

if ! grep -q '^client = Client(host=' "$ENGINE" 2>/dev/null; then
    echo "ERROR: could not find the Ollama address line in $ENGINE"
    exit 1
fi

if [ "$SERVER" = "local" ]; then
    URL="http://127.0.0.1:11434"
else
    SERVER="${SERVER#http://}"
    SERVER="${SERVER%/}"
    case "$SERVER" in *:*) ;; *) SERVER="$SERVER:11434" ;; esac
    URL="http://$SERVER"

    echo ">>> Checking $URL ..."
    if ! TAGS="$(curl -s --max-time 10 "$URL/api/tags")"; then
        echo "ERROR: no answer from $URL"
        echo "On the server, Ollama must listen on the network, not only on itself:"
        echo "  sudo systemctl edit ollama   ->  add:  [Service]"
        echo "                                         Environment=\"OLLAMA_HOST=0.0.0.0:11434\""
        echo "  sudo systemctl restart ollama"
        echo "and the firewall must allow port 11434 from this PC."
        exit 1
    fi
    if ! echo "$TAGS" | grep -q '"qwen3-vl:8b"'; then
        echo "ERROR: the server answers, but it does not have qwen3-vl:8b."
        echo "On the server run:  ollama pull qwen3-vl:8b"
        echo "(The plain 'qwen3' model does not work: CLARA needs the vision model 'qwen3-vl'.)"
        exit 1
    fi
    echo ">>> Server OK, qwen3-vl:8b is there."
fi

sed -i "s#^client = Client(host=.*#client = Client(host=\"$URL\")#" "$ENGINE"
echo ">>> $ENGINE now uses: $URL"
