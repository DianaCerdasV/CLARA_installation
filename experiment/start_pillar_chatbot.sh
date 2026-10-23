#!/bin/bash
echo "Killing old modules..."

# Kill relevant running processes
pkill -9 -f "qwen3_api/app.py"
pkill -9 -f "Inpaint-Anything/app.py"
pkill -9 -f "multimodal_chatbot_ui.py"
pkill -9 -f "flask"

echo "Waiting for ports to be released..."
sleep 2

for port in 4444 5555 7861; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        kill -9 $(lsof -t -i:$port) 2>/dev/null
    fi
done

echo "Ports cleared. Starting Services..."

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Η SOTA συνάρτηση που τα σηκώνει όλα
start_module() {
    local path="$1"
    local script="$2"
    local env="$3"
    local name="$4"

    echo "Starting $name..."

    (
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env"
        cd "$path"
        
        python "$script" &
    )
}

# ---------------------------------------------------------
# SOTA MODULE STARTUP
# ---------------------------------------------------------

# 1. Qwen3 API (Brain) -> Port 5555 (χρησιμοποιεί το qwen3 conda env)
start_module "$PROJECT_ROOT/qwen3_api" "app.py" "qwen3" "Qwen API (Brain)"
sleep 5 

# 2. Inpaint-Anything API (SAM 2.1 + LaMa) -> Port 4444
start_module "$PROJECT_ROOT/Inpaint-Anything" "app.py" "inpaint-anything" "Inpaint API (Scalpel)"
sleep 5 

# 3. Gradio UI (Orchestrator) -> Port 7861 (χρησιμοποιεί το qwen3 conda env)
start_module "$PROJECT_ROOT/core/src" "TEST.py" "qwen3" "Gradio UI"
sleep 5 

echo "All SOTA modules are up and running!"
wait

cleanup() {
    echo "Stopping services..."
    kill "${PIDS[@]}" 2>/dev/null
    sleep 3
    kill -9 "${PIDS[@]}" 2>/dev/null
    echo "Done."
}
trap cleanup EXIT
