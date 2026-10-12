#!/usr/bin/env bash
# ==============================================================================
# AMHIE (Adaptive Multi-Granularity Hierarchical Indexing Engine)
# macOS 1-Click Setup & Launch Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Text styles
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "=================================================================="
echo "    🌳 AMHIE: Adaptive Multi-Granularity Hierarchical Indexing    "
echo "                   macOS Launch Suite                             "
echo "=================================================================="
echo -e "${NC}"

# ------------------------------------------------------------------------------
# 1. Environment Verification
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Checking macOS environment...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found. Install via: brew install python${NC}"
    exit 1
fi

echo -e "  ✓ Python: $(python3 --version)"

# ------------------------------------------------------------------------------
# 2. Virtual Environment Setup
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/5] Setting up Python virtual environment...${NC}"

VENV_PATH="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_PATH" ]; then
    if [ -d "$SCRIPT_DIR/../venv" ]; then
        VENV_PATH="$SCRIPT_DIR/../venv"
    else
        echo -e "  Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    fi
fi

source "$VENV_PATH/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_CLIENT_TOOLBAR_MODE=minimal
echo -e "  ✓ Active Python: $(which python3)"

# Check key packages
if ! python3 -c "import streamlit, fastapi, pymupdf" &> /dev/null; then
    echo -e "${YELLOW}  Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo -e "  ✓ Dependencies installed."
fi

# ------------------------------------------------------------------------------
# 3. Compile C++17 Tree Engine
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/5] Compiling C++17 Tree Engine...${NC}"

mkdir -p csrc
if command -v clang++ &> /dev/null; then
    clang++ -std=c++17 -O3 -fPIC -shared csrc/tree_engine.cpp -o csrc/libtree_engine.dylib 2>/dev/null || true
    cp csrc/libtree_engine.dylib csrc/libtree_engine.so 2>/dev/null || true
fi

if [ -f "csrc/libtree_engine.dylib" ]; then
    echo -e "  ${GREEN}✓ C++ Tree Engine compiled (csrc/libtree_engine.dylib)${NC}"
else
    echo -e "  ⚠ C++ library not compiled. Using Python fallback tree engine."
fi

# ------------------------------------------------------------------------------
# 4. Run Tests
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/5] Running verification tests...${NC}"
python3 tests/test_amhie.py
python3 tests/test_modular_api.py
echo -e "  ${GREEN}✓ All tests passed!${NC}"

# ------------------------------------------------------------------------------
# 5. Launch Services
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/5] Launching AMHIE Services...${NC}"

cleanup() {
    echo -e "\n${YELLOW}Shutting down AMHIE...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    echo -e "${GREEN}Goodbye!${NC}"
}
trap cleanup EXIT INT TERM

# Free ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true
sleep 1

# Start FastAPI backend (background, log to file)
echo -e "  Starting FastAPI backend on ${BOLD}http://localhost:8000${NC}..."
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --log-level warning > fastapi.log 2>&1 &
FASTAPI_PID=$!
sleep 2

echo -e "  Starting Streamlit Dashboard on ${BOLD}http://localhost:8501${NC}..."
echo -e "\n${GREEN}${BOLD}=================================================================="
echo "    🚀 AMHIE IS LIVE!                                              "
echo "    • Streamlit Dashboard   : http://localhost:8501                "
echo "    • FastAPI Backend       : http://localhost:8000                "
echo "    • API Docs (Swagger)    : http://localhost:8000/docs           "
echo "==================================================================${NC}"
echo -e "${CYAN}Press Ctrl+C to stop all services.${NC}\n"

# Launch Streamlit in foreground
python3 -m streamlit run ui/app.py \
    --server.port 8501 \
    --server.fileWatcherType none \
    --server.headless false \
    --browser.gatherUsageStats false
