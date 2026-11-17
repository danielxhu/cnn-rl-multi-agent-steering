#!/usr/bin/env bash
# One-time setup of an Ubuntu 24.04 Azure VM for E7 training (policy/AZURE_E7.md).
#   bash cnn-rl-multi-agent-steering/policy/scripts/azure_setup.sh
# Installs Python packages into ~/venv (CPU-only torch), makes ~/policy-data and
# ~/policy-runs, and runs the E7 tests as a check. Safe to run twice.
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y python3-venv tmux

if [ ! -x "$HOME/venv/bin/python" ]; then
    python3 -m venv "$HOME/venv"
fi
"$HOME/venv/bin/pip" install --upgrade pip
"$HOME/venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
"$HOME/venv/bin/pip" install numpy opencv-python-headless pillow matplotlib

mkdir -p "$HOME/policy-data" "$HOME/policy-runs"

cd "$(dirname "$0")/.."                       # policy/
"$HOME/venv/bin/python" tests/run_tests.py e7_
echo
echo "setup done: $(nproc) vCPUs. Next: upload the val suite to ~/policy-data/val (AZURE_E7.md step 4)."
