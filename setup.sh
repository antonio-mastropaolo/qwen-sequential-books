#!/usr/bin/env bash
# usage: bash setup.sh [--wandb]
set -euo pipefail
cd "$(dirname "$0")"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: nvidia-smi not found"
if ! command -v python3 >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
  sudo_cmd=""; [ "$(id -u)" -ne 0 ] && sudo_cmd="sudo"
  $sudo_cmd apt-get update && $sudo_cmd apt-get install -y python3 python3-venv python3-pip git
fi
VENV="${VENV:-$HOME/venvs/seqbooks}"
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install --upgrade pip
# CUDA 12.1 wheels of torch 2.5.1 (work with any driver >= 530)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if python -c 'import sys;sys.exit(sys.version_info<(3,11))'; then
  grep -v '^torch==' requirements.txt | pip install -r /dev/stdin
else  # py3.10: pins below need >=3.11
  pip install transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 "numpy<2.3" "pandas<3" "matplotlib<3.11"
fi
[ "${1:-}" = "--wandb" ] && pip install wandb==0.18.7 && wandb login
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# pre-download weights (~15 GB)
python -c "from huggingface_hub import snapshot_download as s; s('${MODEL:-Qwen/Qwen2.5-7B}')"
echo "Setup done. Activate with: . $VENV/bin/activate"
