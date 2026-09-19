#!/usr/bin/env bash
# Download the base models this project uses. They are not stored in the repo
# (too large for GitHub's free LFS quota); they live on the Hugging Face Hub and
# are pulled here on demand. Re-running is safe — completed downloads are skipped.
set -euo pipefail

DEST="${MODELS_DIR:-models/downloads/huggingface}"
mkdir -p "$DEST"

MODELS=(
  "facebook/mms-1b-all"                       # speech (~3.6 GB)
  "nickdee96/nllb-200-600m-kikuyu-english"    # translation (~1.2 GB)
  "Kiragu/whisper-small-kikuyu-v5"            # ASR (~0.9 GB)
)

have_hf() { command -v huggingface-cli >/dev/null 2>&1; }

if ! have_hf; then
  echo "huggingface-cli not found. Install it with:"
  echo "    pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

for repo in "${MODELS[@]}"; do
  target="$DEST/${repo/\//--}"
  echo "→ $repo  →  $target"
  huggingface-cli download "$repo" --local-dir "$target" --local-dir-use-symlinks False
done

echo
echo "Base models fetched. To (re)build the phone package models, run:"
echo "    python MOBILE_VERSION/tools/pack_mobile.py"
