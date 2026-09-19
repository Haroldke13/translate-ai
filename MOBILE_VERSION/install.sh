#!/bin/sh
# One-time setup for the optional neural tier on Android/Termux.
#
#   ./install.sh            translation only  (~150 MB of packages)
#   ./install.sh --speech   translation + speech recognition
#
# Nothing here is needed to start the app. Without it you still get exact Bible
# verses and word-by-word English; this adds real sentence translation.
#
# Deliberately avoids pip. On Termux, "pip install transformers" has to compile
# safetensors from Rust source, because there is no prebuilt Android binary for
# it — that pulls in a 576 MB toolchain. Every package below is prebuilt.

set -eu

WANT_SPEECH=0
for arg in "$@"; do
    case "$arg" in
        --speech) WANT_SPEECH=1 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done

if ! command -v pkg >/dev/null 2>&1; then
    cat <<'MSG'
This installer is for Termux on Android.

On a computer, use pip instead:

    pip install ctranslate2 tokenizers          # translation
    pip install onnxruntime numpy               # speech

MSG
    exit 1
fi

echo "Updating package lists…"
pkg update -y

echo "Installing Python and ffmpeg…"
pkg install -y python ffmpeg

echo "Installing the translation runtime (CTranslate2)…"
pkg install -y python-ctranslate2

# tokenizers is not in the main repo. TUR ships a genuine prebuilt aarch64
# build, which is what avoids the Rust toolchain.
echo "Adding the Termux User Repository for the tokenizer…"
pkg install -y tur-repo
pkg install -y python-tokenizers

if [ "$WANT_SPEECH" -eq 1 ]; then
    echo "Installing the speech runtime (ONNX Runtime)…"
    pkg install -y python-onnxruntime python-numpy
fi

echo
echo "Checking what is now available…"
python - <<'PROBE'
from importlib.util import find_spec
for module, purpose in [
    ("ctranslate2", "sentence translation"),
    ("tokenizers", "sentence translation"),
    ("onnxruntime", "speech recognition"),
    ("numpy", "speech recognition"),
]:
    mark = "yes" if find_spec(module) else "NO "
    print(f"  {mark}  {module:14} ({purpose})")
PROBE

cat <<'MSG'

Done. Start the translator with:

    ./start.sh

If Android keeps killing it mid-translation, that is the phantom process
killer. It has to be turned off once from a computer with adb:

    adb shell "settings put global settings_enable_monitor_phantom_procs false"

MSG
