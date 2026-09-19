#!/bin/sh
# Start the translator on an Android phone under Termux.
#
#   ./start.sh              start on http://localhost:8600
#   ./start.sh --port 9000  use a different port
#   ./start.sh --lan        also let other devices on the wifi connect
#
# Written in POSIX sh rather than bash so it runs on a bare Termux install,
# before anything optional has been added.

set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PORT=8600
EXTRA=""

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?--port needs a number}"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --lan) EXTRA="$EXTRA --lan"; shift ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) EXTRA="$EXTRA $1"; shift ;;
    esac
done

# --- find a usable Python -------------------------------------------------
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done

if [ -z "$PYTHON" ]; then
    cat <<'MSG'
Python is not installed.

On Termux, run this once:

    pkg update && pkg install python

then run ./start.sh again.
MSG
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "This needs Python 3.9 or newer. Found: $("$PYTHON" --version 2>&1)" >&2
    exit 1
fi

# --- check the offline data actually came across ---------------------------
if [ ! -f "$HERE/data/kikuyu/verses.jsonl.gz" ] && [ ! -f "$HERE/models/translation/config.json" ]; then
    cat <<'MSG'
Warning: no offline data and no translation model were found in this folder.

The app will start but will not be able to translate anything. On the computer,
run:

    .venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --data

and copy the whole MOBILE_VERSION folder across again.

MSG
fi

# --- keep Android from killing us mid-translation --------------------------
# Android suspends background processes aggressively. Without a wake lock, a
# long recording stops being transcribed the moment the screen goes off.
WAKELOCK=0
if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock && WAKELOCK=1
fi

cleanup() {
    [ "$WAKELOCK" -eq 1 ] && command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock || true
}
trap cleanup EXIT INT TERM

# --- open the browser once the server is actually listening ----------------
if command -v termux-open-url >/dev/null 2>&1; then
    (
        i=0
        while [ "$i" -lt 40 ]; do
            if "$PYTHON" - "$PORT" <<'PROBE' >/dev/null 2>&1
import socket, sys
s = socket.socket(); s.settimeout(0.3)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PROBE
            then
                termux-open-url "http://localhost:$PORT" >/dev/null 2>&1
                exit 0
            fi
            i=$((i + 1))
            sleep 0.25
        done
    ) &
fi

cd "$HERE"
# -u so the banner and progress appear immediately rather than sitting in a
# buffer, which on a phone looks exactly like the app having failed to start.
exec "$PYTHON" -u -m mobile.server --port "$PORT" $EXTRA
