import argparse
from pathlib import Path
from kikuyu_ai.config import Settings
from kikuyu_ai.tts import synthesize

parser = argparse.ArgumentParser()
parser.add_argument("text", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if not synthesize(args.text.read_text(encoding="utf-8"), args.output, Settings.from_env().tts_command):
    raise SystemExit("TTS was not generated; configure KIKUYU_TTS_COMMAND")

