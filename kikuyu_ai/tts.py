from pathlib import Path
import shutil
import subprocess


def synthesize(text: str, target: Path, command: str | None = None) -> bool:
    """Use a configured Piper-compatible command, or return false when unavailable."""
    if not text.strip() or not command:
        return False
    if not shutil.which(command.split()[0]):
        return False
    process = subprocess.run(command.split() + ["--output_file", str(target)], input=text, text=True, capture_output=True)
    return process.returncode == 0 and target.exists()

