#!/usr/bin/env python3
"""Download provider files explicitly supplied by the operator.

Usage: python scripts/download_bible_data.py URL... --output data/bible/raw
This keeps licensing and provider selection transparent; it does not assume
that every listed format is offered by every provider.
"""
import argparse
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("data/bible/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for url in args.urls:
        name = url.rstrip("/").split("/")[-1] or "download"
        destination = args.output / name.split("?")[0]
        with urlopen(url, timeout=60) as response, destination.open("wb") as handle:
            handle.write(response.read())
        print(destination)


if __name__ == "__main__":
    main()

