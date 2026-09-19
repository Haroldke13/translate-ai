#!/usr/bin/env python3
"""Download open-licensed Bible text used to train the translator.

No API keys and no accounts are needed: eBible.org publishes plain verse-per-line
archives over HTTPS. Only openly licensed translations are listed here, and the
license notice shipped inside each archive is copied next to the download so the
attribution stays with the data.

    python scripts/fetch_bible_corpus.py                # kikuyu + english defaults
    python scripts/fetch_bible_corpus.py --list
    python scripts/fetch_bible_corpus.py kik engwebp --output data/bible/raw

Run this once while online. Everything after it works offline.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://ebible.org/Scriptures/{code}_vpl.zip"
USER_AGENT = "kikuyu-ai-corpus-fetcher/1.0"


@dataclass(frozen=True)
class Source:
    code: str
    language: str
    title: str
    license: str


# Only openly licensed texts are listed. Some translations on eBible are
# CC BY-NC-ND, which forbids derivative works; a model trained on a text is a
# derivative of it, so those are deliberately not offered here.
SOURCES: dict[str, Source] = {
    "kik": Source(
        "kik",
        "kikuyu",
        "Biblica Open Kikuyu Holy Word of God (2013)",
        "CC BY-SA 4.0 - attribution and share-alike required",
    ),
    "luo": Source(
        "luo",
        "luo",
        "Biblica Open New Luo Translation, Dholuo (2020)",
        "CC BY-SA 4.0 - attribution and share-alike required",
    ),
    "swhonen": Source(
        "swhonen",
        "swahili",
        "Biblica Open Kiswahili Contemporary Version",
        "CC BY-SA 4.0 - attribution and share-alike required",
    ),
    "gaz": Source(
        "gaz",
        "oromo",
        "Biblica Open Afaan Oromoo translation",
        "CC BY-SA 4.0 - attribution and share-alike required",
    ),
    "engwebp": Source(
        "engwebp",
        "english",
        "World English Bible (Protestant)",
        "Public domain",
    ),
    "engwebpb": Source(
        "engwebpb",
        "english",
        "World English Bible (British spelling)",
        "Public domain",
    ),
    "eng-web": Source(
        "eng-web",
        "english",
        "World English Bible",
        "Public domain",
    ),
}

# Kikuyu and English by default; add "luo" for Dholuo.
DEFAULT_CODES = ("kik", "engwebp")


def about_text(archive: Path) -> str:
    """Return the plain-text license notice bundled in an eBible VPL archive."""
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith("_about.htm")]
        if not names:
            return ""
        html = bundle.read(names[0]).decode("utf-8", "replace")
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"[ \t]+", " ", text).strip()


def download(code: str, output: Path, force: bool) -> Path:
    destination = output / f"{code}_vpl.zip"
    if destination.exists() and not force:
        print(f"{destination} (cached)")
        return destination
    url = BASE_URL.format(code=code)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
    except HTTPError as exc:
        raise SystemExit(f"{url}: HTTP {exc.code}. Use --list to see known codes.") from None
    except URLError as exc:
        raise SystemExit(f"{url}: {exc.reason}. This step needs a network connection once.") from None
    if not payload.startswith(b"PK"):
        raise SystemExit(f"{url}: response was not a zip archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"{destination} ({len(payload)} bytes)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codes", nargs="*", default=[], help=f"eBible codes (default: {' '.join(DEFAULT_CODES)})")
    parser.add_argument("--output", type=Path, default=Path("data/bible/raw"))
    parser.add_argument("--force", action="store_true", help="re-download even when the archive is cached")
    parser.add_argument("--list", action="store_true", help="print the known open-licensed codes and exit")
    args = parser.parse_args()

    if args.list:
        for source in SOURCES.values():
            print(f"{source.code:10} {source.language:8} {source.title} [{source.license}]")
        return

    codes = args.codes or list(DEFAULT_CODES)
    unknown = [code for code in codes if code not in SOURCES]
    if unknown:
        raise SystemExit(f"Unknown code(s): {', '.join(unknown)}. Run --list for the known set.")

    args.output.mkdir(parents=True, exist_ok=True)
    notices = []
    for code in codes:
        source = SOURCES[code]
        archive = download(code, args.output, args.force)
        notice = about_text(archive)
        if notice:
            (args.output / f"{code}_LICENSE.txt").write_text(notice, encoding="utf-8")
        notices.append(f"{source.code}: {source.title} - {source.license}")

    (args.output / "SOURCES.txt").write_text("\n".join(notices) + "\n", encoding="utf-8")
    print("\nAttribution (keep this with any data you redistribute):")
    for line in notices:
        print(f"  {line}")


if __name__ == "__main__":
    main()
