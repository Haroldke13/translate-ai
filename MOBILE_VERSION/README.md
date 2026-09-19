# MOBILE_VERSION — translate on the phone, with the computer switched off

Kikuyu, Kamba, Oromo and Somali into English, running entirely on an Android
phone. No internet, no API keys, no computer.

Copy this folder to the phone, run `./start.sh`, open `http://localhost:8600`.

---

## What actually runs on the phone

Three tiers answer in order, and the app always tells you which one answered.
Nothing is dressed up as better than it is.

| Tier | What it does | Needs installing | Size | Peak memory |
|---|---|---|---|---|
| **Verse** | Bible text → the published English, word for word | nothing | 10 MB | ~90 MB |
| **Word list** | anything else → word-by-word English | nothing | (included above) | ~90 MB |
| **Translation** | any sentence → real English | `./install.sh` | 627 MB | ~1.2 GB |
| **Speech** | a recording → English | `./install.sh --speech` | 349 MB | ~630 MB |

**The first two tiers need nothing but Python.** On a bare `pkg install python`
the app already translates scripture exactly and glosses everything else. That
is the floor, and it works on any phone.

Measured on the packing computer — a phone is roughly 3–5× slower:

- verse lookup: **0.01–0.02 s**
- sentence translation: **0.6–1.6 s**
- speech: **0.43 s** for 3.7 s of audio

## What each language gets

| | Verse lookup | Word list | Spelling repair | Translation | Speech |
|---|---|---|---|---|---|
| **Kikuyu** | 31,094 verses | 42,321 words | 41,164 spellings | yes | yes |
| **Kamba** | — | — | — | yes | yes |
| **Oromo** | 31,094 verses | 29,951 words | not needed | yes | yes |
| **Somali** | 31,095 verses | 33,687 words | not needed | yes | yes |

**Kamba has no offline data at all.** There is no openly licensed Kamba Bible,
so there is nothing to build a verse index or word list from. Kamba works only
with the translation model installed; without it the app says so rather than
guessing. Oromo and Somali need no spelling repair because they are written
without the accented vowels Kikuyu uses.

## Installing

**1. On the computer**, build the package:

```
.venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --data          # 10 MB, always do this
.venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --translation   # 627 MB, optional
.venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --speech model.int8.onnx   # 349 MB, optional
```

**2. Install Termux on the phone** — from [F-Droid](https://f-droid.org/packages/com.termux/)
or GitHub releases, **not from Google Play**. The Play build is an abandoned
branch with known bugs. Never mix APKs from different sources; they are signed
with different keys and Android will refuse the update.

**3. Copy the folder across** by USB, and in Termux:

```
pkg install python
termux-setup-storage
cp -r ~/storage/shared/MOBILE_VERSION ~/
cd ~/MOBILE_VERSION
./start.sh
```

It opens in the browser automatically. Add it to the home screen from the
browser menu and it behaves like an app.

**4. For real sentence translation** (optional, ~150 MB of packages):

```
./install.sh            # translation
./install.sh --speech   # translation and speech
```

## Why the folder is laid out this way

Keep the code in Termux's own home directory (`~/MOBILE_VERSION`). Android's
shared storage has no execute permission and no Unix file modes, so software
run from `/sdcard` misbehaves in ways that are hard to diagnose. The model files
alone can live on shared storage if space is tight.

## The one thing that needs a computer, once

Android 12 and newer kill background processes that use a lot of CPU — exactly
what a translation looks like. You get `[Process completed (signal 9)]`
part-way through. Turning that off needs `adb` from a computer, once:

```
adb shell "settings put global settings_enable_monitor_phantom_procs false"
```

Also exclude Termux from battery optimisation in Android settings. `start.sh`
takes a wake lock automatically, which handles the screen going off but not the
phantom process killer.

## Honest limitations

- **Speech is noticeably worse than on the computer.** The phone uses a 349 MB
  multilingual model; the computer uses MMS at 3.9 GB, which will not load on a
  phone. On the test recording MMS heard `ũhorũ waku` and the phone model heard
  `oholowaku` — recognisable, but run together and with `l` for `r`. When a
  transcript does not look like real words, the app says so above the
  translation instead of presenting confident nonsense.
- **The word-list tier is a gloss, not a translation.** Held out from training,
  it scores BLEU-4 of about 2.5 — it gets the words right and the grammar
  wrong. It is there so the phone always answers something, and it is labelled
  *Word-by-word* every time it does.
- **The word list only knows Bible vocabulary.** Everyday words that never
  appear in scripture are not in it, and show up in brackets like `[word]`.
- **Kamba has no offline tier at all**, as above.
- **Verse matching needs the words to be close.** Below 82% overlap the app
  will not claim it is that verse; it offers it as *Closest verse* instead.

## What this does not do

No microphone recording over a network address — that needs `localhost` or
HTTPS, and on the phone you are on `localhost`, so it works. Translating *into*
Kikuyu is not supported; the model is trained one way, into English.

## Folder map

```
start.sh              start the app (Termux, or any Linux)
install.sh            add the optional translation and speech runtimes
mobile/
  server.py           the web app, standard library only
  engine.py           picks the tier and labels which one answered
  verses.py           inverted-index verse lookup
  gloss.py            word-by-word fallback
  spelling.py         fills in ĩ and ũ
  neural.py           CTranslate2 translation, ONNX speech
  audio.py            decodes recordings without audioop
  multipart.py        parses uploads without cgi
  languages.py        the four languages
web/                  the interface
data/<language>/      verses, word list, spellings
models/               optional translation and speech models
tools/pack_mobile.py  builds all of the above, on the computer
tests/                48 tests
```

## Notes for whoever maintains this

Termux ships **Python 3.14**, where `audioop` and `cgi` were removed. Both were
reimplemented here rather than depended on; there is a test that fails if either
import comes back.

The translation model is **CTranslate2**, not PyTorch, because `safetensors` has
no prebuilt Android binary — `pip install transformers` on Termux compiles Rust
and pulls in a 576 MB toolchain. CTranslate2 is a 5.5 MB package that runs the
same weights in int8.

Speech is **ONNX**, not MMS, because MMS bakes its per-language adapter into the
weights: four languages would mean four separate exports of about a gigabyte
each, and CTranslate2's wav2vec2 converter silently drops adapters entirely.

The NLLB tokenizer has `kik_Latn` **hard-coded** in its post-processor, because
the checkpoint was fine-tuned on Kikuyu. `neural.py` overwrites that tag on
every request. Without it, Somali is translated as though it were Kikuyu and
Genesis 1:1 comes back as *"It has been discovered that God has given birth to
that plant."*
