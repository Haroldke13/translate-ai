# Problems solved

A running log of defects found and fixed in this project, newest first. Each
entry records what broke, why it broke, and what the fix was, so the same thing
is not rediscovered later.

---

## 2026-09-04 — Phone build (MOBILE_VERSION)

### `audioop` and `cgi` do not exist on the target phone
**Symptom:** none on this machine — `import audioop` succeeded here.
**Cause:** the import resolved to a third-party shim in `.venv/lib/python3.13/site-packages/audioop/`, not the standard library. Both modules were removed in Python 3.13/3.14 (PEP 594), and Termux ships **Python 3.14**. The offline tier promises to run on a bare `pkg install python`, where neither module and no shim exists.
**Fix:** reimplemented WAV decoding (8/16/24/32-bit, mono mixdown, resampling) in `mobile/audio.py` and multipart parsing in `mobile/multipart.py`, both stdlib-only. Added `test_no_removed_stdlib_modules`, which fails if either import returns. Verified by importing all nine modules under `/usr/bin/python3 -S` (3.14.4) with site-packages stripped.

### The NLLB tokenizer stamps `kik_Latn` on every language
**Symptom:** Somali Genesis 1:1 came back as *"It has been discovered that God has given birth to that plant."* instead of *"In the beginning, God created the heavens and the earth."*
**Cause:** the checkpoint is Kikuyu-fine-tuned, so `tokenizer.json`'s post-processor hard-codes `kik_Latn` as the source tag. `Tokenizer.encode()` applies it regardless of the requested language. This is the same defect that makes the desktop PWA mistranslate non-Kikuyu input.
**Fix:** `NeuralTranslator._encode` overwrites token 0 with the correct tag when it matches the `xxx_Xxxx` shape, prepending it otherwise. All four languages verified correct afterwards.

### A mis-heard recording produced confident, fluent nonsense
**Symptom:** the phone speech model transcribed a Kikuyu voice note as `oholowaku`; the translator turned that into **"Alright"**, presented exactly like a correct result.
**Cause:** the neural tier had no quality gate. The verse tier requires 0.82 overlap and the gloss tier requires 50% known words, but the neural tier translated anything handed to it.
**Fix:** `Engine.heard_well` scores the transcript against the language's own vocabulary; below 34% the result carries a visible warning. Returns `None` for Kamba, which has no word list, so no false warning is raised.

### `multipart` reported uploads two bytes too large
**Symptom:** a 10,240-byte upload was recorded as 10,242 bytes; file contents were correct.
**Cause:** `file.truncate(n)` does not move the file position, so the later `tell()` returned the pre-truncation offset.
**Fix:** compute the size before truncating and use that value.

### The verse index cache was invalidated by copying to the phone
**Symptom:** would have shown as a 15–20 second stall on the phone's first translation in each language.
**Cause:** the cache signature included the corpus file's mtime, and USB/MTP transfers do not preserve mtime.
**Fix:** signature is file size alone. Caches now survive the copy; measured 0.1 s per language to load 31,000 verses.

### `GlossTable.best()` answered from an empty table
**Symptom:** `best("muno")` returned `None` even though the packed table contained `mũno`.
**Cause:** `load()` was called by `gloss()` but not by `best()` or `alternatives()`, so a direct call read the not-yet-populated dict.
**Fix:** `_lookup()` calls `load()`.

### Half-typed verses lost their "closest verse" hint
**Symptom:** typing about half of Genesis 1:1 offered no suggestion.
**Cause:** the near-verse threshold was raised to 0.55; a three-of-six word overlap scores exactly 0.50.
**Fix:** threshold set to 0.50.
