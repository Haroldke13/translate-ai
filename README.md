# Kikuyu AI Translator

> **Model files are not stored in this repository.** The base models are large
> (~6.9 GB) and public, so they are pulled from the Hugging Face Hub on demand
> rather than committed — GitHub's free Git LFS quota is 1 GB. Fetch them with:
>
> ```bash
> pip install -U 'huggingface_hub[cli]'
> ./scripts/fetch_models.sh
> ```
>
> Git LFS is already configured (`.gitattributes`) so any weight file you do commit
> is routed through it. See `UPLOAD_LARGE_FILES_TO_GITHUB.md` for the large-file workflow.

An offline-first Kikuyu to English speech translation service, with an installable
mobile app. Everything runs on your own hardware: no API keys, no accounts, and no
network calls at inference time.

## Use it on a phone

The server ships a mobile web app (a PWA) at `/`. Run the server on a computer on
your home network and open it from the phone's browser; the models stay on the
computer, which is what makes this usable on a phone at all.

```sh
pip install -e '.[web,ml]'
python -m kikuyu_ai.web.api --lan --https
```

There is also a plain Flask upload page if you just want to send a file and read
the English back, with no recording and no install step:

```sh
pip install -e '.[flask,ml]'
python -m kikuyu_ai.web.flask_app --lan
```

Pick a free port with `--port` if something else is already using 5000. Uploads
run as a background job and the result page polls until it is finished, because
transcribing a long recording takes far longer than a browser will wait on a
form post. One recording is decoded at a time, so a second upload queues rather
than loading the models twice. Setting `KIKUYU_API_URL` makes it forward uploads
to a FastAPI server instead of loading the models itself.

The same page also translates written text: type or paste Kikuyu and read the
English back. A verse that already exists in `data/bible/parallel.jsonl` is
returned as published rather than machine translated.

### Typing without the tilde vowels

Kikuyu is written with ĩ and ũ, but phone keyboards rarely offer them, so people
type `ruciu` for `rũciũ`. The translation model treats those as different words
and usually echoes the unaccented one straight back, untranslated.

Typed text therefore has its accents restored before the model sees it, using a
lookup table built from the aligned Bible corpus — 31k verses of correctly
accented Kikuyu. Only about 1.6% of that vocabulary has more than one accented
spelling, so the most frequent form is a reliable guess.

| You type | Read as | Result |
| --- | --- | --- |
| `RUCIU` | rũciũ | tomorrow |
| `ruciini` | rũciinĩ | in the morning |
| `Mundu ucio ni mwega` | Mũndũ ũcio nĩ mwega | That person is good |

A word you accented yourself is left alone, and so is a word the corpus has never
seen, so English and names pass through untouched. Text typed entirely in capitals
is lower-cased first, for the same reason: an all-caps word looks like an unknown
token to the model. The page shows what your text was read as whenever the
spelling was changed, so nothing happens silently.

The table is cached at `data/bible/orthography.json` and rebuilt automatically
when the corpus changes.

## Translate a recording

🎤 Choose a recording and select the matching language (Kikuyu, Luo, Swahili, Somali, Oromo, Kamba)

## Languages

Kikuyu is the default. Pick another from the buttons at the top of the page, or
with `?lang=luo`. An unrecognised code falls back to Kikuyu rather than failing.

| Language | NLLB tag | MMS adapter | Bible verses |
| --- | --- | --- | --- |
| Kikuyu | kik_Latn | kik | 31,094 |
| Luo (Dholuo) | luo_Latn | luo | 31,095 |
| Swahili | swh_Latn | swh | 31,095 |
| Somali | som_Latn | som | 31,095 |
| Oromo | gaz_Latn | orm | 31,094 |
| Kamba | kam_Latn | kam | none published |

All six do speech, text, and verse lookup. Kamba has no openly licensed Bible
text on eBible.org, so it has no verse lookup and no spelling repair; speech and
translation still work.

A language needs three things, and all three must line up:

* an **NLLB tag**, so the translation model knows what it is reading,
* an **MMS adapter**, for speech, and
* optionally **aligned Bible verses**, for exact verse lookup and for the
  spelling repair that fills in missing accented letters.

Adding a language is an adapter download, not a training run. MMS covers 1,198
languages with one shared 3.9 GB encoder plus a ~9 MB adapter each:

```sh
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/mms-1b-all', 'adapter.kam.safetensors',
    local_dir='models/downloads/huggingface/facebook--mms-1b-all')"
```

The upload form appears only when both the encoder and that language's adapter
are on disk, so a missing download disables the form with an explanation instead
of failing mid-job.

### Other Kenyan languages

MMS also has adapters for Giryama, Pokomo, Saamia, Samburu, Rendille, Teso,
Masaaba, Sukuma, Gwere, Kunda and Markweeta — but NLLB-200 has no tag for any of
them, so they could be transcribed and never translated into English. They are
left out rather than shipped half-working.

Not in MMS at all: Meru, Embu, Gusii (Kisii), Kalenjin and its varieties
(Kipsigis, Nandi, Tugen), Maasai, Turkana, Pokot, Luhya varieties, Digo, and
Borana. These would need a speech model trained from scratch.

### Speech accuracy

Measured on the Luo test split of Google FLEURS (real speech, human transcripts,
CC-BY): **36% mean word error rate** — roughly one word in three wrong. Usable
for getting the gist and for producing text worth correcting, not for verbatim
transcription. Worst on numbers and technical terms.

For Kikuyu there is no public labelled test set, so the two available models were
compared on 2 minutes of real Kikuyu audio by how many of the words they produce
are real Kikuyu, checked against the 53,620 word forms in the Bible corpus:

| | Kiragu whisper-small-kikuyu | MMS-1B (kik) |
| --- | --- | --- |
| Words that are real Kikuyu | 38.8% | **51.2%** |
| Words carrying tilde vowels | 11% | **46%** |
| Peak memory | **1.5 GB** | 4.5 GB |
| Time for 4 clips | **63 s** | 159 s |

MMS is the default for Kikuyu: it produces more real words and, importantly,
restores the ĩ and ũ that the translation step depends on. It costs three times
the memory and runs about 2.5x slower. To go back:

```sh
KIKUYU_ASR_ENGINE=whisper python -m kikuyu_ai.web.flask_app --lan
```

This is a proxy measure, not a word error rate: it counts whether words exist in
Kikuyu, not whether they are the words that were said. Treat it as evidence that
MMS is better here, not as an accuracy figure.

### Running on a phone under Termux

Short answer for MMS: **no.** It needs about **4.5 GB of RAM** at peak, plus 3.9 GB
of disk. Most Android phones will be killed by the OS before it finishes loading,
and PyTorch has no official Android wheels, so `pip install torch` under Termux
generally fails or has to be built from source.

The supported way to use a phone is the one already built in: run the models on a
computer and point the phone at it.

```sh
# on the computer
python -m kikuyu_ai.web.flask_app --lan
# or, for the installable app with microphone recording
python -m kikuyu_ai.web.api --lan --https
```

The phone then needs only a browser. If you must run something on the device
itself, the older Whisper path is the lighter one at ~1.5 GB, and would need
PyTorch working under Termux first; a CTranslate2 or ONNX build of a small model
is a more realistic route than MMS.

## Long recordings: clip, convert, resume

Transcribing a three-hour recording in one pass takes hours, and an interruption
loses all of it. `kikuyu_ai.clipper` cuts the recording into clips, records what
finished in `clips.json` next to the output, and continues from the last finished
clip on the next run.

```sh
# split into 60-second clips and convert them to 16 kHz mono WAV
python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --seconds 60

# only the sections you care about
python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon \
    --ranges 2:30-5:00,17:40-19:05

# convert and translate, stopping and resuming as often as you like
python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --translate
python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --translate --resume

# where did it get to, and start again from nothing
python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --status
python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --restart
```

Times are written as seconds, `MM:SS`, or `HH:MM:SS`. Progress is saved after
every clip, so killing the process loses at most the clip in flight; re-running
the same command carries on. A clip that fails is recorded with its reason and
retried next time instead of stopping the run, and `--overlap` keeps a few
seconds of context across cut points so a sentence split across a boundary is not
lost. The output folder is tied to the recording that created it, so pointing a
different file at it is refused rather than silently mixing two recordings.

With `--translate`, each clip's Kikuyu and English are stored alongside it and
joined into `transcript.txt` in time order.

`--lan` binds every interface and prints the address to type on the phone.
`--https` generates a local self-signed certificate with `openssl` and never
contacts a certificate authority. Accept the certificate warning once on the phone.

HTTPS matters for one specific reason: phone browsers only expose the microphone
on `localhost` or HTTPS. Over plain HTTP the app hides the record button and
falls back to a file picker that opens the phone's own voice recorder, which
still works but takes more taps.

In the browser menu choose **Add to home screen** to install it. After that it
opens full screen with its own icon. The app shell is cached by a service worker,
so it opens with no network at all, and corrections you write while disconnected
are queued on the phone and sync when the server is reachable again. Translating
audio still needs the server, because that is where the models live.

The app gives you press-and-hold recording, file upload, a typing tab, the English
result with the matched Bible verse when there is one, subtitle downloads, and a
correction box that feeds straight back into training.

Running the models directly on the phone under Termux is also supported; see
**Online FastAPI mode** below for the trade-off.

## Quick start

```sh
cd kikuyu_ai
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest
python -m kikuyu_ai.cli recording.mp3

# Optional HTTP API:
pip install -e '.[web]'
python -m kikuyu_ai.web.api
python -m ui.gradio_app
```

### Audio formats

The pipeline takes any recording `ffmpeg` can decode: mp3, m4a, aac, ogg, opus,
webm, flac, wma, amr, aiff, caf, wv, ac3, and video files such as mp4, mov, mkv,
avi and ts, from which the audio track is extracted.

The file extension is a hint, never a gate. The format is identified from the
bytes, so a recording named `recording` with no extension works, and AAC audio
saved under a `.wav` name works too — both are common on phones. A file that is
empty, or that holds no readable audio, is rejected with a reason rather than
failing deep in the pipeline.

Install `ffmpeg` separately. A plain PCM WAV is handled in-process and needs no
ffmpeg; every other format requires it. Configure `KIKUYU_TTS_COMMAND` with a Piper-compatible command that accepts text on stdin and `--output_file PATH` if you want translated speech output.

## Online FastAPI mode

For Android/Termux, the fastest setup is to run the models on a stronger machine and use the phone as a client. On the server:

```sh
pip install -e ".[web,ml]"
KIKUYU_API_HOST=0.0.0.0 KIKUYU_API_PORT=8000 python -m kikuyu_ai.web.api
```

On Termux, point the app at that server URL. The CLI, `access_translator.py`, and Gradio UI will upload to FastAPI instead of loading local models:

```sh
export KIKUYU_API_URL=http://SERVER_IP:8000
python -m kikuyu_ai.cli recording.mp3
python access_translator.py --audio recording.mp3
python -m ui.gradio_app
```

Text-only remote translation is available through `POST /translate-text` and `python access_translator.py --text "..."` when `KIKUYU_API_URL` is set.

## Model configuration

The app loads configuration from environment variables and from `.env` in the project root. This checkout is configured for:

```sh
KIKUYU_ASR_BACKEND=transformers
KIKUYU_ASR_MODEL=Kiragu/whisper-small-kikuyu-v5
KIKUYU_TRANSLATION_MODEL=nickdee96/nllb-200-600m-kikuyu-english
KIKUYU_TRANSLATION_SRC_LANG=kik_Latn
KIKUYU_TRANSLATION_TGT_LANG=eng_Latn
```

On Android/Termux or other low-memory systems, keep the default memory-safe behavior unless you know the device has enough RAM. The app loads model files offline only, skips low-energy audio chunks, retries repetitive ASR chunks at smaller boundaries, limits Torch CPU threads, and unloads ASR before loading translation by default. Tune these with:

```sh
KIKUYU_ASR_CHUNK_SECONDS=12
KIKUYU_ASR_RETRY_CHUNK_SECONDS=6
KIKUYU_ASR_SILENCE_RMS=0.001
KIKUYU_ASR_MAX_NEW_TOKENS=96
KIKUYU_ASR_NO_REPEAT_NGRAM_SIZE=3
KIKUYU_ASR_REPETITION_PENALTY=1.15
KIKUYU_TRANSLATION_MAX_NEW_TOKENS=128
KIKUYU_TORCH_THREADS=1
KIKUYU_KEEP_ASR_LOADED=0
KIKUYU_KEEP_TRANSLATION_LOADED=0
```

For ASR verification without loading the larger translation model, run `python -m kikuyu_ai.cli --asr-only recording.mp3`.

Install the optional ML stack before running real transcription or translation:

```sh
pip install -e '.[ml]'
```

The configured ASR model is an experimental Kikuyu Whisper model and runs through Transformers, not `faster-whisper`. The configured translation model is an NLLB-based Kikuyu-to-English model, so the app passes `kik_Latn` and `eng_Latn` into the translation pipeline. Translation runs segment-by-segment so long recordings are not truncated to one tokenizer window. Each session writes `asr_segments.jsonl`, `english_segments.jsonl`, `subtitles.srt`, and `english_subtitles.srt` when the corresponding segment data is available.

For a CTranslate2 Whisper model, install `pip install -e '.[ct2]'`, set `KIKUYU_ASR_BACKEND=faster-whisper`, set `KIKUYU_ASR_MODEL` to the model name or local model directory, and optionally set `KIKUYU_ASR_LANGUAGE` if the model supports an explicit language code.

## Training data

Three sources feed the translator, and all of them are built locally with no API
keys. Each writes a JSONL file of `{"kikuyu": ..., "english": ...}` rows, and a
final step merges them into one training file.

### 1. The Bible, in both languages

```sh
python scripts/fetch_bible_corpus.py
python scripts/build_parallel_dataset.py \
    data/bible/raw/kik_vpl.zip data/bible/raw/engwebp_vpl.zip \
    data/bible/parallel.jsonl --translation-output data/translation/bible.jsonl
```

This is the only step that needs a network connection, and only once. It pulls
verse-per-line archives from eBible.org and writes the license notice next to the
download. The defaults are deliberately open-licensed: the Kikuyu text is Biblica's
Open Kikuyu edition under CC BY-SA 4.0, and the English is the World English Bible,
which is public domain. **CC BY-SA requires you to keep the attribution and to
share derivative work under the same licence** — that includes a model trained on
it, so check this against how you plan to distribute the result. `--list` shows the
known codes; `scripts/download_bible_data.py` still takes explicit URLs if you have
your own licensed export.

The two translations align on 31,094 verses across all 66 books, which is the bulk
of the training data. `data/bible/parallel.jsonl` also serves the app at runtime:
when someone speaks a verse, the exact published English is returned instead of a
machine translation.

### 2. PDF translations

```sh
pip install -e '.[pdf]'

# The same document as two PDFs, one per language:
python scripts/ingest_pdf_corpus.py paired kikuyu.pdf english.pdf

# Or one PDF holding both languages:
python scripts/ingest_pdf_corpus.py bilingual book.pdf
```

Text is extracted with `pypdf`, or poppler's `pdftotext` if that is what you have.
Paired documents are aligned with the Gale-Church length model, so no dictionary
and no model is involved; bilingual documents are split into language runs and
paired two at a time. Both modes drop running headers, page numbers and verse
numbers, then filter out pairs that are too short, too long, too lopsided, or
identical on both sides.

A scanned PDF with no text layer is reported rather than silently producing
nothing. Run OCR on it first (`ocrmypdf in.pdf out.pdf`) and re-run.

Check the printed sample rows before training. A misaligned PDF is worse than no
PDF, because every row teaches the model a wrong pair.

### 3. Voice translations

```sh
python scripts/ingest_voice_corpus.py --asr-manifest data/asr/manifest.jsonl
```

This reads three local sources:

- **Corrections** saved from the app's *Improve this translation* box.
- **Recordings with sidecars** in `data/voice/`: put `clip01.wav` next to
  `clip01.kikuyu.txt` (what was said) and `clip01.english.txt` (what it means).
  These train both the translator and the speech recogniser.
- **Reviewed sessions**, marked by writing `kikuyu.corrected.txt` or
  `english.corrected.txt` into the session folder.

Raw uncorrected app output is excluded unless you pass `--include-machine-output`.
Training on the model's own unreviewed guesses reinforces its existing
hallucinations rather than fixing them, so the flag exists but the default does not
use it.

`--asr-manifest` writes the manifest that `training/train_asr.py` expects, so the
same recordings improve transcription too.

### Merge and train

```sh
python scripts/build_training_corpus.py --repeat data/translation/voice.jsonl=8
python training/train_translation.py data/translation/parallel.jsonl \
    --output models/translation/kikuyu-english
```

The builder deduplicates across every source, filters unusable rows, and holds out
an eval split. Later inputs win on duplicates, so a hand-corrected sentence
replaces the bulk-imported version of the same sentence.

Voice and PDF rows are usually a rounding error next to 31k Bible verses, which
biases the model toward scripture register. `--repeat FILE=N` upsamples the small
high-value corpora; held-out eval rows are never upsampled, so evaluation stays
honest. Point the app at the result with
`KIKUYU_TRANSLATION_MODEL=models/translation/kikuyu-english`.

## API

`GET /` returns the installable mobile app; the old status page with links to
`/health`, `/docs` and `/redoc` moved to `/status`. `POST /translate` with multipart field `file` returns the Kikuyu transcript, English result, confidence values, optional Bible match, and session file paths. `POST /corrections` stores human corrections in `data/corrections.db` for later fine-tuning.

## ASR fine-tuning

Create a JSONL manifest with one utterance per line:

```json
{"audio":"data/asr/train/clip001.wav","text":"Kikuyu transcript here","split":"train"}
{"audio":"data/asr/dev/clip101.wav","text":"Kikuyu transcript here","split":"eval"}
```

Install the training stack and run the Whisper-compatible trainer:

```sh
pip install -e '.[training]'
python training/train_asr.py data/asr/manifest.jsonl --output models/asr/kikuyu-whisper
```

Use `--eval-manifest data/asr/eval.jsonl` when train and eval rows are in separate files. If no explicit eval rows or eval manifest are provided, the script uses `--validation-split 0.1`. It normalizes audio through the app's existing audio pipeline, caches 16 kHz mono WAV files under the output directory, and saves the fine-tuned model plus processor to `--output`.

After training, point the app at the new local directory:

```sh
KIKUYU_ASR_MODEL=models/asr/kikuyu-whisper
```

## Translation fine-tuning

Create a licensed or corrected Kikuyu-English JSONL file:

```json
{"kikuyu":"Kikuyu source sentence","english":"English target sentence","split":"train"}
{"kikuyu":"Kikuyu eval sentence","english":"English eval sentence","split":"eval"}
```

Fine-tune the configured NLLB model:

```sh
python training/train_translation.py data/translation/parallel.jsonl --output models/translation/kikuyu-english
```

After training, point the app at the new local directory:

```sh
KIKUYU_TRANSLATION_MODEL=models/translation/kikuyu-english
```

Real retraining requires corrected labels. Do not train on the app's own uncorrected output as if it were ground truth; that usually reinforces the same hallucinations and repetitions.

The Gradio UI supports upload and microphone input when `gradio` is installed.
