# Voice fine-tuning (M9)

Train a **high-fidelity voice** for a target speaker — beyond the M5 zero-shot clone. Where a
clone is *reference conditioning* (the TTS speaks in the voice of a stored sample at generation
time), a fine-tune adapts the TTS model itself to the speaker on a small dataset, for higher
fidelity. The workflow is one CLI — `personavoice-voice-train` — backed by
`personavoice.training.voice`. Logic is pure and unit-tested; the heavy trainer is shelled out to
(`f5-tts_finetune-cli` or a Chatterbox trainer), so the repo installs and tests without a GPU.

```
dataset ──▶ run (prepare + train) ──▶ eval (A/B vs clone) ──▶ register (fold into the registry)
```

A fine-tuned voice is stored in `<models>/finetuned/finetuned.json` and, once **assigned** to a
persona, takes precedence in `VoiceRegistry.resolve_for_persona`: **fine-tuned ▶ clone ▶ preset**.
The cloning TTS adapters (Chatterbox/F5) load the trained checkpoint via `VoiceRef.model_path`.

## Engines & licenses

| Engine | Trainer | License | Use |
|--------|---------|---------|-----|
| **f5** | `f5-tts_finetune-cli` (canonical, mature) | weights **CC-BY-NC** | dev / personal personas |
| **chatterbox** | community trainer (`trainer_script`) | **MIT** | the clean-license / shippable path |

F5-TTS has the canonical, reproducible finetune CLI, so it's the default engine — but its default
checkpoint's *weights* are CC-BY-NC (non-commercial). A voice fine-tuned on **your own / a
consented speaker's** clips for a personal persona is fine; don't redistribute the weights
commercially. For anything shipped, fine-tune **Chatterbox** (MIT) — it has no official CLI, so
point `trainer_script` at a community Chatterbox trainer. Voice fine-tuning is **CUDA-only** (the
training track); the Mac path stays zero-shot cloning (M5).

> **Consent.** Only fine-tune voices from samples you're authorized to use (same gate as M5
> cloning). See the plan's "Voice-clone misuse / consent" risk row.

## 1. Dataset

A target-speaker dataset is a `metadata.csv` of pipe-separated `audio_path|text` lines (the
LJSpeech / F5 `prepare_csv_wavs` shape), with the WAVs under a `wavs/` subdir:

```
training/voice/datasets/my_voice/
├── metadata.csv
└── wavs/clip_0001.wav, clip_0002.wav, ...
```

Build it by auto-transcribing a folder of clips with the cascade's STT (or pass an existing
`metadata.csv`):

```
personavoice-voice-train dataset --voice my_voice --audio-dir clips/ --probe-durations
```

`datasets/sample/metadata.csv` shows the format. Aim for **a few minutes** of clean speech across
**10+ clips of a few seconds each** (`validate_dataset` enforces this when durations are known).

## 2. Train (CUDA)

```
# preview the prepare + trainer commands (writes any config, launches nothing)
personavoice-voice-train run --voice my_voice --engine f5 --dry-run

# launch — F5 first builds its arrow dataset (f5-tts_prepare_csv_wavs), then fine-tunes
personavoice-voice-train run --voice my_voice --engine f5 \
  --config training/voice/configs/my_voice.f5.yaml
```

The heavy step runs in the CUDA training image (the trainers aren't in the light `cuda` wheel,
same policy as vLLM/LLaMA-Factory). F5-TTS: `pip install f5-tts` in the image. On a Blackwell GPU
(sm_120, RTX 5090) use cu128 torch wheels (the M2/M7 notes apply). Checkpoints land in
`<models>/finetuned/<voice>/`.

### Verified runner (the real f5-tts ≥1.1 flow)

The `run` plan-builder is the declarative intent; the **verified end-to-end CUDA runner** is
`run_finetune.sh`, built + run in `Dockerfile.blackwell`. It exists because f5-tts ≥1.1 differs
from the older docs the plan was written against:

- prep is now a **module** (`python -m f5_tts.train.datasets.prepare_csv_wavs <CSV> <OUT>`), not
  the `f5-tts_prepare_csv_wavs` script, and wants a **CSV with a `audio_file|text` header and
  ABSOLUTE wav paths** (not the repo's relative `wavs/…` manifest);
- F5 fixes its data root (`…/data/<dataset>_<tokenizer>/`) and ckpt root (`…/ckpts/<dataset>/`)
  by `--dataset_name`; finetune mode also needs the base model's **pinyin vocab**
  (`Emilia_ZH_EN_pinyin/vocab.txt`, fetched from `SWivid/F5-TTS`), which isn't pip-bundled;
- `--batch_size_per_gpu` is in **frames** under F5's default `frame` batching (the plan's bare
  `4` would be 4 frames), and `--logger`'s choices include the Python `None` object, so it must
  be omitted rather than passed `None`.

`run_finetune.sh` reconciles all of that (build CSV → fetch vocab → prepare → finetune → copy the
checkpoint back to the mounted `<models>/finetuned/<voice>/`):

```
docker build -f training/voice/Dockerfile.blackwell -t personavoice/voice-trainer:blackwell .
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace \
  -v persona-voice_hf-cache:/models/hf personavoice/voice-trainer:blackwell \
  bash training/voice/run_finetune.sh        # env: DATASET, EPOCHS, BATCH_SIZE, …
```

> **EMA note.** F5 saves both EMA (`use_ema=True`, the inference default) and raw trained weights.
> Its EMA uses `beta=0.9999` + `update_every=10` (a ~10⁵-step window), so a small fine-tune's EMA
> barely diverges from the base for thousands of steps — evaluate/deploy the **raw trained
> weights** (`--no-ema`) for a small-data fine-tune; they carry the speaker adaptation.

## 3. Eval — A/B vs the zero-shot clone

The acceptance bar is *"clearly higher fidelity than the zero-shot clone."* The proxy is
**speaker similarity**: synthesize the same probes with the fine-tuned voice and with the M5 clone,
embed both plus held-out **real** target clips, and compare cosine similarity to the real speaker.

```
personavoice-voice-train eval --voice my_voice --clone my_clone \
  --target-dir held_out_clips/ --margin 0.02 --register
```

Prints `fine-tuned similarity`, `clone similarity`, and the delta; **PASS** when the fine-tune
clears the target speaker by `--margin`. `--register` stores the verdict on the voice. Needs the
`voice-eval` extra (`pip install -e '.[voice-eval]'`, Resemblyzer speaker embeddings). Pair with a
human MOS spot-check for the final call.

`personavoice-voice-train eval` drives the A/B through the active **cascade** TTS adapter — but
there is no CUDA F5 *cascade* adapter (Chatterbox is the CUDA cloning adapter and can't load an F5
checkpoint). So for an F5 fine-tune on CUDA, run the dedicated harness `evaluate_f5_ab.py`, which
drives F5 directly (fine-tune vs F5 **zero-shot** — the fair same-engine baseline) yet still scores
with the repo's own pure `score_ab` + `ResemblyzerEmbedder`, and writes MOS sample wavs:

```
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace/src \
  -v persona-voice_hf-cache:/models/hf personavoice/voice-trainer:blackwell \
  python3.12 training/voice/evaluate_f5_ab.py --voice my_voice --margin 0.01 --no-ema
```

## 4. Register & assign (fold the winner in)

```
personavoice-voice-train register --voice my_voice \
  --checkpoint models/finetuned/my_voice --engine f5 --assign companion
personavoice-voice-train list
```

Assigning makes the persona speak through the fine-tuned checkpoint everywhere — the demos and the
live LiveKit agent pick it up via the voice registry (it outranks any clone assigned to that
persona). `server --check` lists fine-tuned voices + assignments and suppresses the
"won't sound distinct" warning for an assigned voice. Remove an assignment with
`personavoice-voice-train list --unassign <persona>`.

## Install

- **A/B eval (any machine):** `pip install -e '.[voice-eval]'`.
- **Trainers (CUDA image only):** F5-TTS (`pip install f5-tts`) or a community Chatterbox trainer
  referenced by `trainer_script` — heavy + need a CUDA toolchain, so they stay out of the light
  `cuda` wheel extra.
