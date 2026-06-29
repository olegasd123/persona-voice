# Voice Fine-Tuning

Train a high-fidelity voice for a target speaker. A clone is reference conditioning: the TTS
speaks from a stored sample at generation time. A fine-tune adapts the TTS model itself to the
speaker on a small dataset.

The workflow uses `personavoice-voice-train`:

```text
dataset -> run -> eval -> register
```

A fine-tuned voice is stored in `<models>/finetuned/finetuned.json`. When it is assigned to a
persona, `VoiceRegistry.resolve_for_persona` uses this order:

```text
fine-tuned -> clone -> preset
```

Chatterbox is the supported voice fine-tune engine. The repo builds and tests the plan without a
GPU; the real trainer is a script path set by `trainer_script`.

## Consent

Only fine-tune voices from samples you are allowed to use. Do not clone or fine-tune a voice
without clear consent.

## 1. Dataset

A target-speaker dataset is a `metadata.csv` of pipe-separated `audio_path|text` lines, with the
WAV files under `wavs/`:

```text
training/voice/datasets/my_voice/
├── metadata.csv
└── wavs/clip_0001.wav, clip_0002.wav, ...
```

Build it by auto-transcribing a folder of clips with the cascade STT, or pass an existing
`metadata.csv`:

```bash
personavoice-voice-train dataset --voice my_voice --audio-dir clips/ --probe-durations
```

`datasets/sample/metadata.csv` shows the format. Aim for a few minutes of clean speech across
10+ short clips.

## 2. Train

Preview the trainer config and command:

```bash
personavoice-voice-train run --voice my_voice --dry-run
```

Run with a config:

```bash
personavoice-voice-train run --voice my_voice \
  --config training/voice/configs/my_voice.chatterbox.yaml
```

The heavy trainer should run in a CUDA training image. Set `trainer_script` to the Chatterbox
fine-tune script used by that image. Checkpoints land in `<models>/finetuned/<voice>/`.

## 3. Eval

The acceptance bar is clear improvement over the zero-shot clone. The eval synthesizes the same
probes with the fine-tuned voice and the clone, embeds both plus held-out real target clips, and
compares speaker similarity.

```bash
personavoice-voice-train eval --voice my_voice --clone my_clone \
  --target-dir held_out_clips/ --margin 0.02 --register
```

This needs the `voice-eval` extra:

```bash
pip install -e '.[voice-eval]'
```

Pair the score with a human audio check before shipping a voice.

## 4. Register

```bash
personavoice-voice-train register --voice my_voice \
  --checkpoint models/finetuned/my_voice --engine chatterbox --assign companion
personavoice-voice-train list
```

The demos and LiveKit agent pick up the assigned voice through the voice registry. Remove an
assignment with:

```bash
personavoice-voice-train list --unassign <persona>
```

## Install

- A/B eval: `pip install -e '.[voice-eval]'`.
- Trainer: install the Chatterbox trainer in your CUDA image and point `trainer_script` at it.
