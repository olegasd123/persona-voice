# Bundled voices

Out-of-the-box voice samples so the voice library isn't empty on first run.

The two bundled wavs ship as **presets** (`config/voices.yaml`, their `sample` field):

- `Feminine.wav` → preset `Feminine`
- `Masculine.wav` → preset `Masculine`

On a cloning backend (`chatterbox` on CUDA) the backend zero-shot-clones the sample, so these
are speakable per session with **no enrollment step**. They don't map onto Kokoro, so they
don't appear on the Mac backend.

To enroll **your own** wavs as clones instead, drop them here and run:

```bash
python scripts/seed_voices.py      # or: personavoice-seed-voices
```

Each `*.wav` is enrolled as a clone named after its **file stem**. Enrollment is **idempotent**
(already-present clones are skipped; `--force` re-enrolls) and goes through the same path a
client upload uses (`VoiceCloner` → `ClonesStore`). On a preset-only backend it still enrolls
but lists `available=false` in `GET /voices`.

**Format:** any mono/stereo WAV of ~2–60 s of clear speech works — the audio codec mixes to
mono and resamples on enroll, so no pre-processing is needed.

**Likeness/consent:** only add a voice you're authorized to use (see the licensing notes in the
main README's voice section).
