# Seed voices

Bundled out-of-the-box voice samples so the voice library isn't empty on first run (N4).

Each `*.wav` here is enrolled as a clone named after its **file stem** by:

```bash
python scripts/seed_voices.py      # or: personavoice-seed-voices
```

- `female.wav` → clone `female`
- `male.wav` → clone `male`

Enrollment is **idempotent** (already-present clones are skipped; `--force` re-enrolls) and
goes through the same path a client upload uses (`VoiceCloner` → `ClonesStore`). On a cloning
backend (`f5_mlx` on Mac, `chatterbox` on CUDA) the clone is audible; on a preset-only backend
it still enrolls but lists `available=false` in `GET /voices`.

**Format:** any mono/stereo WAV of ~2–60 s of clear speech works — the audio codec mixes to
mono and resamples on enroll, so no pre-processing is needed.

**Likeness/consent:** only add a voice you're authorized to use (see the licensing notes in the
main README's voice section).
