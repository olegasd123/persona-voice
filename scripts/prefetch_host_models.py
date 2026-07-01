#!/usr/bin/env python3
"""Pre-download the host-side STT/TTS weights for the CUDA stack.

The CUDA stack serves the LLM with vLLM in Docker, but the STT (faster-whisper) and TTS
(Chatterbox) models load *on the host*, inside the conversation worker, and download from
Hugging Face to the host cache on first use. setup-cuda runs this once so the first real
`run-cuda` doesn't stall on those downloads.

Best-effort by design: a failure here is not fatal (the weights simply download on the first
run instead), so each stage is wrapped and the script always exits 0. Models are downloaded
into the host's Hugging Face cache (the same cache the worker reads), not into ./models.
"""

from __future__ import annotations

import os
import sys


def _warm_stt() -> None:
    # Matches config/backends/cuda.yaml's default (PERSONAVOICE_STT_MODEL override honoured).
    model = os.environ.get("PERSONAVOICE_STT_MODEL", "large-v3-turbo")
    from faster_whisper import download_model  # downloads the CT2 model; no GPU needed

    print(f"[stt] downloading faster-whisper '{model}' ...")
    path = download_model(model)
    print(f"      -> {path}")


def _warm_tts() -> None:
    # Chatterbox has no download-only API; load the base on CPU to populate the host cache.
    from chatterbox.tts import ChatterboxTTS

    print("[tts] downloading Chatterbox base weights (loading on CPU to fetch them) ...")
    ChatterboxTTS.from_pretrained(device="cpu")
    print("      -> cached")


def main() -> int:
    for name, fn in (("stt", _warm_stt), ("tts", _warm_tts)):
        try:
            fn()
        except Exception as exc:  # best-effort: weights will download on first run instead
            print(f"[{name}] skipped ({exc}); it will download on the first run.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
