#!/usr/bin/env python3
"""Download (or list) the models for a backend.

Usage:
    python scripts/download_models.py --backend mac --list
    python scripts/download_models.py --backend mac            # download via huggingface_hub
    python scripts/download_models.py --backend cuda --list

The manifest pins each model's repo, revision, and license. Hugging Face models are
fetched with `huggingface_hub` (install the backend extra: `pip install -e '.[mac]'`).
Ollama / vLLM models are pulled by their own runtimes — the script prints the command.

License note (M0 task): the licenses below are recorded from each model card. Re-verify
before any redistribution — licenses drift. See README.md "Models & licenses".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class Model:
    stage: str
    repo: str
    revision: str  # pin a commit/tag where possible; "main" = follow latest (revisit)
    license: str
    fetch: str  # "hf" | "ollama" | "vllm"
    note: str = ""


# Revisions left as "main" are intentionally TODO: pin to a specific commit before relying
# on reproducibility. Licenses verified against each model card (2026-06); see the caveats.
#
# License caveats (matter for redistribution / commercial use):
#   * Orpheus-3b-0.1-ft is tagged Apache-2.0 but its weights are fine-tuned from
#     Llama-3.2-3B-Instruct, so Meta's Llama 3.2 Community License ALSO applies
#     (attribution + acceptable-use terms). Chatterbox (MIT) is the clean alternative.
#   * F5-TTS pretrained weights (the M5 Mac cloning option, f5-tts-mlx) are CC-BY-NC
#     (non-commercial) because of the Emilia training set, even though the CODE is MIT.
#     For commercial use prefer Kokoro (Apache-2.0, no clone) or an Apache-licensed
#     OpenF5 checkpoint, or do cloning on the CUDA side with Chatterbox (MIT).
MANIFEST: dict[str, list[Model]] = {
    "mac": [
        Model("stt", "mlx-community/whisper-large-v3-turbo", "main", "MIT", "hf"),
        Model(
            "llm",
            "qwen2.5:7b-instruct",
            "n/a",
            "Apache-2.0",
            "ollama",
            note="run: ollama pull qwen2.5:7b-instruct",
        ),
        Model("tts", "hexgrad/Kokoro-82M", "main", "Apache-2.0", "hf"),
        # f5-tts-mlx (M5 cloning) is fetched on first use; weights are CC-BY-NC (see above).
    ],
    "cuda": [
        Model("stt", "Systran/faster-whisper-large-v3", "main", "MIT", "hf"),
        Model(
            "llm",
            "Qwen/Qwen2.5-7B-Instruct",
            "main",
            "Apache-2.0",
            "vllm",
            note="served by vLLM; AWQ variant Qwen/Qwen2.5-7B-Instruct-AWQ for 4-bit",
        ),
        Model(
            "tts",
            "canopylabs/orpheus-3b-0.1-ft",
            "main",
            "Apache-2.0 + Llama-3.2",
            "hf",
            note="weights derive from Llama-3.2-3B -> Llama 3.2 license also applies; "
            "Chatterbox (ResembleAI/chatterbox, MIT) is the clean-license alternative",
        ),
    ],
}


def _print_list(backend: str) -> None:
    print(f"Models for backend={backend}:\n")
    for m in MANIFEST[backend]:
        print(f"  [{m.stage}] {m.repo}")
        print(f"        revision={m.revision}  license={m.license}  via={m.fetch}")
        if m.note:
            print(f"        {m.note}")
    print()
    print("License note: verify each license on its model card before redistribution.")


def _download(backend: str, models_dir: str) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Install the backend extra, e.g.:\n"
            f"    pip install -e '.[{backend}]'",
            file=sys.stderr,
        )
        return 1

    rc = 0
    for m in MANIFEST[backend]:
        if m.fetch != "hf":
            print(f"[{m.stage}] {m.repo}: {m.note or 'fetched by its own runtime; skipping'}")
            continue
        print(f"[{m.stage}] downloading {m.repo} (rev={m.revision}) ...")
        try:
            path = snapshot_download(
                repo_id=m.repo,
                revision=None if m.revision in ("main", "n/a") else m.revision,
                cache_dir=models_dir,
            )
            print(f"        -> {path}")
        except Exception as exc:  # report and continue with the rest of the manifest
            print(f"        FAILED: {exc}", file=sys.stderr)
            rc = 1
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/list Persona-Voice models.")
    parser.add_argument("--backend", choices=("mac", "cuda"), required=True)
    parser.add_argument("--list", action="store_true", help="list models without downloading")
    parser.add_argument("--models-dir", default="models", help="cache dir for downloads")
    args = parser.parse_args(argv)

    if args.list:
        _print_list(args.backend)
        return 0
    return _download(args.backend, args.models_dir)


if __name__ == "__main__":
    raise SystemExit(main())
