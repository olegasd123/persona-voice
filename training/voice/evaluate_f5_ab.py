"""Voice A/B: F5 fine-tuned voice vs F5 zero-shot, scored with the repo's speaker-similarity proxy.

The repo's `evaluate_voices` orchestrates the A/B through a persona-voice TTS adapter, but there
is no CUDA F5 *cascade* adapter (Chatterbox is the CUDA cloning adapter, and it can't load an F5
checkpoint). So on CUDA we drive F5 directly here — yet still score with the project's own pure
`score_ab` + `ResemblyzerEmbedder`, so the verdict is the repo's metric, not a bespoke one.

The fair same-engine baseline for an F5 fine-tune is **F5 zero-shot** (the base weights conditioned
on a held-out reference clip): both the fine-tune and the baseline use the *same* reference, so the
only difference is whether the model has been adapted to the speaker. Speaker similarity is cosine
to held-out *real* target clips (Resemblyzer). Run inside the Blackwell trainer image:

    docker run --rm --gpus all -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace/src \
      -v persona-voice_hf-cache:/models/hf personavoice/voice-trainer:blackwell \
      python3.12 training/voice/evaluate_f5_ab.py --voice ljspeech --margin 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from personavoice.audio import read_wav_file  # noqa: E402
from personavoice.training.voice.evaluate import ResemblyzerEmbedder, score_ab  # noqa: E402

# The CLI's neutral probes, inlined to avoid the cli module's heavy import chain (server.config
# → dotenv) inside the trainer image. Kept in sync with cli.DEFAULT_VOICE_PROBES.
DEFAULT_VOICE_PROBES = (
    "Thanks so much for being here today.",
    "Let me think about that for a moment.",
    "That's a really interesting question.",
    "I'd love to hear more about what you mean.",
    "We can take this one step at a time.",
)


def pick_finetuned_ckpt(ckpt_dir: Path) -> Path:
    """The fine-tuned checkpoint (newest), excluding the copied `pretrained_` base."""
    cands = [
        p
        for p in list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.safetensors"))
        if not p.name.startswith("pretrained_")
    ]
    if not cands:
        raise SystemExit(f"no fine-tuned checkpoint in {ckpt_dir} (only the pretrained base?)")
    # Prefer model_last.*, else highest step number.
    last = [p for p in cands if "last" in p.stem]
    if last:
        return last[0]
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description="F5 fine-tune vs zero-shot A/B.")
    ap.add_argument("--voice", default="ljspeech")
    ap.add_argument("--ckpt-dir", default=None, help="default models/finetuned/<voice>")
    ap.add_argument("--heldout-dir", default=None, help="default data/<voice>_heldout")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--no-ema", action="store_true", help="use raw trained weights, not the EMA")
    ap.add_argument("--out", default=None, help="MOS sample dir (default <ckpt-dir>/ab_samples)")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir or f"models/finetuned/{args.voice}")
    heldout = Path(args.heldout_dir or f"data/{args.voice}_heldout")
    out_dir = Path(args.out or (ckpt_dir / "ab_samples"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = ckpt_dir / "vocab.txt"
    ft_ckpt = pick_finetuned_ckpt(ckpt_dir)
    ref_line = (heldout / "ref.txt").read_text(encoding="utf-8").strip()
    ref_name, ref_text = ref_line.split("|", 1)
    ref_wav = heldout / ref_name

    # Target = held-out real clips OTHER than the synthesis reference (independent of it).
    target_clips = sorted(p for p in heldout.glob("*.wav") if p.name != ref_name)
    probes = list(DEFAULT_VOICE_PROBES)
    print(f"A/B '{args.voice}': fine-tuned {ft_ckpt.name} vs F5 zero-shot")
    print(f"  ref={ref_name}  probes={len(probes)}  target_clips={len(target_clips)}")

    from f5_tts.api import F5TTS

    use_ema = not args.no_ema
    print(f"  use_ema={use_ema}")
    ft = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(ft_ckpt), vocab_file=str(vocab),
               use_ema=use_ema, device="cuda")
    # Zero-shot baseline = the exact base the fine-tune started from (the copied pretrained_*),
    # so the A/B isolates the fine-tune; fall back to the Hub default if it's absent.
    base_ckpt = next(ckpt_dir.glob("pretrained_*.safetensors"), None)
    if base_ckpt is not None:
        base = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(base_ckpt), vocab_file=str(vocab), device="cuda")
    else:
        base = F5TTS(model="F5TTS_v1_Base", device="cuda")

    def synth(model: object, probe: str, path: Path) -> bytes:
        model.infer(  # type: ignore[attr-defined]
            ref_file=str(ref_wav), ref_text=ref_text, gen_text=probe,
            file_wave=str(path), seed=42, remove_silence=False,
        )
        return read_wav_file(path)

    emb = ResemblyzerEmbedder()
    target_embs = [emb.embed(read_wav_file(c)) for c in target_clips]

    ft_embs, cl_embs = [], []
    for i, probe in enumerate(probes):
        ft_wav = out_dir / f"probe{i:02d}_finetuned.wav"
        cl_wav = out_dir / f"probe{i:02d}_zeroshot.wav"
        ft_embs.append(emb.embed(synth(ft, probe, ft_wav)))
        cl_embs.append(emb.embed(synth(base, probe, cl_wav)))
        print(f"  [{i + 1}/{len(probes)}] {probe!r}")

    scores = score_ab(ft_embs, cl_embs, target_embs, margin=args.margin)
    print("\n==== Voice A/B (speaker similarity to held-out real target) ====")
    print(f"  fine-tuned similarity : {scores.finetuned_similarity:.4f}")
    print(f"  zero-shot similarity  : {scores.clone_similarity:.4f}")
    print(f"  delta (ft - zeroshot) : {scores.delta:+.4f}   (margin {scores.margin:.3f})")
    print(f"  verdict               : {'PASS — fine-tune clearly closer' if scores.passes else 'no clear win'}")
    print(f"  MOS samples           : {out_dir}")

    (ckpt_dir / "ab_result.json").write_text(scores.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps(scores.model_dump(), indent=2))
    return 0 if scores.passes else 3


if __name__ == "__main__":
    raise SystemExit(main())
