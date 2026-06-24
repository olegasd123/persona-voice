"""Seed voices: enroll the bundled out-of-the-box clones so the library isn't empty.

N1 (the per-persona voice picker) and N2 (the Voice Library page) are only interesting
once there's something to pick. This ships two ready clones — `female` / `male` — under
`assets/seed_voices/`, enrolled on first run through the **same path a client upload uses**
(`VoiceCloner` → `ClonesStore.record`), so they behave exactly like a user-enrolled voice.

Each `*.wav` in the seed directory becomes a clone named after the file stem. Enrollment is
**idempotent**: an already-present clone is skipped unless `--force`. On a cloning backend
(`f5_mlx` on Mac, `chatterbox` on CUDA) the full cloner runs (auto-transcribing the
reference text F5 needs); on a preset-only backend the sample is still recorded so it
appears in `GET /voices` with `available=false` (the catalog already gates that). The clones
land in the shared `ClonesStore`, so a multi-user deployment shows them to every user.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..adapters.factory import build_backend
from ..audio import read_wav_file
from ..server.config import (
    ConfigError,
    Settings,
    load_backend_config,
    load_clones_store,
)
from .clone import ClonedVoice, CloneError, ClonesStore, validate_sample

logger = logging.getLogger("personavoice.seed")

# Where the bundled seed wavs live (committed, unlike the gitignored models/). Override with
# --dir or PERSONAVOICE_SEED_VOICES_DIR.
DEFAULT_SEED_DIR = Path("assets/seed_voices")

# Enroll one sample under a clone name. Decoupling the loop from how a sample is enrolled
# keeps `seed_clones` testable with a fake enroller (no backend / model load).
EnrollFn = Callable[[str, bytes], Awaitable[None]]


@dataclass
class SeedResult:
    """What one seed run did: names enrolled, skipped (already present), and failed."""

    seeded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (name, error)

    @property
    def ok(self) -> bool:
        return not self.failed


def _resolve_seed_dir(seed_dir: str | Path | None) -> Path:
    """The seed-wav directory: explicit arg, else $PERSONAVOICE_SEED_VOICES_DIR, else default."""
    import os

    base = seed_dir or os.getenv("PERSONAVOICE_SEED_VOICES_DIR") or DEFAULT_SEED_DIR
    return Path(base).expanduser()


def seed_voice_names(seed_dir: str | Path | None = None) -> set[str]:
    """Names of the bundled "inbox" seed voices: the `*.wav` stems under the seed dir.

    These ship with the deployment (e.g. `male`/`female`) and are protected from deletion in
    the voice library — the catalog marks them `removable=False` and the server rejects a
    delete. Derived live from the seed dir (resolved the same way the seeder does), so changing
    the shipped set is just a matter of changing the files. A missing dir yields an empty set.
    """
    base = _resolve_seed_dir(seed_dir)
    if not base.is_dir():
        return set()
    return {path.stem for path in base.glob("*.wav")}


def discover_samples(seed_dir: str | Path) -> dict[str, bytes]:
    """Read every `*.wav` under `seed_dir`, keyed by the file stem (the clone name).

    A missing directory yields an empty mapping so a checkout without seed assets is a
    no-op rather than an error.
    """
    seed_dir = Path(seed_dir).expanduser()
    if not seed_dir.is_dir():
        return {}
    return {path.stem: read_wav_file(path) for path in sorted(seed_dir.glob("*.wav"))}


async def seed_clones(
    enroll: EnrollFn,
    store: ClonesStore,
    samples: dict[str, bytes],
    *,
    force: bool = False,
) -> SeedResult:
    """Enroll each sample as a clone, skipping ones already in `store` (unless `force`).

    A `CloneError` for one sample (e.g. too short) is captured in `failed` rather than
    aborting the run, so a single bad seed never blocks the others.
    """
    result = SeedResult()
    for name in sorted(samples):
        if not force and name in store:
            result.skipped.append(name)
            continue
        try:
            await enroll(name, samples[name])
            result.seeded.append(name)
        except CloneError as exc:
            result.failed.append((name, str(exc)))
    return result


def make_enroller(
    backend: object,
    store: ClonesStore,
    *,
    supports_cloning: bool,
) -> EnrollFn:
    """Build the enroll function for the active backend.

    Cloning backend → the full `VoiceCloner` path (transcribes the reference text). Preset-only
    backend → validate + persist the sample with a bare store record, so the seed still appears
    in the catalog (as `available=false`) instead of failing.
    """
    if supports_cloning:
        from .clone import VoiceCloner

        cloner = VoiceCloner(backend, store)

        async def _clone_enroll(name: str, wav: bytes) -> None:
            await cloner.clone(wav, name)

        return _clone_enroll

    async def _record_enroll(name: str, wav: bytes) -> None:
        validate_sample(wav)
        store.dir.mkdir(parents=True, exist_ok=True)
        dest = store.dir / f"{name}.wav"
        dest.write_bytes(wav)
        store.record(
            ClonedVoice(
                name=name,
                sample_path=str(dest),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    return _record_enroll


async def _run(settings: Settings, seed_dir: Path, *, force: bool) -> SeedResult:
    samples = discover_samples(seed_dir)
    if not samples:
        logger.warning("no seed wavs found in %s; nothing to seed", seed_dir)
        return SeedResult()
    store = load_clones_store(settings)
    backend = build_backend(load_backend_config(settings))
    supports_cloning = getattr(backend.tts, "supports_cloning", False)
    enroll = make_enroller(backend, store, supports_cloning=supports_cloning)
    result = await seed_clones(enroll, store, samples, force=force)
    if not supports_cloning and result.seeded:
        logger.warning(
            "active TTS %r can't clone; seeded %d voice(s) as catalog entries only "
            "(available=false until a cloning backend is active)",
            getattr(backend.tts, "name", "?"),
            len(result.seeded),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="personavoice-seed-voices",
        description="Enroll the bundled seed voices (assets/seed_voices/*.wav) as clones.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"seed wav directory (default ${{PERSONAVOICE_SEED_VOICES_DIR}} or {DEFAULT_SEED_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-enroll even if a clone of that name exists"
    )
    parser.add_argument("--backend", choices=("mac", "cuda"), default=None, help="override BACKEND")
    args = parser.parse_args(argv)

    from ..obs import configure_logging

    configure_logging()
    seed_dir = _resolve_seed_dir(args.dir)

    try:
        settings = Settings.load(backend=args.backend)
        result = asyncio.run(_run(settings, seed_dir, force=args.force))
    except (ConfigError, CloneError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # surface backend/model errors without a traceback wall
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.seeded:
        print(f"seeded {len(result.seeded)} voice(s): {', '.join(result.seeded)}")
    if result.skipped:
        print(f"skipped {len(result.skipped)} already present: {', '.join(result.skipped)}")
    for name, err in result.failed:
        print(f"failed {name}: {err}", file=sys.stderr)
    if not (result.seeded or result.skipped or result.failed):
        print(f"no seed wavs found in {seed_dir}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
