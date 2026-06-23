#!/usr/bin/env python3
"""Enroll the bundled seed voices as clones (thin shim around `personavoice.voice.seed`).

Usage:
    python scripts/seed_voices.py                 # enroll assets/seed_voices/*.wav (idempotent)
    python scripts/seed_voices.py --force         # re-enroll even if already present
    python scripts/seed_voices.py --dir my_voices --backend cuda

Equivalent to the `personavoice-seed-voices` console script. Runs against the active backend
(f5_mlx on Mac, chatterbox on CUDA); see the module docstring for the preset-only behavior.
"""

from __future__ import annotations

import sys

from personavoice.voice.seed import main

if __name__ == "__main__":
    sys.exit(main())
