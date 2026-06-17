#!/usr/bin/env python3
"""Generate the Persona Voice app icon for iOS and Android.

Design: a "two-voice dialogue" waveform — two mirrored groups of rounded bars
rising toward the centre — on a subtle violet→indigo diagonal gradient. The left
voice is white, the right voice is lavender, evoking speech-to-speech (persona ↔ you).

Run from the client/ directory:
    python3 tool/generate_app_icon.py

Outputs (overwrites in place):
  - iOS  : ios/Runner/Assets.xcassets/AppIcon.appiconset/*.png  (opaque, full-bleed)
  - Andr : android/.../mipmap-*/ic_launcher.png            (legacy, rounded)
           android/.../mipmap-*/ic_launcher_foreground.png  (adaptive foreground)
           android/.../mipmap-*/ic_launcher_monochrome.png  (themed-icon layer)
  - Master: assets/icon/app_icon.png (1024, source of truth)

The Android adaptive XML + gradient background drawable are checked into the repo
(see android/.../mipmap-anydpi-v26/ic_launcher.xml and drawable/ic_launcher_background.xml).
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = os.path.dirname(HERE)

# ---- palette ---------------------------------------------------------------
GRAD_TL = (139, 121, 230)   # #8B79E6  light violet (top-left)
GRAD_BR = (67, 51, 143)     # #43338F  deep indigo (bottom-right)
VOICE_L = (255, 255, 255)   # white   – left voice
VOICE_R = (197, 184, 255)   # #C5B8FF lavender – right voice

# ---- glyph geometry (in 1024-px design space, scaled to any canvas) --------
BASE = 1024
# Concept #4 "two-voice dialogue" — bar layout in a 150-unit design tile.
# (x_centre, top, bottom). Left group is the white voice, right group the lavender voice;
# the right voice hangs slightly lower, giving the staggered call-and-response look.
SRC_BARS_L = ((40, 62, 78), (56, 50, 90), (72, 40, 100))
SRC_BARS_R = ((94, 58, 110), (110, 48, 100), (126, 70, 88))
SRC_BAR_W = 8.0
GLYPH_WIDTH_FRAC = 0.66     # glyph width as a fraction of the canvas (at GLYPH_SCALE = 1.0)
GLYPH_SCALE = 1.00          # overall size multiplier; lower for more padding
SS = 4                      # supersampling factor for crisp anti-aliasing


def make_gradient(size: int) -> Image.Image:
    """Diagonal top-left → bottom-right linear gradient, opaque RGB."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    t = (gx + gy) * 0.5                      # 0 at TL, 1 at BR
    c0 = np.array(GRAD_TL, dtype=np.float32)
    c1 = np.array(GRAD_BR, dtype=np.float32)
    grad = c0[None, None, :] * (1.0 - t)[..., None] + c1[None, None, :] * t[..., None]
    return Image.fromarray(np.clip(grad, 0, 255).astype("uint8"))


def draw_glyph(size: int, scale: float, mono: bool = False) -> Image.Image:
    """Centred two-voice 'dialogue' waveform (concept #4) on a transparent RGBA canvas."""
    bars = [(x, t, b, VOICE_L) for (x, t, b) in SRC_BARS_L]
    bars += [(x, t, b, VOICE_L if mono else VOICE_R) for (x, t, b) in SRC_BARS_R]
    r = SRC_BAR_W / 2
    x_min = min(x for x, _, _, _ in bars) - r
    x_max = max(x for x, _, _, _ in bars) + r
    y_min = min(t for _, t, _, _ in bars) - r
    y_max = max(b for _, _, b, _ in bars) + r
    bbox_w = x_max - x_min
    bcx, bcy = (x_min + x_max) / 2, (y_min + y_max) / 2      # bbox centre → canvas centre

    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = (big * GLYPH_WIDTH_FRAC * scale) / bbox_w            # 150-space → supersampled px
    cx = cy = big / 2
    bar_w = SRC_BAR_W * s
    radius = bar_w / 2

    for x, top, bot, color in bars:
        cxp = cx + (x - bcx) * s
        y0 = cy + (top - bcy) * s
        y1 = cy + (bot - bcy) * s
        d.rounded_rectangle([cxp - bar_w / 2, y0, cxp + bar_w / 2, y1], radius=radius, fill=color)

    return img.resize((size, size), Image.LANCZOS)


def rounded_mask(size: int, radius_frac: float = 0.225) -> Image.Image:
    """Anti-aliased rounded-square alpha mask (for legacy Android icons)."""
    big = size * SS
    m = Image.new("L", (big, big), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, big - 1, big - 1], radius=int(big * radius_frac), fill=255
    )
    return m.resize((size, size), Image.LANCZOS)


def build_master() -> Image.Image:
    """Full-bleed 1024 gradient + glyph (padded framing), RGBA."""
    base = make_gradient(BASE).convert("RGBA")
    base.alpha_composite(draw_glyph(BASE, GLYPH_SCALE))
    return base


def save(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"  {os.path.relpath(path, CLIENT)}  ({img.width}x{img.height})")


def main() -> None:
    master = build_master()                  # RGBA, full-bleed

    # ---- iOS: opaque, full-bleed, no alpha (system rounds corners) ----
    ios_dir = os.path.join(CLIENT, "ios/Runner/Assets.xcassets/AppIcon.appiconset")
    ios_master = master.convert("RGB")
    ios_sizes = {
        "Icon-App-20x20@1x.png": 20, "Icon-App-20x20@2x.png": 40, "Icon-App-20x20@3x.png": 60,
        "Icon-App-29x29@1x.png": 29, "Icon-App-29x29@2x.png": 58, "Icon-App-29x29@3x.png": 87,
        "Icon-App-40x40@1x.png": 40, "Icon-App-40x40@2x.png": 80, "Icon-App-40x40@3x.png": 120,
        "Icon-App-60x60@2x.png": 120, "Icon-App-60x60@3x.png": 180,
        "Icon-App-76x76@1x.png": 76, "Icon-App-76x76@2x.png": 152,
        "Icon-App-83.5x83.5@2x.png": 167, "Icon-App-1024x1024@1x.png": 1024,
    }
    print("iOS:")
    for name, px in ios_sizes.items():
        img = ios_master if px == 1024 else ios_master.resize((px, px), Image.LANCZOS)
        save(img, os.path.join(ios_dir, name))

    # ---- Android legacy: rounded, RGBA (pre-API-26 launchers) ----
    res = os.path.join(CLIENT, "android/app/src/main/res")
    densities = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}
    legacy_master = master.copy()
    legacy_master.putalpha(rounded_mask(BASE))
    print("Android legacy (ic_launcher.png):")
    for d, mult in densities.items():
        px = int(48 * mult)
        save(legacy_master.resize((px, px), Image.LANCZOS),
             os.path.join(res, f"mipmap-{d}", "ic_launcher.png"))

    # ---- Android adaptive foreground + monochrome (108dp canvas, 72dp safe zone) ----
    fg_master = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    fg_master.alpha_composite(draw_glyph(BASE, GLYPH_SCALE))     # glyph inside safe zone
    mono_master = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    mono_master.alpha_composite(draw_glyph(BASE, GLYPH_SCALE, mono=True))
    print("Android adaptive (foreground + monochrome):")
    for d, mult in densities.items():
        px = int(108 * mult)
        save(fg_master.resize((px, px), Image.LANCZOS),
             os.path.join(res, f"mipmap-{d}", "ic_launcher_foreground.png"))
        save(mono_master.resize((px, px), Image.LANCZOS),
             os.path.join(res, f"mipmap-{d}", "ic_launcher_monochrome.png"))

    # ---- master source ----
    print("Master:")
    save(master.convert("RGB"), os.path.join(CLIENT, "assets/icon/app_icon.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
