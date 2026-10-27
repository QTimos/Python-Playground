"""recolor.py - recolor any image to a fixed colour palette (Tokyo-Night style).

All colour math is done in OKLab, so gradients are smooth and perceptually even.

Examples
  python3 recolor.py photo.jpg                          # -> photo_recolored.png
  python3 recolor.py photo.jpg --size 3840x2160         # recolor + resize to 4K
  python3 recolor.py photo.jpg --mode hue               # keep distinct regions distinct
  python3 recolor.py photo.jpg --mode hue --accents blue,cyan,magenta
  python3 recolor.py *.png --ramp dusk --grain 0.02
  python3 recolor.py photo.jpg --ramp bg_dark,selection,blue,cyan,fg   # custom ramp
  python3 recolor.py --list

Modes
  gradient  brightness -> colour ramp (default). Smooth, classic duotone/gradient-map look.
  hue       brightness -> ramp, plus each pixel's hue is snapped to the nearest palette
            accent. Keeps trees/sky/water etc. visibly different. Best for colourful images.
  nearest   snap each pixel to the closest palette colour (flat, posterized).

Requires: pip install pillow numpy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

COLORS = {
    "bg": "#0F111A", "bg_alt": "#171B26", "bg_dark": "#0A0C14",
    "fg": "#C0CAF5", "fg_dark": "#A9B1D6", "grey": "#3B4261",
    "red": "#F7768E", "green": "#9ECE6A", "yellow": "#E0AF68",
    "blue": "#7AA2F7", "magenta": "#BB9AF7", "cyan": "#7DCFFF",
    "orange": "#FFB86C", "pink": "#F5C2E7", "selection": "#2A3A5A",
}
ACCENTS = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta", "pink"]

# dark -> light ramps (names from COLORS). Edit or add your own.
RAMPS = {
    "night":  ["bg_dark", "bg", "bg_alt", "selection", "grey", "blue", "cyan", "fg"],
    "dusk":   ["bg_dark", "bg", "selection", "grey", "blue", "magenta", "pink", "fg"],
    "warm":   ["bg_dark", "bg", "bg_alt", "grey", "magenta", "orange", "yellow", "fg"],
    "forest": ["bg_dark", "bg", "selection", "grey", "blue", "green", "cyan", "fg"],
    "mono":   ["bg_dark", "bg", "bg_alt", "selection", "grey", "fg_dark", "fg"],
}


# ----------------------------------------------------------------- colour math
def hex_to_srgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def srgb_to_oklab(rgb):
    r, g, b = [srgb_to_linear(rgb[..., i]) for i in range(3)]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], axis=-1)


def oklab_to_srgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return linear_to_srgb(np.stack([r, g, bl], axis=-1))


def palette_lab(names):
    return srgb_to_oklab(np.stack([hex_to_srgb(COLORS[n]) for n in names]))


# --------------------------------------------------------------------- helpers
def bayer(h, w, n=8):
    """Ordered-dither threshold map in [-0.5, 0.5)."""
    m = np.array([[0, 2], [3, 1]], dtype=np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    m = (m + 0.5) / (n * n) - 0.5
    return np.tile(m, (h // n + 1, w // n + 1))[:h, :w]


def stretch(L, lo_pct=1.0, hi_pct=99.0, gamma=1.0, contrast=1.0):
    """Normalise lightness to 0..1 using the image's own range."""
    lo, hi = np.percentile(L, lo_pct), np.percentile(L, hi_pct)
    t = np.clip((L - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        t = t ** gamma
    if contrast != 1.0:
        t = np.clip((t - 0.5) * contrast + 0.5, 0, 1)
    return t


def ramp_lookup(t, stops_lab):
    """Piecewise-linear interpolation through the ramp, in OKLab."""
    n = len(stops_lab)
    pos = t * (n - 1)
    i = np.clip(np.floor(pos).astype(np.int32), 0, n - 2)
    f = (pos - i)[..., None]
    f = f * f * (3 - 2 * f)  # smoothstep: softer transitions between stops
    return stops_lab[i] * (1 - f) + stops_lab[i + 1] * f


def resize(img, size, fit):
    if not size:
        return img
    if fit == "stretch":
        return img.resize(size, Image.LANCZOS)
    if fit == "contain":
        return ImageOps.pad(img, size, Image.LANCZOS, color=tuple(int(x * 255) for x in hex_to_srgb(COLORS["bg_dark"])))
    return ImageOps.fit(img, size, Image.LANCZOS)  # cover: fill and centre-crop


# ----------------------------------------------------------------------- modes
def mode_gradient(lab, ramp, a):
    t = stretch(lab[..., 0], gamma=a.gamma, contrast=a.contrast)
    return ramp_lookup(t, ramp)


def mode_hue(lab, ramp, a):
    """Lightness from the ramp; hue snapped to the closest allowed accent colour."""
    t = stretch(lab[..., 0], gamma=a.gamma, contrast=a.contrast)
    base = ramp_lookup(t, ramp)

    acc_names = [n for n in a.accents]
    acc = palette_lab(acc_names)
    acc_h = np.arctan2(acc[:, 2], acc[:, 1])

    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    diff = np.abs(np.angle(np.exp(1j * (hue[..., None] - acc_h[None, None, :]))))
    pick = diff.argmin(axis=-1)

    # how strongly the accent tint applies: grows with the pixel's own chroma
    ref = max(np.percentile(chroma, 90), 1e-4)
    w = np.clip(chroma / ref, 0, 1) ** 0.7 * a.saturation
    tint = acc[pick][..., 1:]                       # accent's (a, b)
    tint = tint * (0.45 + 0.55 * t[..., None])      # darker areas get a calmer tint
    out = base.copy()
    out[..., 1:] = base[..., 1:] * (1 - w[..., None]) + tint * w[..., None]
    return out


def mode_nearest(lab, ramp, a):
    names = a.ramp_names if a.ramp_only else list(COLORS)
    pal = palette_lab(names)
    h, w, _ = lab.shape
    flat = lab.reshape(-1, 3)
    out = np.empty_like(flat)
    for i in range(0, len(flat), 500_000):           # chunked: keeps RAM low at 4K
        c = flat[i:i + 500_000]
        d = ((c[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        out[i:i + 500_000] = pal[d.argmin(axis=1)]
    return out.reshape(h, w, 3)


MODES = {"gradient": mode_gradient, "hue": mode_hue, "nearest": mode_nearest}


# ------------------------------------------------------------------------ main
def process(path, a):
    img = ImageOps.exif_transpose(Image.open(path))
    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA", "PA") else None
    img = resize(img.convert("RGB"), a.size, a.fit)
    if alpha is not None and a.size:
        alpha = resize(alpha, a.size, a.fit)

    src = np.asarray(img, dtype=np.float32) / 255.0
    lab = srgb_to_oklab(src)
    if a.invert:
        lab[..., 0] = 1.0 - lab[..., 0]

    out_lab = MODES[a.mode](lab, palette_lab(a.ramp_names), a)
    out = oklab_to_srgb(out_lab)

    if a.strength < 1.0:
        out = src * (1 - a.strength) + out * a.strength

    if a.grain > 0:  # luminance grain, gives a filmic, less "digital" finish
        rng = np.random.default_rng(a.seed)
        out = out + rng.normal(0, a.grain, out.shape[:2])[..., None]

    out = out * 255.0
    if not a.no_dither:  # ordered dither before 8-bit rounding: kills gradient banding
        out = out + bayer(*out.shape[:2])[..., None]
    result = Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    if alpha is not None:
        result.putalpha(alpha)
    return result


def parse_size(s):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 3840x2160")


def parse_names(s, valid, what):
    names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [n for n in names if n not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown {what}: {', '.join(bad)} (try --list)")
    return names


def main():
    p = argparse.ArgumentParser(description="Recolor images to a fixed palette. Run with --list to see colours and ramps.")
    p.add_argument("inputs", nargs="*", help="image file(s)")
    p.add_argument("-o", "--output", help="output path (single input only; extension picks format)")
    p.add_argument("--mode", choices=MODES, default="gradient")
    p.add_argument("--ramp", default="night",
                   help="preset name or comma-separated colour names, dark->light (default: night)")
    p.add_argument("--accents", default=",".join(ACCENTS),
                   help="hue mode: accent colours allowed (default: all)")
    p.add_argument("--ramp-only", action="store_true", help="nearest mode: restrict to the ramp's colours")
    p.add_argument("--size", type=parse_size, help="output size WxH, e.g. 3840x2160")
    p.add_argument("--fit", choices=["cover", "contain", "stretch"], default="cover",
                   help="how to fit --size when aspect ratios differ (default: cover = crop)")
    p.add_argument("--contrast", type=float, default=1.0, help="gradient/hue contrast (default 1.0)")
    p.add_argument("--gamma", type=float, default=1.0, help="<1 brightens, >1 darkens (default 1.0)")
    p.add_argument("--saturation", type=float, default=1.0, help="hue mode tint strength (default 1.0)")
    p.add_argument("--strength", type=float, default=1.0, help="blend with original, 0..1 (default 1.0)")
    p.add_argument("--grain", type=float, default=0.0, help="film grain amount, try 0.01-0.03")
    p.add_argument("--seed", type=int, default=0, help="grain seed")
    p.add_argument("--invert", action="store_true", help="invert brightness before mapping")
    p.add_argument("--no-dither", action="store_true", help="disable anti-banding dither")
    p.add_argument("--list", action="store_true", help="show colours and ramps, then exit")
    a = p.parse_args()

    if a.list:
        print("colours:", ", ".join(COLORS))
        for k, v in RAMPS.items():
            print(f"ramp {k:7s}: {','.join(v)}")
        return
    if not a.inputs:
        p.error("no input files")
    if a.output and len(a.inputs) > 1:
        p.error("-o only works with a single input")

    try:
        a.ramp_names = RAMPS[a.ramp] if a.ramp in RAMPS else parse_names(a.ramp, COLORS, "colour")
        a.accents = parse_names(a.accents, COLORS, "accent")
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    if len(a.ramp_names) < 2:
        p.error("a ramp needs at least 2 colours")

    for name in a.inputs:
        src = Path(name)
        dst = Path(a.output) if a.output else src.with_name(f"{src.stem}_recolored.png")
        res = process(src, a)
        if dst.suffix.lower() in (".jpg", ".jpeg") and res.mode == "RGBA":
            res = res.convert("RGB")
        res.save(dst, quality=95) if dst.suffix.lower() in (".jpg", ".jpeg") else res.save(dst, optimize=True)
        print(f"{src} -> {dst}  [{res.size[0]}x{res.size[1]}, {a.mode}]")


if __name__ == "__main__":
    main()
