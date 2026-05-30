#!/usr/bin/env python3
"""Procesa escudos: máscara circular, fondo transparente, export 1024×1024 PNG."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ORIGINALS = Path(
    "/home/emmanuz/.cursor/projects/home-emmanuz-Documentos-code-Camba-Cua/assets"
)
OUT_SIZE = 1024


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def is_emblem_pixel(rgb: np.ndarray, thresh: float = 30) -> np.ndarray:
    return luminance(rgb) > thresh


def gold_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    return (
        (r > 115) & (g > 75) & (b < 130) & (r > g) & (g > b * 0.55) & (r + g + b > 120)
    )


def white_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 195) & (g > 195) & (b > 195)


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x * x + y * y
    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = float(c), float(d)
    r = float(math.sqrt(max(e + cx * cx + cy * cy, 1.0)))
    return cx, cy, r


def outer_radius_per_ray(
    mask: np.ndarray, cx: float, cy: float, n: int = 720
) -> np.ndarray:
    """Radio máximo con contenido en cada dirección (evita huecos internos negros)."""
    h, w = mask.shape
    radii = np.zeros(n)
    max_scan = int(min(h, w))
    for i in range(n):
        a = 2 * math.pi * i / n
        maxr = 0
        for rad in range(1, max_scan):
            x = int(round(cx + rad * math.cos(a)))
            y = int(round(cy + rad * math.sin(a)))
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                maxr = rad
        radii[i] = maxr
    return radii


def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def apply_hard_circular_mask(
    rgba: np.ndarray, cx: float, cy: float, radius: float, aa: float = 0.75
) -> np.ndarray:
    """Máscara circular perfecta con antialias mínimo en el borde."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    out = rgba.copy()
    alpha = np.clip((radius - dist) / aa, 0.0, 1.0)
    out[..., 3] = (alpha * 255).astype(np.uint8)

    outside = dist > radius + aa
    out[outside] = [0, 0, 0, 0]

    lum = luminance(out[..., :3])
    halo = (dist > radius - 4) & (dist <= radius + aa) & (lum < 35)
    out[halo] = [0, 0, 0, 0]
    out[halo, 3] = 0

    return out


def apply_circular_mask(
    rgba: np.ndarray, cx: float, cy: float, radius: float, aa: float = 1.2
) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    out = rgba.copy()
    alpha = np.clip((radius - dist) / aa, 0.0, 1.0)
    out[..., 3] = (out[..., 3].astype(float) / 255.0 * alpha * 255).astype(np.uint8)

    outside = dist > radius + aa
    out[outside] = [0, 0, 0, 0]

    # Halo negro del JPEG justo fuera del emblema
    lum = luminance(out[..., :3])
    halo = (dist > radius - 6) & (dist <= radius + aa) & (lum < 35)
    out[halo] = [0, 0, 0, 0]

    return out


def to_square_1024(
    rgba: np.ndarray, cx: float, cy: float, radius: float, hard_circle: bool = False
) -> Image.Image:
    h, w = rgba.shape[:2]
    pad = 2
    left = max(0, int(cx - radius - pad))
    top = max(0, int(cy - radius - pad))
    side = int(2 * (radius + pad))
    right = min(w, left + side)
    bottom = min(h, top + side)

    img = Image.fromarray(rgba)
    crop = img.crop((left, top, right, bottom))
    cw, ch = crop.size
    side_px = max(cw, ch)
    square = Image.new("RGBA", (side_px, side_px), (0, 0, 0, 0))
    square.paste(crop, ((side_px - cw) // 2, (side_px - ch) // 2), crop)

    final = square.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(final)
    yy, xx = np.mgrid[:OUT_SIZE, :OUT_SIZE]
    fcx = fcy = OUT_SIZE / 2.0
    fr = OUT_SIZE / 2.0 - 0.5
    fdist = np.sqrt((xx - fcx) ** 2 + (yy - fcy) ** 2)
    arr[fdist > fr] = [0, 0, 0, 0]

    if hard_circle:
        # Re-enmascarar en el lienzo final: círculo geométrico exacto
        alpha = np.clip((fr - fdist) / 0.75, 0.0, 1.0)
        arr[..., 3] = (alpha * 255).astype(np.uint8)
        arr[fdist > fr + 0.75] = [0, 0, 0, 0]
        validate_circular_alpha(arr, fcx, fcy, fr, "export 1024")

    return Image.fromarray(arr)


def process_camba(src: Path, dst: Path) -> None:
    rgba = load_rgba(src)
    rgb = rgba[..., :3]
    h, w = rgb.shape[:2]

    emblem = is_emblem_pixel(rgb)
    ys, xs = np.where(emblem)
    cx0, cy0 = float(xs.mean()), float(ys.mean())

    radii = outer_radius_per_ray(emblem, cx0, cy0)
    valid = radii[radii > 50]
    radius = float(np.percentile(valid, 97))

    # Refinar centro con puntos del borde exterior
    pts = []
    for i, r in enumerate(radii):
        if r < 50:
            continue
        a = 2 * math.pi * i / len(radii)
        pts.append((cx0 + r * math.cos(a), cy0 + r * math.sin(a)))
    cx, cy, _ = fit_circle(np.array(pts))

    # Borde exterior del anillo blanco/dorado (sin residuo negro JPEG)
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    white = white_mask(rgb) & emblem
    if white.any():
        white_r = dist[white]
        radius = min(float(np.percentile(white_r, 99.5)), float(np.percentile(valid, 98)))
    radius = min(radius, float(np.percentile(outer_radius_per_ray(emblem, cx, cy), 98)))

    rgba = apply_circular_mask(rgba, cx, cy, radius)
    final = to_square_1024(rgba, cx, cy, radius)
    final.save(dst, "PNG", optimize=True)
    print(f"  → {dst.name} ({OUT_SIZE}×{OUT_SIZE}, r={radius:.1f}, center=({cx:.0f},{cy:.0f}))")


def measure_gold_ring(rgba: np.ndarray, cx: float, cy: float) -> dict:
    """Mide el anillo dorado exterior por ángulo para detectar recortes JPEG."""
    rgb = rgba[..., :3]
    gold = gold_mask(rgb)
    h, w = gold.shape

    outer_starts: list[float] = []
    outer_ends: list[float] = []
    outer_widths: list[float] = []

    for i in range(360):
        a = 2 * math.pi * i / 360
        segments: list[tuple[int, int]] = []
        in_g = False
        start = 0
        for rad in range(1, int(min(h, w))):
            x = int(round(cx + rad * math.cos(a)))
            y = int(round(cy + rad * math.sin(a)))
            if 0 <= x < w and 0 <= y < h:
                is_g = gold[y, x]
                if is_g and not in_g:
                    start = rad
                    in_g = True
                elif not is_g and in_g:
                    segments.append((start, rad - 1))
                    in_g = False
        if in_g:
            segments.append((start, int(min(h, w)) - 1))

        if segments:
            s, e = segments[-1]
            outer_starts.append(float(s))
            outer_ends.append(float(e))
            outer_widths.append(float(e - s + 1))

    starts = np.array(outer_starts)
    ends = np.array(outer_ends)
    widths = np.array(outer_widths)

    safe_radius = float(ends.min())

    tight = ends <= safe_radius + 3.0
    pts = []
    for i, (r_end, ok) in enumerate(zip(ends, tight)):
        if not ok:
            continue
        a = 2 * math.pi * i / 360
        pts.append((cx + r_end * math.cos(a), cy + r_end * math.sin(a)))
    if len(pts) >= 20:
        cx, cy, _ = fit_circle(np.array(pts))

    return {
        "cx": cx,
        "cy": cy,
        "safe_radius": safe_radius,
        "ring_width": float(np.median(widths)),
        "outer_starts": starts,
        "outer_ends": ends,
    }


def sample_gold_colors(
    rgb: np.ndarray, gold: np.ndarray
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    samples = rgb[gold].astype(float)
    bright = samples[samples.mean(axis=1) > 140]
    base = bright if len(bright) > 50 else samples
    med = np.median(base, axis=0)
    gc = tuple(int(v) for v in med)
    gc_hi = tuple(min(255, int(v * 1.12)) for v in med)
    gc_lo = tuple(max(0, int(v * 0.82)) for v in med)
    return gc_hi, gc, gc_lo


def draw_perfect_gold_border(
    rgba: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    ring_width: float,
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> np.ndarray:
    """Trazo circular perfecto 360° que cierra el borde dorado recortado."""
    out = rgba.copy()
    h, w = out.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    gc_hi, gc, gc_lo = colors
    stroke = max(3.0, ring_width)
    gap = max(5.0, stroke * 1.1)

    rings = [
        (radius - stroke * 0.5, stroke * 0.55, gc_hi),
        (radius - stroke - gap, stroke * 0.45, gc),
        (radius - stroke - gap - stroke * 0.85, stroke * 0.35, gc_lo),
    ]

    for r_center, half_w, color in rings:
        band = (dist >= r_center - half_w) & (dist <= r_center + half_w)
        out[band, 0] = color[0]
        out[band, 1] = color[1]
        out[band, 2] = color[2]
        out[band, 3] = 255

    return out


def validate_circular_alpha(
    rgba: np.ndarray, cx: float, cy: float, radius: float, label: str
) -> None:
    """Verifica que el alfa forme un círculo perfecto sin bordes planos."""
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    opaque = alpha > 128
    if not opaque.any():
        raise ValueError(f"{label}: sin píxeles opacos")

    boundary_r: list[float] = []
    for i in range(360):
        a = 2 * math.pi * i / 360
        max_r = 0.0
        for rad in range(1, int(min(h, w))):
            x = int(round(cx + rad * math.cos(a)))
            y = int(round(cy + rad * math.sin(a)))
            if 0 <= x < w and 0 <= y < h and opaque[y, x]:
                max_r = float(rad)
        boundary_r.append(max_r)

    br = np.array(boundary_r)
    spread = float(br.max() - br.min())
    cardinals = [float(br[0]), float(br[90]), float(br[180]), float(br[270])]

    if spread > 2.5:
        raise ValueError(
            f"{label}: borde no circular (spread={spread:.2f}px, cardinals={cardinals})"
        )

    outside_leak = opaque & (dist > radius + 1.5)
    if outside_leak.any():
        raise ValueError(f"{label}: {outside_leak.sum()} píxeles fuera del radio")

    print(
        f"  ✓ Validación {label}: círculo OK "
        f"(r≈{br.mean():.1f}px, spread={spread:.2f}px, cardinals={cardinals})"
    )


def process_catedra(src: Path, dst: Path) -> None:
    rgba = load_rgba(src)
    rgb = rgba[..., :3]
    gold = gold_mask(rgb)

    ys, xs = np.where(gold)
    cx0, cy0 = float(xs.mean()), float(ys.mean())

    ring = measure_gold_ring(rgba, cx0, cy0)
    cx, cy = ring["cx"], ring["cy"]
    # Radio inscrito: diámetro más corto, justo antes del achatamiento JPEG
    radius = ring["safe_radius"] - 1.0
    ring_width = ring["ring_width"]
    colors = sample_gold_colors(rgb, gold)

    print(
        f"  · Radio inscrito: {radius:.1f}px (min detectado: {ring['safe_radius']:.1f}px)"
    )

    # Recortar contenido con círculo perfecto (elimina bordes planos heredados)
    rgba = apply_hard_circular_mask(rgba, cx, cy, radius - ring_width - 2)
    # Reconstruir borde dorado completo en el límite exterior
    rgba = draw_perfect_gold_border(rgba, cx, cy, radius, ring_width, colors)
    rgba = apply_hard_circular_mask(rgba, cx, cy, radius, aa=0.75)

    validate_circular_alpha(rgba, cx, cy, radius, "pre-export")

    final = to_square_1024(rgba, cx, cy, radius, hard_circle=True)
    final.save(dst, "PNG", optimize=True)
    print(f"  → {dst.name} ({OUT_SIZE}×{OUT_SIZE}, r={radius:.1f}, center=({cx:.0f},{cy:.0f}))")


def main() -> None:
    sources = {
        "escudo-camba-cua.png": ORIGINALS
        / "WhatsApp_Image_2026-05-29_at_18.55.25-f2d903eb-3c9f-4434-a0c3-5de6246f3b0c.png",
        "escudo-catedra.png": ORIGINALS / "Catedra-e2f8c04c-2cee-435a-906d-e2594a18676c.png",
    }
    ASSETS.mkdir(parents=True, exist_ok=True)

    for out_name, src in sources.items():
        tmp = ASSETS / f"_src_{out_name}"
        shutil.copy(src, tmp)
        dst = ASSETS / out_name
        print(f"Procesando {out_name}...")
        if "camba" in out_name:
            process_camba(tmp, dst)
        else:
            process_catedra(tmp, dst)
        tmp.unlink(missing_ok=True)

    print("Listo.")


if __name__ == "__main__":
    main()
